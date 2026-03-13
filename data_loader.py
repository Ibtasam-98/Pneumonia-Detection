"""Data loading and preprocessing module"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from typing import Tuple, Optional, List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Change this line - remove package prefix
from config import config
from utils import print_header, print_success, print_warning, print_error


class ChestXRayDataLoader:
    """Handles loading and preprocessing of chest X-ray images"""

    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader

        Args:
            data_dir: Path to the dataset directory
        """
        self.data_dir = data_dir or config.DEFAULT_DATA_DIR
        self.img_height, self.img_width = config.get_image_size()
        self.class_names = config.CLASS_NAMES
        self.splits = ['train', 'test', 'val']

    def debug_structure(self) -> bool:
        """Debug dataset structure"""
        print_header("Debugging Dataset Structure")

        if not os.path.exists(self.data_dir):
            print_error(f"Main dataset directory not found: {self.data_dir}")
            return False

        table_data = []

        for split in self.splits:
            split_path = os.path.join(self.data_dir, split)

            if os.path.exists(split_path):
                split_data = [split]
                total_images = 0

                for class_name in self.class_names:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        image_files = [f for f in os.listdir(class_path)
                                     if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                        num_images = len(image_files)
                        split_data.append(f"{num_images}")
                        total_images += num_images
                    else:
                        split_data.append("0")

                split_data.append(str(total_images))
                table_data.append(split_data)
            else:
                table_data.append([split, "Not Found", "Not Found", "0"])

        headers = ["Split", "NORMAL", "PNEUMONIA", "Total"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        return True

    def load_images(self, data_dir: str = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and preprocess images from directory

        Args:
            data_dir: Path to dataset directory (overrides instance data_dir)

        Returns:
            Tuple of (features array, labels array)
        """
        if data_dir:
            self.data_dir = data_dir

        print_header("Loading and Preprocessing Images")

        features = []
        labels = []

        for split in self.splits:
            split_path = os.path.join(self.data_dir, split)
            if not os.path.exists(split_path):
                print_warning(f"{split_path} not found, skipping...")
                continue

            for class_name in self.class_names:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    print_warning(f"{class_path} not found, skipping...")
                    continue

                print(f"Processing {split}/{class_name}...")
                image_files = [f for f in os.listdir(class_path)
                             if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

                for image_file in tqdm(image_files, desc=f"{split}/{class_name}"):
                    img_path = os.path.join(class_path, image_file)

                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_width, self.img_height))
                        img_flattened = img.flatten()

                        features.append(img_flattened)
                        labels.append(class_name)

                    except Exception as e:
                        print_warning(f"Error loading {img_path}: {e}")
                        continue

        if len(features) == 0:
            print_error("No images were loaded.")
            return None, None

        features_array = np.array(features)
        labels_array = np.array(labels)

        print_success(f"Loaded {len(features)} images")

        # Print class distribution
        unique, counts = np.unique(labels_array, return_counts=True)
        table_data = []
        for class_name, count in zip(unique, counts):
            percentage = (count / len(labels_array)) * 100
            table_data.append([class_name, count, f"{percentage:.2f}%"])

        print(tabulate(table_data, headers=["Class", "Count", "Percentage"], tablefmt="grid"))

        return features_array, labels_array

    def prepare_data(self, features: np.ndarray, labels: np.ndarray,
                    test_size: float = None, random_state: int = None) -> Tuple:
        """
        Prepare data for training by encoding labels and splitting

        Args:
            features: Feature array
            labels: Label array
            test_size: Test set size (overrides config)
            random_state: Random state (overrides config)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, label_encoder)
        """
        test_size = test_size or config.TEST_SIZE
        random_state = random_state or config.RANDOM_STATE

        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, test_size=test_size,
            random_state=random_state, stratify=y_encoded
        )

        return X_train, X_test, y_train, y_test, le

    def get_class_weights(self, labels: np.ndarray) -> dict:
        """Calculate class weights for imbalanced datasets"""
        unique, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique)

        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = n_samples / (n_classes * count)

        return weights