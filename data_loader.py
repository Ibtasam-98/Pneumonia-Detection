"""Data loading and preprocessing module"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import Config


class DataLoader:
    def __init__(self, config=None):
        self.config = config or Config()
        self.scaler = StandardScaler()
        self.pca = None
        self.features = None
        self.labels = None

    def debug_dataset_structure(self, data_dir=Config.DEFAULT_DATA_DIR):
        """Debug function to check dataset structure"""
        print("\n🔍 Debugging dataset structure...")

        if not os.path.exists(data_dir):
            print(f"❌ Main dataset directory not found: {data_dir}")
            return False

        splits = ['train', 'test', 'val']
        table_data = []

        for split in splits:
            split_path = os.path.join(data_dir, split)

            if os.path.exists(split_path):
                split_data = [split]
                total_images = 0

                for class_name in self.config.CLASS_NAMES:
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

    def load_and_preprocess_images(self, data_dir=Config.DEFAULT_DATA_DIR):
        """Load and preprocess images from directory"""
        print("Loading and preprocessing images...")

        features = []
        labels = []
        splits = ['train', 'test', 'val']

        for split in splits:
            split_path = os.path.join(data_dir, split)
            if not os.path.exists(split_path):
                print(f"Warning: {split_path} not found, skipping...")
                continue

            for class_name in self.config.CLASS_NAMES:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    print(f"Warning: {class_path} not found, skipping...")
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
                        img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
                        img_flattened = img.flatten()

                        features.append(img_flattened)
                        labels.append(class_name)

                    except Exception:
                        continue

        if len(features) == 0:
            print("❌ No images were loaded.")
            return None, None

        self.features = np.array(features)
        self.labels = np.array(labels)

        print(f"\n✅ Loaded {len(features)} images")
        self._print_class_distribution()

        return self.features, self.labels

    def _print_class_distribution(self):
        """Print class distribution table"""
        unique, counts = np.unique(self.labels, return_counts=True)
        table_data = []
        for class_name, count in zip(unique, counts):
            percentage = (count / len(self.labels)) * 100
            table_data.append([class_name, count, f"{percentage:.2f}%"])

        print(tabulate(table_data, headers=["Class", "Count", "Percentage"], tablefmt="grid"))

    def preprocess_data(self, use_advanced_features=False, use_pca=False, n_components=50):
        """Preprocess the data for ML models"""
        print("Preprocessing data...")

        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)

        if use_advanced_features:
            from feature_extractor import FeatureExtractor
            extractor = FeatureExtractor(self.config)
            X_processed = extractor.extract_advanced_features(self.features)
        else:
            X_processed = self.features

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_encoded
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_pca:
            print(f"Applying PCA with {n_components} components...")
            self.pca = PCA(n_components=n_components, random_state=self.config.RANDOM_STATE)
            X_train_final = self.pca.fit_transform(X_train_scaled)
            X_test_final = self.pca.transform(X_test_scaled)
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"Explained variance ratio: {explained_variance:.4f}")
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        print(f"Training set shape: {X_train_final.shape}")
        print(f"Test set shape: {X_test_final.shape}")

        return X_train_final, X_test_final, y_train, y_test

    def get_preprocessors(self):
        """Get scaler and PCA transformers"""
        return self.scaler, self.pca

    def get_features_labels(self):
        """Get features and labels"""
        return self.features, self.labels