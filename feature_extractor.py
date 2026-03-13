"""Feature extraction module for chest X-ray images"""
import numpy as np
import cv2
import os
from tqdm import tqdm
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Change these imports
from config import config
from utils import print_header, print_success, print_info, print_warning


class FeatureExtractor:
    """Extracts features from chest X-ray images"""

    def __init__(self, img_height: int = None, img_width: int = None):
        """
        Initialize feature extractor

        Args:
            img_height: Image height
            img_width: Image width
        """
        self.img_height = img_height or config.IMG_HEIGHT
        self.img_width = img_width or config.IMG_WIDTH
        self.scaler = StandardScaler()
        self.pca = None

    def extract_advanced_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract advanced features from images

        Args:
            images: Array of flattened images

        Returns:
            Array of extracted features
        """
        print_header("Extracting Advanced Features")

        advanced_features = []

        for img_flat in tqdm(images, desc="Feature extraction"):
            img = img_flat.reshape(self.img_height, self.img_width, 3)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            feature_vector = []

            # 1. Histogram features
            hist = cv2.calcHist([img_gray], [0], None, [16], [0, 256])
            feature_vector.extend(hist.flatten())

            # 2. Statistical features
            feature_vector.append(np.mean(img_gray))
            feature_vector.append(np.std(img_gray))
            feature_vector.append(np.median(img_gray))

            # 3. Texture features (Sobel gradients)
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            feature_vector.append(np.mean(sobelx))
            feature_vector.append(np.mean(sobely))
            feature_vector.append(np.std(sobelx))
            feature_vector.append(np.std(sobely))

            # 4. Edge density
            edges = cv2.Canny(img_gray, 50, 150)
            feature_vector.append(np.sum(edges > 0) / edges.size)

            # 5. Additional texture features
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
            laplacian = cv2.filter2D(img_gray, -1, kernel)
            feature_vector.append(np.mean(np.abs(laplacian)))
            feature_vector.append(np.std(laplacian))

            advanced_features.append(feature_vector)

        return np.array(advanced_features)

    def extract_hog_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features

        Args:
            images: Array of flattened images

        Returns:
            Array of HOG features
        """
        try:
            from skimage.feature import hog

            hog_features = []

            for img_flat in tqdm(images, desc="HOG extraction"):
                img = img_flat.reshape(self.img_height, self.img_width, 3)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=False)
                hog_features.append(features)

            return np.array(hog_features)
        except ImportError:
            print_warning("scikit-image not installed. Skipping HOG features.")
            return np.array([])

    def extract_lbp_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Patterns (LBP) features

        Args:
            images: Array of flattened images

        Returns:
            Array of LBP features
        """
        try:
            from skimage.feature import local_binary_pattern

            lbp_features = []

            for img_flat in tqdm(images, desc="LBP extraction"):
                img = img_flat.reshape(self.img_height, self.img_width, 3)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                lbp = local_binary_pattern(img_gray, 8, 1, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
                lbp_features.append(hist)

            return np.array(lbp_features)
        except ImportError:
            print_warning("scikit-image not installed. Skipping LBP features.")
            return np.array([])

    def preprocess_data(self, features: np.ndarray, labels: np.ndarray,
                       X_train: np.ndarray, X_test: np.ndarray,
                       use_advanced_features: bool = False,
                       use_pca: bool = False, n_components: int = None) -> Tuple:
        """
        Preprocess data for ML models

        Args:
            features: Raw features
            labels: Labels (not used but kept for API consistency)
            X_train: Training features
            X_test: Test features
            use_advanced_features: Whether to use advanced feature extraction
            use_pca: Whether to apply PCA
            n_components: Number of PCA components

        Returns:
            Tuple of (X_train_processed, X_test_processed)
        """
        print_header("Preprocessing Data")

        n_components = n_components or config.PCA_COMPONENTS

        if use_advanced_features:
            print_info("Using advanced feature extraction")
            X_train_processed = self.extract_advanced_features(X_train)
            X_test_processed = self.extract_advanced_features(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test

        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        X_test_scaled = self.scaler.transform(X_test_processed)

        if use_pca:
            print_info(f"Applying PCA with {n_components} components")
            self.pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
            X_train_final = self.pca.fit_transform(X_train_scaled)
            X_test_final = self.pca.transform(X_test_scaled)
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print_success(f"Explained variance ratio: {explained_variance:.4f}")
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        print_info(f"Training set shape: {X_train_final.shape}")
        print_info(f"Test set shape: {X_test_final.shape}")

        return X_train_final, X_test_final

    def save(self, filepath: str):
        """Save feature extractor state"""
        import joblib
        save_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'img_height': self.img_height,
            'img_width': self.img_width
        }
        joblib.dump(save_data, filepath)
        print_success(f"Feature extractor saved to {filepath}")

    def load(self, filepath: str):
        """Load feature extractor state"""
        import joblib
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature extractor file not found: {filepath}")

        save_data = joblib.load(filepath)
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        self.img_height = save_data['img_height']
        self.img_width = save_data['img_width']
        print_success(f"Feature extractor loaded from {filepath}")