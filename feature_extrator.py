import cv2
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, config):
        self.config = config

    def extract_advanced_features(self, images):
        """Extract advanced features from images"""
        print("Extracting advanced features...")

        advanced_features = []
        img_height = self.config.IMG_HEIGHT
        img_width = self.config.IMG_WIDTH

        for img_flat in tqdm(images, desc="Feature extraction"):
            img = img_flat.reshape(img_height, img_width, 3)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            feature_vector = []

            # 1. Histogram features
            hist = cv2.calcHist([img_gray], [0], None, [16], [0, 256])
            feature_vector.extend(hist.flatten())

            # 2. Statistical features
            feature_vector.append(np.mean(img_gray))
            feature_vector.append(np.std(img_gray))
            feature_vector.append(np.median(img_gray))

            # 3. Texture features
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            feature_vector.append(np.mean(sobelx))
            feature_vector.append(np.mean(sobely))
            feature_vector.append(np.std(sobelx))
            feature_vector.append(np.std(sobely))

            # 4. Edge density
            edges = cv2.Canny(img_gray, 50, 150)
            feature_vector.append(np.sum(edges > 0) / edges.size)

            advanced_features.append(feature_vector)

        return np.array(advanced_features)