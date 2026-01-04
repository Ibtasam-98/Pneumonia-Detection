"""Prediction module"""
import cv2
import numpy as np
import time
import os
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')

from config import Config


class Predictor:
    def __init__(self, config=None):
        self.config = config or Config()

    def predict_single_image(self, image_path, model, scaler, pca, class_names):
        """Predict a single image using specified model"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        img_flattened = img.flatten().reshape(1, -1)

        img_scaled = scaler.transform(img_flattened)

        if pca:
            img_processed = pca.transform(img_scaled)
        else:
            img_processed = img_scaled

        # Measure inference time
        start_time = time.perf_counter()
        prediction = model.predict(img_processed)[0]
        inference_time = (time.perf_counter() - start_time) * 1000

        prediction_proba = model.predict_proba(img_processed)[0] if hasattr(model, 'predict_proba') else [0, 0]

        result = class_names[prediction]
        confidence = prediction_proba[prediction] if hasattr(model, 'predict_proba') else 1.0

        print(f"\n{'=' * 60}")
        print("PREDICTION RESULT")
        print(f"{'=' * 60}")

        pred_table = [
            ["Image", os.path.basename(image_path)],
            ["Model", type(model).__name__],
            ["Prediction", result],
            ["Confidence", f"{confidence:.4f}"],
            ["Inference Time", f"{inference_time:.2f} ms"]
        ]

        if hasattr(model, 'predict_proba'):
            pred_table.append(
                ["Probabilities", f"NORMAL={prediction_proba[0]:.4f}, PNEUMONIA={prediction_proba[1]:.4f}"])

        print(tabulate(pred_table, tablefmt="grid"))
        print(f"{'=' * 60}")

        return result, confidence, inference_time