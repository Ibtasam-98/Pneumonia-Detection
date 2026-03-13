"""Main predictor class that orchestrates all components"""
import os
import cv2
import numpy as np
import joblib
from typing import Optional, Tuple, Dict
from tabulate import tabulate

# Change these imports
from config import config
from data_loader import ChestXRayDataLoader
from feature_extractor import FeatureExtractor
from models.svm_model import SVMModel
from models.knn_model import KNNModel
from models.random_forest_model import RandomForestModel
from evaluator import ModelEvaluator
from visualizer import MetricsVisualizer
from utils import (
    print_header, print_success, print_error,
    print_info, print_warning, TimingManager
)


class ChestXRayMLPredictor:
    """Main class for Chest X-Ray Pneumonia Detection using ML"""

    def __init__(self, img_height: int = None, img_width: int = None):
        """
        Initialize the predictor

        Args:
            img_height: Height for image resizing
            img_width: Width for image resizing
        """
        self.img_height = img_height or config.IMG_HEIGHT
        self.img_width = img_width or config.IMG_WIDTH

        # Initialize components
        self.data_loader = ChestXRayDataLoader()
        self.feature_extractor = FeatureExtractor(self.img_height, self.img_width)
        self.evaluator = ModelEvaluator(config.CLASS_NAMES)
        self.visualizer = MetricsVisualizer(config.CLASS_NAMES, config.VISUALIZATION_DIR)

        # Models dictionary
        self.models = {}

        # Results storage
        self.results = {}
        self.metrics_history = {}
        self.training_times = {}
        self.inference_times = {}

        # Create necessary directories
        config.create_directories()

    def debug_dataset_structure(self, data_dir: str = None) -> bool:
        """Debug dataset structure"""
        if data_dir:
            self.data_loader.data_dir = data_dir
        return self.data_loader.debug_structure()

    def load_and_preprocess_images(self, data_dir: str = None) -> Tuple:
        """Load and preprocess images"""
        features, labels = self.data_loader.load_images(data_dir)
        return features, labels

    def prepare_data(self, features: np.ndarray, labels: np.ndarray,
                    use_advanced_features: bool = False,
                    use_pca: bool = False, n_components: int = None) -> Tuple:
        """Prepare data for training"""
        X_train, X_test, y_train, y_test, _ = self.data_loader.prepare_data(features, labels)

        X_train_processed, X_test_processed = self.feature_extractor.preprocess_data(
            features, labels, X_train, X_test,
            use_advanced_features=use_advanced_features,
            use_pca=use_pca, n_components=n_components
        )

        return X_train_processed, X_test_processed, y_train, y_test

    def train_svm(self, X_train, y_train, cv_tuning: bool = True):
        """Train SVM model"""
        model = SVMModel(random_state=config.RANDOM_STATE)
        model.train(X_train, y_train, cv_tuning)
        self.models['svm'] = model
        self.training_times['svm'] = model.training_time
        return model

    def train_knn(self, X_train, y_train, cv_tuning: bool = True):
        """Train KNN model"""
        model = KNNModel(random_state=config.RANDOM_STATE)
        model.train(X_train, y_train, cv_tuning)
        self.models['knn'] = model
        self.training_times['knn'] = model.training_time
        return model

    def train_random_forest(self, X_train, y_train, cv_tuning: bool = True):
        """Train Random Forest model"""
        model = RandomForestModel(random_state=config.RANDOM_STATE)
        model.train(X_train, y_train, cv_tuning)
        self.models['random_forest'] = model
        self.training_times['random_forest'] = model.training_time
        return model

    def measure_all_inference_times(self, X_test):
        """Measure inference times for all models"""
        print_header("Measuring Inference Times")

        for model_name, model in self.models.items():
            inference_time = model.measure_inference_time(X_test, config.INFERENCE_RUNS)
            self.inference_times[model_name] = inference_time
            print_info(f"{model_name.upper()}: {inference_time:.2f} ms/image")

    def print_timing_summary(self):
        """Print timing summary table"""
        if not self.training_times or not self.inference_times:
            print_warning("No timing data available.")
            return

        print_header("Model Training and Inference Timing Summary")

        table_data = []
        for model_name in self.models.keys():
            train_time = self.training_times.get(model_name, 0)
            inf_time = self.inference_times.get(model_name, 0)
            table_data.append([
                model_name.upper(),
                f"{train_time:.2f} s",
                f"{inf_time:.2f} ms"
            ])

        print(tabulate(table_data,
                      headers=["Model", "Training Time", "Inference Time/Image"],
                      tablefmt="grid"))

    def evaluate_all_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all models comprehensively"""
        print_header("Comprehensive Model Evaluation")

        # Measure inference times first
        self.measure_all_inference_times(X_test)

        results = {}

        for model_name, model in self.models.items():
            print(f"\n{'=' * 80}")
            print(f"EVALUATING {model_name.upper()}")
            print(f"{'=' * 80}")

            # Calculate all metrics
            metrics = self.evaluator.calculate_all_metrics(
                model.model, X_train, X_test, y_train, y_test,
                model_name,
                training_time=self.training_times.get(model_name, 0),
                inference_time=self.inference_times.get(model_name, 0)
            )

            # Print all metrics
            self.evaluator.print_all_metrics_terminal(metrics)

            # Print confusion matrix
            self.evaluator.print_confusion_matrix(metrics['confusion_matrix'], model_name)

            # Print classification report
            y_pred = model.predict(X_test)
            self.evaluator.print_classification_report(y_test, y_pred, model_name)

            # Print learning curves
            self.evaluator.print_learning_curves(model.model, X_train, y_train, model_name)

            # Print bootstrapping metrics
            self.evaluator.print_bootstrapping_metrics(model.model, X_train, y_train, model_name)

            results[model_name] = metrics
            self.metrics_history = self.evaluator.metrics_history

        # Print timing summary
        self.print_timing_summary()

        # Create consolidated visualization
        self.visualizer.create_consolidated_visualization(self.metrics_history)

        # Print summary table
        self.evaluator.print_summary_table()

        return results

    def predict_single_image(self, image_path: str, model_name: str = 'svm') -> Tuple:
        """
        Predict a single image using specified model

        Args:
            image_path: Path to the image
            model_name: Name of the model to use

        Returns:
            Tuple of (prediction, confidence, inference_time)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img_flattened = img.flatten().reshape(1, -1)

        # Apply preprocessing
        img_scaled = self.feature_extractor.scaler.transform(img_flattened)

        if self.feature_extractor.pca:
            img_processed = self.feature_extractor.pca.transform(img_scaled)
        else:
            img_processed = img_scaled

        # Predict
        model = self.models[model_name]

        inference_time = model.measure_inference_time(img_processed, num_runs=1)
        prediction = model.predict(img_processed)[0]

        if hasattr(model.model, 'predict_proba'):
            prediction_proba = model.predict_proba(img_processed)[0]
            confidence = prediction_proba[prediction]
            proba_normal = prediction_proba[0]
            proba_pneumonia = prediction_proba[1]
        else:
            confidence = 1.0
            proba_normal = proba_pneumonia = None

        result = config.CLASS_NAMES[prediction]

        print(f"\n{'=' * 60}")
        print("PREDICTION RESULT")
        print(f"{'=' * 60}")

        pred_table = [
            ["Image", os.path.basename(image_path)],
            ["Model", model_name.upper()],
            ["Prediction", result],
            ["Confidence", f"{confidence:.4f}"],
            ["Inference Time", f"{inference_time:.2f} ms"]
        ]

        if proba_normal is not None:
            pred_table.append(
                ["Probabilities", f"NORMAL={proba_normal:.4f}, PNEUMONIA={proba_pneumonia:.4f}"]
            )

        print(tabulate(pred_table, tablefmt="grid"))
        print(f"{'=' * 60}")

        return result, confidence, inference_time

    def save_models(self, filename: str = None):
        """Save all models and preprocessors"""
        filename = filename or config.DEFAULT_MODEL_FILE

        save_data = {
            'models': {name: model.model for name, model in self.models.items()},
            'model_objects': self.models,
            'feature_extractor': self.feature_extractor,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'class_names': config.CLASS_NAMES,
            'metrics_history': self.metrics_history,
            'training_times': self.training_times,
            'inference_times': self.inference_times
        }

        joblib.dump(save_data, filename)
        print_success(f"Models and metrics saved to {filename}")

    def load_models(self, filename: str = None):
        """Load models and preprocessors"""
        filename = filename or config.DEFAULT_MODEL_FILE

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        save_data = joblib.load(filename)

        # Load models
        self.models = save_data.get('model_objects', {})

        # If model_objects not found, reconstruct from stored models
        if not self.models and 'models' in save_data:
            from models.svm_model import SVMModel
            from models.knn_model import KNNModel
            from models.random_forest_model import RandomForestModel

            for name, model_obj in save_data['models'].items():
                if name == 'svm':
                    model = SVMModel()
                elif name == 'knn':
                    model = KNNModel()
                elif name == 'random_forest':
                    model = RandomForestModel()
                else:
                    continue

                model.model = model_obj
                self.models[name] = model

        # Load feature extractor
        if 'feature_extractor' in save_data:
            self.feature_extractor = save_data['feature_extractor']
        else:
            # Try to load from individual components
            self.feature_extractor.scaler = save_data.get('scaler', self.feature_extractor.scaler)
            self.feature_extractor.pca = save_data.get('pca', self.feature_extractor.pca)

        self.img_height = save_data.get('img_height', self.img_height)
        self.img_width = save_data.get('img_width', self.img_width)
        self.metrics_history = save_data.get('metrics_history', {})
        self.training_times = save_data.get('training_times', {})
        self.inference_times = save_data.get('inference_times', {})

        print_success(f"Models loaded from {filename}")

        # Print available models
        table_data = [[model_name, type(model.model).__name__]
                     for model_name, model in self.models.items()]
        print(tabulate(table_data, headers=["Model Name", "Type"], tablefmt="grid"))