"""Base model class for all ML models"""
from abc import ABC, abstractmethod
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# Change these imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils import print_header, print_success, print_info


class BaseModel(ABC):
    """Abstract base class for all ML models"""

    def __init__(self, name: str, random_state: int = None):
        """
        Initialize base model

        Args:
            name: Model name
            random_state: Random state for reproducibility
        """
        self.name = name
        self.random_state = random_state or config.RANDOM_STATE
        self.model = None
        self.training_time = 0
        self.best_params = None
        self.cv_scores = None

    @abstractmethod
    def _create_model(self, **kwargs):
        """Create the underlying sklearn model"""
        pass

    @abstractmethod
    def _get_param_grid(self) -> dict:
        """Get parameter grid for grid search"""
        pass

    def train(self, X_train, y_train, cv_tuning: bool = True, verbose: bool = True):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
            cv_tuning: Whether to perform hyperparameter tuning
            verbose: Whether to print progress

        Returns:
            Trained model
        """
        if verbose:
            print_header(f"Training {self.name.upper()}")

        start_time = time.time()

        if cv_tuning:
            if verbose:
                print_info("Performing hyperparameter tuning...")

            param_grid = self._get_param_grid()
            base_model = self._create_model()

            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            if verbose:
                print_success(f"Best parameters: {grid_search.best_params_}")
                print_success(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            self.model = self._create_model()
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time

        if verbose:
            print_success(f"Training completed in {self.training_time:.2f} seconds")

        # Perform cross-validation
        self._cross_validate(X_train, y_train, verbose)

        return self.model

    def _cross_validate(self, X_train, y_train, verbose: bool = True):
        """Perform cross-validation"""
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=self.random_state)
        self.cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

        if verbose:
            print_info(f"Cross-validation scores: {self.cv_scores}")
            print_success(f"CV Mean: {np.mean(self.cv_scores):.4f} ± {np.std(self.cv_scores):.4f}")

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support probability prediction")

    def get_params(self):
        """Get model parameters"""
        if self.model is None:
            return {}
        return self.model.get_params()

    def measure_inference_time(self, X_test, num_runs: int = None) -> float:
        """
        Measure average inference time per sample

        Args:
            X_test: Test features
            num_runs: Number of inference runs

        Returns:
            Average inference time in milliseconds
        """
        if self.model is None:
            return 0

        num_runs = num_runs or config.INFERENCE_RUNS

        # Warm up
        self.model.predict(X_test[:1])

        # Measure inference time
        start_time = time.perf_counter()
        for _ in range(num_runs):
            self.model.predict(X_test[:1])
        total_time = time.perf_counter() - start_time

        return (total_time / num_runs) * 1000  # Convert to ms

    def save(self, filepath: str):
        """Save model to file"""
        import joblib
        save_data = {
            'name': self.name,
            'model': self.model,
            'training_time': self.training_time,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'random_state': self.random_state
        }
        joblib.dump(save_data, filepath)
        print_success(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file"""
        import joblib

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        save_data = joblib.load(filepath)
        self.name = save_data['name']
        self.model = save_data['model']
        self.training_time = save_data['training_time']
        self.best_params = save_data['best_params']
        self.cv_scores = save_data['cv_scores']
        self.random_state = save_data['random_state']

        print_success(f"Model loaded from {filepath}")