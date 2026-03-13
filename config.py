"""Configuration settings for the Chest X-Ray ML Predictor"""
import os
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Config:
    """Configuration class for the project"""

    # Image parameters
    IMG_HEIGHT: int = 100
    IMG_WIDTH: int = 100

    # Data paths
    DEFAULT_DATA_DIR: str = 'dataset/chest_xray'
    DEFAULT_MODEL_FILE: str = 'chest_xray_models.pkl'
    VISUALIZATION_DIR: str = 'visualizations'

    # Class names
    CLASS_NAMES: List[str] = None  # Will be initialized in __post_init__

    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2

    # SVM parameters
    SVM_PARAM_GRID: dict = None  # Will be initialized in __post_init__

    # KNN parameters
    KNN_PARAM_GRID: dict = None  # Will be initialized in __post_init__

    # Random Forest parameters
    RF_PARAM_GRID: dict = None  # Will be initialized in __post_init__

    # PCA parameters
    PCA_COMPONENTS: int = 50

    # Cross-validation
    CV_FOLDS: int = 5

    # Bootstrapping
    BOOTSTRAP_ITERATIONS: int = 50

    # Inference timing
    INFERENCE_RUNS: int = 100

    # Color scheme for visualizations
    MODEL_COLORS: dict = None  # Will be initialized in __post_init__

    def __post_init__(self):
        """Initialize mutable defaults"""
        self.CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

        self.SVM_PARAM_GRID = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf']
        }

        self.KNN_PARAM_GRID = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        self.RF_PARAM_GRID = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        self.MODEL_COLORS = {
            'svm': '#1f77b4',  # Blue
            'knn': '#ff7f0e',  # Orange
            'random_forest': '#2ca02c'  # Green
        }

    def get_image_size(self) -> Tuple[int, int]:
        """Get image dimensions"""
        return self.IMG_HEIGHT, self.IMG_WIDTH

    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.VISUALIZATION_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.DEFAULT_MODEL_FILE) if os.path.dirname(self.DEFAULT_MODEL_FILE) else '.',
                    exist_ok=True)


# Global configuration instance
config = Config()