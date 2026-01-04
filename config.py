"""Configuration settings for the Chest X-Ray ML Predictor"""
import os


class Config:
    # Image settings
    IMG_HEIGHT = 100
    IMG_WIDTH = 100

    # Class names
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

    # Paths
    DEFAULT_DATA_DIR = 'dataset/chest_xray'
    VISUALIZATION_DIR = "visualizations"
    DEFAULT_MODEL_FILE = 'chest_xray_models.pkl'

    # Model settings
    USE_ADVANCED_FEATURES = False
    USE_PCA = True
    PCA_COMPONENTS = 50

    # Training settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Visualization settings
    MODEL_COLORS = {
        'svm': '#1f77b4',  # Blue
        'knn': '#ff7f0e',  # Orange
        'random_forest': '#2ca02c'  # Green
    }

    # Bootstrap settings
    N_BOOTSTRAPS = 50

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.VISUALIZATION_DIR, exist_ok=True)