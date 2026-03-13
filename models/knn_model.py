"""K-Nearest Neighbors model implementation"""
from sklearn.neighbors import KNeighborsClassifier

from models.base_model import BaseModel
from config import config


class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier"""

    def __init__(self, random_state: int = None):
        """Initialize KNN model"""
        super().__init__('knn', random_state)

    def _create_model(self, **kwargs):
        """Create KNN model"""
        return KNeighborsClassifier(**kwargs)

    def _get_param_grid(self) -> dict:
        """Get KNN parameter grid"""
        return config.KNN_PARAM_GRID