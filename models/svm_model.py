"""SVM model implementation"""
from sklearn.svm import SVC

# Change this import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel
from config import config


class SVMModel(BaseModel):
    """Support Vector Machine with RBF kernel"""

    def __init__(self, random_state: int = None):
        """Initialize SVM model"""
        super().__init__('svm', random_state)

    def _create_model(self, **kwargs):
        """Create SVM model"""
        return SVC(random_state=self.random_state, probability=True, **kwargs)

    def _get_param_grid(self) -> dict:
        """Get SVM parameter grid"""
        return config.SVM_PARAM_GRID