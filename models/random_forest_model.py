"""Random Forest model implementation"""
from sklearn.ensemble import RandomForestClassifier

from models.base_model import BaseModel
from config import config


class RandomForestModel(BaseModel):
    """Random Forest classifier"""

    def __init__(self, random_state: int = None):
        """Initialize Random Forest model"""
        super().__init__('random_forest', random_state)

    def _create_model(self, **kwargs):
        """Create Random Forest model"""
        return RandomForestClassifier(random_state=self.random_state, **kwargs)

    def _get_param_grid(self) -> dict:
        """Get Random Forest parameter grid"""
        return config.RF_PARAM_GRID