"""Random Forest Model implementation"""
from sklearn.ensemble import RandomForestClassifier
from model.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__('random_forest', random_state)
        self.model = self.create_model()

    def create_model(self, **kwargs):
        """Create Random Forest model"""
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            random_state=self.random_state
        )

    def train(self, X_train, y_train, cv_tuning=True):
        """Train Random Forest with hyperparameter tuning"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        } if cv_tuning else None

        return super().train(X_train, y_train, cv_tuning, param_grid)