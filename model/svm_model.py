"""SVM Model implementation"""
from sklearn.svm import SVC
from model.base_model import BaseModel


class SVMModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__('svm', random_state)
        self.model = self.create_model()

    def create_model(self, **kwargs):
        """Create SVM model with RBF kernel"""
        return SVC(
            C=kwargs.get('C', 1.0),
            kernel=kwargs.get('kernel', 'rbf'),
            gamma=kwargs.get('gamma', 'scale'),
            random_state=self.random_state,
            probability=True
        )

    def train(self, X_train, y_train, cv_tuning=True):
        """Train SVM with hyperparameter tuning"""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf']
        } if cv_tuning else None

        return super().train(X_train, y_train, cv_tuning, param_grid)