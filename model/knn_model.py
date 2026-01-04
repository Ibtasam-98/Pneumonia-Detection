"""KNN Model implementation"""
from sklearn.neighbors import KNeighborsClassifier
from model.base_model import BaseModel


class KNNModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__('knn', random_state)
        self.model = self.create_model()

    def create_model(self, **kwargs):
        """Create KNN model"""
        return KNeighborsClassifier(
            n_neighbors=kwargs.get('n_neighbors', 5),
            weights=kwargs.get('weights', 'uniform'),
            metric=kwargs.get('metric', 'euclidean')
        )

    def train(self, X_train, y_train, cv_tuning=True):
        """Train KNN with hyperparameter tuning"""
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        } if cv_tuning else None

        return super().train(X_train, y_train, cv_tuning, param_grid)