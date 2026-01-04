"""Base model class"""
import time
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV


class BaseModel(ABC):
    def __init__(self, model_name, random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.training_time = 0
        self.best_params = None
        self.best_score = 0

    @abstractmethod
    def create_model(self, **kwargs):
        """Create the model instance"""
        pass

    def train(self, X_train, y_train, cv_tuning=True, param_grid=None):
        """Train the model"""
        start_time = time.time()
        print(f"\n{'=' * 50}")
        print(f"TRAINING {self.model_name.upper()}")
        print(f"{'=' * 50}")

        if cv_tuning and param_grid:
            grid_search = GridSearchCV(
                self.model, param_grid,
                cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_

            print(f"Best parameters: {self.best_params}")
            print(f"Best cross-validation score: {self.best_score:.4f}")
        else:
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")

        return self.model

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities if available"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def get_model(self):
        """Get the trained model"""
        return self.model

    def get_training_time(self):
        """Get training time"""
        return self.training_time

    def get_model_name(self):
        """Get model name"""
        return self.model_name