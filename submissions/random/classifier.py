import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.dummy import DummyClassifier
import warnings

warnings.filterwarnings("ignore")


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    IterativeImputer(
                        random_state=42, max_iter=10, skip_complete=True
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        self.model = DummyClassifier()  # Use of DummyClassifier
        self.pipe = Pipeline(
            [("transformer", self.transformer), ("model", self.model)]
        )

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        probabilities = self.pipe.predict_proba(X)
        if probabilities.shape[1] == 10:
            return probabilities
        else:
            new_probabilities = np.zeros((probabilities.shape[0], 10))
            new_probabilities[:, : probabilities.shape[1]] = probabilities
            return new_probabilities
