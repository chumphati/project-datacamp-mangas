import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.model = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs')
        self.pipe = Pipeline([
            ('transformer', self.transformer),
            ('model', self.model)
        ])

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
            new_probabilities[:, :probabilities.shape[1]] = probabilities
            return new_probabilities
