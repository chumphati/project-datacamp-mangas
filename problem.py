import os
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
import rampwf as rw

problem_title = 'Anime TV-Shows classification'

Predictions = rw.prediction_types.make_multiclass(label_names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 10 classes

workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]


def _get_data(path=".", split="train"):
    data = pd.read_csv(os.path.join(str(path), "data", split + ".csv"))

    #all selected columns
    col = ["Members", "Score", "Episodes"]
    data_sel = data[col].copy()

    #episodes
    data_sel["Episodes"] = pd.to_numeric(data_sel["Episodes"], errors="coerce")
    data_sel = data_sel.dropna(subset=["Episodes"]).copy()

    #members
    data_sel["Members"] = pd.to_numeric(data_sel["Members"], errors="coerce")
    data_sel = data_sel.dropna(subset=["Members"]).copy()

    #target: take the floor of the score to get categories
    data_sel["Score"] = pd.to_numeric(data_sel["Score"], errors="coerce").apply(np.floor)
    median_score = data_sel["Score"].median()
    data_sel["Score"].fillna(median_score, inplace=True) #remplacer les na par la médiane
    data_sel["Score"] = np.maximum(data_sel["Score"] - 2, 0) #est ce que normaliser ne casse pas le score ?

    X = data_sel.drop(columns=["Score"]).to_numpy()
    y = data_sel["Score"].values


    return X, y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    return cv.split(X, y)