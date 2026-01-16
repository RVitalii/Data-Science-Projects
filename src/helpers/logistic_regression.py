import numpy as np

from typing import List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from src.types import Estimator, ArrayLike


def get_pca_column_names(n: int) -> List[str]:
    return [f"PCA_{i}" for i in range(1, n + 1)]


def get_optimal_pca_components(
    x: ArrayLike,
    y: ArrayLike,
    model: Estimator,
    cv: int = 5,
    scoring: str = "accuracy",
    debug: bool = False
) -> int:
    best_score = -np.inf
    best_components = 0

    x_arr = np.asarray(x)
    n_features = x_arr.shape[1]

    for n in range(1, n_features + 1):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
            ("model", clone(model))
        ])

        score = cross_val_score(pipeline, x_arr, y, cv=cv, scoring=scoring).mean()

        if debug:
            print(f"n_components={n}, CV Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_components = n

    return best_components