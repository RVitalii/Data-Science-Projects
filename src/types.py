import pandas as pd
import numpy as np

from numbers import Real
from typing import Protocol, Sequence, Union


ArrayLike = Union[
    Sequence[Real],
    np.ndarray,
    pd.DataFrame,
    pd.Series
]


class Distribution(Protocol):
    def pdf(self, x: ArrayLike, *args) -> ArrayLike: ...
    def cdf(self, x: ArrayLike, *args) -> ArrayLike: ...
    def ppf(self, q: ArrayLike, *args) -> ArrayLike: ...


class Estimator(Protocol):
    def fit(self, x: ArrayLike, y: ArrayLike) -> "Estimator": ...