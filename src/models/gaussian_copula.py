import pandas as pd
import numpy as np

from typing import (
    Sequence, 
    Dict,
    Optional
)
from scipy.stats import (
    norm, 
    multivariate_normal
)
from sklearn.covariance import LedoitWolf
from src.types import Distribution, ArrayLike


class MarginalDist:
    def __init__(
        self, 
        dist: Distribution, 
        params: Sequence[float], 
        name: Optional[str] = None
    ) -> None:
        self._dist = dist
        self._params = tuple(params)
        self.name = name or dist.__class__.__name__

    def pdf(self, x: ArrayLike) -> ArrayLike:
         return self._dist.pdf(x, *self._params)

    def cdf(self, x: ArrayLike) -> ArrayLike:
        return self._dist.cdf(x, *self._params)

    def ppf(self, q: ArrayLike) -> ArrayLike:
        return self._dist.ppf(q, *self._params)

    def __repr__(self) -> str:
        return f"MarginalDist(name={self.name}, dist={self._dist}, params={self._params})"


class GaussianCopula:
    def __init__(self, container: Dict[str, MarginalDist]) -> None:
        self._container = container
        self._columns = list(self._container.keys())

    def fit(self, data: pd.DataFrame) -> None:
        missing = set(self._container) - set(data.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if data[list(self._container)].isnull().any().any():
            raise ValueError("NaNs detected in input data")

        self._fit_marginals(data)
        self._fit_dependence()

    def _fit_marginals(self, data: pd.DataFrame) -> None:
        uniform = {
            col: np.clip(self._container[col].cdf(data[col]), 1e-10, 1 - 1e-10)
            for col in self._container
        }
        self._uniform = pd.DataFrame(uniform)

    def _fit_dependence(self) -> None:
        self._standard_normal = pd.DataFrame(
            norm.ppf(self._uniform),
            columns=self._uniform.columns
        )
        self._mean = np.zeros(len(self._container))
        self._cov_matrix = (
            LedoitWolf(assume_centered=True)
            .fit(self._standard_normal)
            .covariance_
        )

    def sample(
        self, 
        size: int, 
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        if not hasattr(self, "_cov_matrix"):
            raise RuntimeError("GaussianCopula must be fitted before sampling")
        
        samples = multivariate_normal.rvs(
            mean=self._mean,
            cov=self._cov_matrix,
            size=size,
            random_state=seed
        )
        samples = np.atleast_2d(samples)
        std_norm_df = pd.DataFrame(samples, columns=self._columns)
        uniform_samples = pd.DataFrame(
            norm.cdf(std_norm_df).clip(1e-10, 1 - 1e-10),
            columns=std_norm_df.columns,
            index=std_norm_df.index
        )
        out = {
            col: self._container[col].ppf(uniform_samples[col])
            for col in self._container
        }

        return pd.DataFrame(out, columns=self._columns)