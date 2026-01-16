import pandas as pd
from typing import List, Dict
from src.types import Distribution
from src.models.gaussian_copula import MarginalDist, GaussianCopula


def fit_distributions(
    data: pd.DataFrame,
    target_cols: List[str],
    cluster_col: str,
    dist_mapper: Dict[str, Distribution]
) -> Dict[str, Dict[str, MarginalDist]]:

    dist_fitted: Dict[str, Dict[str, MarginalDist]] = {}

    for cluster in sorted(data[cluster_col].unique()):
        cluster_key = str(cluster)

        if cluster_key not in dist_mapper:
            raise KeyError(f"No distribution mapping for cluster '{cluster}'.")

        dist = dist_mapper[cluster_key]
        cluster_data = data[data[cluster_col] == cluster]
        cluster_data = cluster_data.dropna()

        fitted = {
            col: MarginalDist(dist, dist.fit(cluster_data[col]))
            for col in target_cols
        }

        dist_fitted[cluster_key] = fitted

    return dist_fitted


def fit_copulas(
    data: pd.DataFrame,
    target_cols: List[str],
    cluster_col: str,
    fitted_dist: Dict[str, Dict[str, MarginalDist]]
) -> Dict[str, GaussianCopula]:

    copulas_fitted: Dict[str, GaussianCopula] = {}

    for cluster in sorted(data[cluster_col].unique()):
        cluster_key = str(cluster)

        if cluster_key not in fitted_dist:
            raise KeyError(f"No fitted distributions for cluster '{cluster}'.")

        container = fitted_dist[cluster_key]
        cluster_data = data[data[cluster_col] == cluster][target_cols]

        copula = GaussianCopula(container)
        copula.fit(cluster_data)

        copulas_fitted[cluster_key] = copula

    return copulas_fitted


def sample_10d_returns(
    fitted_copulas: Dict[str, GaussianCopula],
    size: int,
    seed: int
) -> Dict[str, pd.Series]:

    returns_10d: Dict[str, pd.Series] = {}

    for cluster, copula in fitted_copulas.items():
        returns_1d = copula.sample(size, seed=seed)
        returns_1d = returns_1d + 1
        returns_10d[cluster] = returns_1d.prod(axis=1) - 1

    return returns_10d


def get_var_values(
    mapper: Dict[str, pd.Series],
    cl: float = 0.99,
    interpolation: str = 'linear'
) -> pd.DataFrame:

    var_values = []

    for cluster, returns in mapper.items():
        var_value = returns.quantile(q=(1 - cl), interpolation=interpolation)
        var_values.append((cluster, var_value))

    return pd.DataFrame(var_values, columns=['cluster', f'var_{int(cl * 100)}'])