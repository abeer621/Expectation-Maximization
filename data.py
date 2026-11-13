"""Datasets: a synthetic overlapping-Gaussian mixture and the Mall Customers data."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

TRUE_MEANS = np.array([[0.0, 0.0], [2.5, 0.5], [1.2, 3.0], [3.8, 2.8]])
TRUE_COVS = np.array([
    [[1.2, 0.7], [0.7, 0.9]],
    [[1.0, -0.5], [-0.5, 1.1]],
    [[0.8, 0.3], [0.3, 1.4]],
    [[1.1, 0.6], [0.6, 0.9]],
])
TRUE_WEIGHTS = np.array([0.30, 0.25, 0.25, 0.20])

MALL_URL = ("https://raw.githubusercontent.com/kennedykwangari/"
            "Mall-Customer-Segmentation-Data/master/Mall_Customers.csv")


def make_synthetic(n=600, seed=0):
    """Sample ``n`` points from four overlapping 2-D Gaussians. Returns (X, labels)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, (mu, cov, pi) in enumerate(zip(TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS)):
        n_k = int(round(n * pi))
        X.append(rng.multivariate_normal(mu, cov, n_k))
        y.append(np.full(n_k, k))
    X, y = np.vstack(X), np.concatenate(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_mall_customers(path_or_url=MALL_URL):
    """Return the Mall Customers DataFrame (local CSV path or URL)."""
    if str(path_or_url).startswith("http"):
        import requests
        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    else:
        df = pd.read_csv(path_or_url)
    df.columns = df.columns.str.strip()
    return df
