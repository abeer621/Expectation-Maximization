import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from em_gmm import GaussianMixtureEM, matched_accuracy
from em_gmm.data import make_synthetic


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n=600, seed=7)


@pytest.mark.parametrize("init", ["random", "kmeans", "kmeans++"])
def test_log_likelihood_is_monotone(data, init):
    X, _ = data
    gmm = GaussianMixtureEM(4, init=init, random_state=0).fit(X)
    diffs = np.diff(gmm.log_likelihoods_)
    assert np.all(diffs > -1e-8), "EM must never decrease the log-likelihood"


def test_parameters_are_valid(data):
    X, _ = data
    gmm = GaussianMixtureEM(4, random_state=0).fit(X)
    assert np.isclose(gmm.weights_.sum(), 1.0)
    assert np.allclose(gmm.predict_proba(X).sum(1), 1.0)
    for cov in gmm.covariances_:
        assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_matches_sklearn_when_started_from_same_means(data):
    X, _ = data
    ours = GaussianMixtureEM(4, init="kmeans", max_iter=500, tol=1e-10, random_state=0).fit(X)
    # start sklearn from the same initial state (identity covs, uniform weights)
    rng = np.random.default_rng(0)
    init_means = ours._kmeans(X, rng)
    ref = GaussianMixture(4, covariance_type="full", means_init=init_means,
                          weights_init=np.full(4, 0.25),
                          precisions_init=np.tile(np.eye(2), (4, 1, 1)),
                          max_iter=500, tol=1e-10, reg_covar=1e-6).fit(X)
    assert ours.score(X) == pytest.approx(ref.score(X) * len(X), rel=1e-4)


def test_recovers_well_separated_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, (100, 2)) for c in ([0, 0], [6, 0], [0, 6])])
    y = np.repeat([0, 1, 2], 100)
    gmm = GaussianMixtureEM(3, init="kmeans++", random_state=0).fit(X)
    acc, _ = matched_accuracy(y, gmm.predict(X))
    assert acc > 0.99


def test_singularity_guard():
    X = np.vstack([np.zeros((20, 2)), np.random.default_rng(0).normal(5, 1, (80, 2))])
    gmm = GaussianMixtureEM(2, random_state=0).fit(X)   # duplicate points -> degenerate component
    assert np.isfinite(gmm.log_likelihoods_[-1])
