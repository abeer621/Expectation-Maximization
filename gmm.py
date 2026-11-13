"""Gaussian Mixture Model fitted with the Expectation-Maximization algorithm."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

_LOG_2PI = np.log(2 * np.pi)


class GaussianMixtureEM:
    """Full-covariance Gaussian mixture fitted with EM (Bishop, PRML ch. 9).

    Parameters
    ----------
    n_components : int
        Number of mixture components K.
    init : {"random", "kmeans", "kmeans++"}
        How the initial component means are chosen.
        ``random``   - K distinct data points drawn uniformly.
        ``kmeans``   - Lloyd's K-means started from random data points.
        ``kmeans++`` - Lloyd's K-means started from K-means++ seeding.
        Covariances start as identity and weights as uniform in all cases.
    max_iter : int
        Maximum number of EM iterations per run.
    tol : float
        Stop when the log-likelihood improves by less than ``tol``.
    reg_covar : float
        Ridge added to each covariance diagonal. Guards against the
        singularities of the Gaussian-mixture likelihood (a component
        collapsing onto a single point sends the likelihood to infinity).
    n_init : int
        Number of restarts; the run with the highest log-likelihood is kept.
    random_state : int or None
        Seed for reproducibility.

    Attributes (after ``fit``)
    --------------------------
    weights_, means_, covariances_ : fitted parameters, shapes (K,), (K, D), (K, D, D)
    log_likelihoods_ : log-likelihood trace of the selected run
    n_iter_ : number of EM iterations of the selected run
    converged_ : whether the tolerance was reached before ``max_iter``
    """

    def __init__(self, n_components, *, init="kmeans", max_iter=200, tol=1e-6,
                 reg_covar=1e-6, n_init=1, random_state=None):
        if init not in {"random", "kmeans", "kmeans++"}:
            raise ValueError(f"unknown init {init!r}")
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            run = self._fit_once(X, rng)
            if best is None or run["lls"][-1] > best["lls"][-1]:
                best = run
        self.weights_ = best["weights"]
        self.means_ = best["means"]
        self.covariances_ = best["covs"]
        self.log_likelihoods_ = best["lls"]
        self.converged_ = best["converged"]
        self.n_iter_ = len(best["lls"])
        self.n_features_ = X.shape[1]
        return self

    def _fit_once(self, X, rng):
        N, D = X.shape
        K = self.n_components
        means = self._init_means(X, rng)
        covs = np.tile(np.eye(D), (K, 1, 1))
        weights = np.full(K, 1.0 / K)

        lls, converged = [], False
        for it in range(self.max_iter):
            resp, ll = self._e_step(X, weights, means, covs)
            lls.append(ll)
            if it > 0 and abs(ll - lls[-2]) < self.tol:
                converged = True
                break
            weights, means, covs = self._m_step(X, resp)
        else:
            # loop exhausted: score the parameters produced by the last M-step
            lls.append(self._e_step(X, weights, means, covs)[1])
        return dict(weights=weights, means=means, covs=covs, lls=lls, converged=converged)

    # ---------------------------------------------------------- EM steps
    def _log_weighted_prob(self, X, weights, means, covs):
        """log(pi_k * N(x_n | mu_k, Sigma_k)) as an (N, K) array, via Cholesky."""
        N, D = X.shape
        out = np.empty((N, len(weights)))
        for k in range(len(weights)):
            L = np.linalg.cholesky(covs[k])
            z = solve_triangular(L, (X - means[k]).T, lower=True)
            maha = np.sum(z ** 2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, k] = np.log(weights[k]) - 0.5 * (D * _LOG_2PI + log_det + maha)
        return out

    def _e_step(self, X, weights, means, covs):
        """Responsibilities r_nk and the log-likelihood at the given parameters."""
        log_w = self._log_weighted_prob(X, weights, means, covs)
        log_norm = logsumexp(log_w, axis=1)             # log p(x_n)
        resp = np.exp(log_w - log_norm[:, None])
        return resp, log_norm.sum()

    def _m_step(self, X, resp):
        N, D = X.shape
        Nk = resp.sum(axis=0) + 10 * np.finfo(float).eps  # effective counts
        weights = Nk / N
        means = (resp.T @ X) / Nk[:, None]
        covs = np.empty((self.n_components, D, D))
        for k in range(self.n_components):
            diff = X - means[k]
            covs[k] = (resp[:, [k]] * diff).T @ diff / Nk[k]
            covs[k].flat[:: D + 1] += self.reg_covar
        return weights, means, covs

    # ---------------------------------------------------------- init
    def _init_means(self, X, rng):
        K = self.n_components
        if self.init == "random":
            return X[rng.choice(len(X), K, replace=False)].copy()
        return self._kmeans(X, rng, plus_plus=(self.init == "kmeans++"))

    def _kmeans(self, X, rng, n_iter=20, plus_plus=False):
        K = self.n_components
        if plus_plus:
            centers = [X[rng.integers(len(X))]]
            for _ in range(K - 1):
                d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
                centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
            centers = np.array(centers)
        else:
            centers = X[rng.choice(len(X), K, replace=False)].copy()
        for _ in range(n_iter):
            labels = np.argmin(((X[:, None] - centers[None]) ** 2).sum(-1), axis=1)
            new = np.array([X[labels == k].mean(0) if (labels == k).any() else centers[k]
                            for k in range(K)])
            if np.allclose(centers, new):
                break
            centers = new
        return centers

    # ---------------------------------------------------------- inference
    def predict_proba(self, X):
        return self._e_step(np.asarray(X, float), self.weights_, self.means_, self.covariances_)[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X):
        """Total log-likelihood of X under the fitted model."""
        return self._e_step(np.asarray(X, float), self.weights_, self.means_, self.covariances_)[1]

    # ------------------------------------------------- model selection
    def n_parameters(self, D=None):
        """Free parameters of a full-covariance GMM."""
        D = D or self.n_features_
        K = self.n_components
        return (K - 1) + K * D + K * D * (D + 1) // 2

    def aic(self, X):
        return 2 * self.n_parameters(np.shape(X)[1]) - 2 * self.score(X)

    def bic(self, X):
        N, D = np.shape(X)
        return self.n_parameters(D) * np.log(N) - 2 * self.score(X)
