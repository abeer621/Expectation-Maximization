"""Choosing K with AIC and BIC on the synthetic data (true K = 4)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from em_gmm import GaussianMixtureEM, ari, matched_accuracy
from em_gmm.data import make_synthetic
from em_gmm.plotting import plot_clusters

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
K_TRUE, K_RANGE = 4, range(1, 9)
X, y = make_synthetic(n=600, seed=7)


def sweep(X, seed=42, n_init=5):
    return {K: GaussianMixtureEM(K, init="kmeans", max_iter=300, n_init=n_init,
                                 random_state=seed).fit(X) for K in K_RANGE}


models = sweep(X)
aic = {K: m.aic(X) for K, m in models.items()}
bic = {K: m.bic(X) for K, m in models.items()}
K_aic, K_bic = min(aic, key=aic.get), min(bic, key=bic.get)
print(f"AIC selects K={K_aic}   BIC selects K={K_bic}")
print(f"{'K':>3}{'log-lik':>11}{'AIC':>10}{'BIC':>10}")
for K in K_RANGE:
    print(f"{K:>3}{models[K].score(X):>11.1f}{aic[K]:>10.1f}{bic[K]:>10.1f}")

# Robustness: how often does each criterion pick K=4 across datasets?
picks = {"aic": [], "bic": []}
for s in range(30):
    Xs, _ = make_synthetic(n=600, seed=100 + s)
    ms = {K: GaussianMixtureEM(K, init="kmeans", max_iter=300, n_init=3,
                               random_state=0).fit(Xs) for K in K_RANGE}
    picks["aic"].append(min(ms, key=lambda K: ms[K].aic(Xs)))
    picks["bic"].append(min(ms, key=lambda K: ms[K].bic(Xs)))
print("\nK selected over 30 fresh synthetic datasets (n=600):")
for c, v in picks.items():
    vals, counts = np.unique(v, return_counts=True)
    print(f"  {c.upper()}: " + ", ".join(f"K={a}: {b}" for a, b in zip(vals, counts)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
Ks = list(K_RANGE)
axes[0].plot(Ks, [aic[k] for k in Ks], "o-", c="#f4a261", label="AIC")
axes[0].plot(Ks, [bic[k] for k in Ks], "s-", c="#2a9d8f", label="BIC")
axes[0].axvline(K_TRUE, ls=":", c="grey", label="true K")
axes[0].set(xlabel="K", ylabel="criterion (lower is better)", title="Information criteria"); axes[0].legend()
for ax, (name, K) in zip(axes[1:], (("AIC", K_aic), ("BIC", K_bic))):
    m = models[K]; pred = m.predict(X)
    plot_clusters(ax, X, pred, m, f"{name}-selected K={K}  (ARI={ari(y, pred):.2f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "select_k.png", dpi=150)
