"""Random vs K-means initialisation on the synthetic overlapping-Gaussian data.

Part 1 reproduces the single-run comparison from the report.
Part 2 repeats it over many seeds, since a single run says little about a
method whose whole weakness is sensitivity to the starting point.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from em_gmm import GaussianMixtureEM, ari, matched_accuracy
from em_gmm.data import make_synthetic
from em_gmm.plotting import plot_clusters

K, N_SEEDS = 4, 50
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
X, y = make_synthetic(n=600, seed=7)


def evaluate(init, seed):
    m = GaussianMixtureEM(K, init=init, max_iter=200, tol=1e-6, random_state=seed).fit(X)
    pred = m.predict(X)
    return m, matched_accuracy(y, pred)[0], ari(y, pred)


# ---- Part 1: single run (same seed for both, so only the init differs) ----
print("Single run (seed=7)")
print(f"{'':22}{'random':>10}{'kmeans':>10}{'kmeans++':>10}")
runs = {i: evaluate(i, 7) for i in ("random", "kmeans", "kmeans++")}
for label, f in [("Matched accuracy", lambda r: r[1]), ("Adjusted Rand Index", lambda r: r[2]),
                 ("Final log-likelihood", lambda r: r[0].log_likelihoods_[-1]),
                 ("Iterations", lambda r: r[0].n_iter_)]:
    print(f"{label:22}" + "".join(f"{f(runs[i]):>10.3f}" if label != "Iterations"
                                   else f"{f(runs[i]):>10d}" for i in runs))

# ---- Part 2: many seeds ----
print(f"\n{N_SEEDS} seeds (mean ± std)")
summary = {}
for init in ("random", "kmeans", "kmeans++"):
    res = [evaluate(init, s) for s in range(N_SEEDS)]
    summary[init] = dict(
        acc=[r[1] for r in res], ari=[r[2] for r in res],
        ll=[r[0].log_likelihoods_[-1] for r in res], it=[r[0].n_iter_ for r in res],
        conv=[r[0].converged_ for r in res])
print(f"{'init':10}{'acc':>16}{'ARI':>16}{'log-lik':>18}{'iters':>14}{'converged':>11}")
for init, s in summary.items():
    pm = lambda v, p=3: f"{np.mean(v):.{p}f} ± {np.std(v):.{p}f}"
    print(f"{init:10}{pm(s['acc']):>16}{pm(s['ari']):>16}{pm(s['ll'], 1):>18}"
          f"{pm(s['it'], 0):>14}{np.mean(s['conv']):>11.0%}")

# ---- Figure ----
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
plot_clusters(axes[0], X, y, title="True labels")
for ax, init in zip(axes[1:3], ("random", "kmeans")):
    m, acc, a = runs[init]
    plot_clusters(ax, X, m.predict(X), m, f"{init} init  (acc={acc:.2f}, ARI={a:.2f})")
for init, c in zip(("random", "kmeans"), ("#f4a261", "#2a9d8f")):
    axes[3].plot(runs[init][0].log_likelihoods_, color=c, label=init)
axes[3].set(title="Log-likelihood trace (seed 7)", xlabel="iteration", ylabel="log-likelihood")
axes[3].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "init_comparison.png", dpi=150)
