"""EM + BIC model selection on the Mall Customers dataset."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from em_gmm import GaussianMixtureEM
from em_gmm.data import load_mall_customers
from em_gmm.plotting import PALETTE, plot_ellipse

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

df = load_mall_customers()
X_raw = df[FEATURES].to_numpy(float)
scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)

K_RANGE = range(2, 9)
models = {K: GaussianMixtureEM(K, init="kmeans", max_iter=300, n_init=5, random_state=0).fit(X)
          for K in K_RANGE}
bic = {K: m.bic(X) for K, m in models.items()}
aic = {K: m.aic(X) for K, m in models.items()}
K_bic, K_aic = min(bic, key=bic.get), min(aic, key=aic.get)
print(f"BIC selects K={K_bic}   AIC selects K={K_aic}")
for K in K_RANGE:
    print(f"K={K}  log-lik={models[K].score(X):8.1f}  AIC={aic[K]:8.1f}  BIC={bic[K]:8.1f}")

best = models[K_bic]
labels = best.predict(X)
print(f"\nCluster summary (K={K_bic}, original units)")
for k in range(K_bic):
    mu = scaler.inverse_transform(best.means_[[k]])[0]
    print(f"  C{k}: weight={best.weights_[k]:.2f}  income={mu[0]:5.1f}k$  spending={mu[1]:5.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
Ks = list(K_RANGE)
axes[0].plot(Ks, [bic[k] for k in Ks], "s-", c="#2a9d8f", label="BIC")
axes[0].plot(Ks, [aic[k] for k in Ks], "o-", c="#f4a261", label="AIC")
axes[0].set(xlabel="K", ylabel="criterion (lower is better)", title="Model selection"); axes[0].legend()
for k in range(K_bic):
    m = labels == k
    axes[1].scatter(X_raw[m, 0], X_raw[m, 1], c=PALETTE[k], s=18, alpha=0.7, label=f"C{k}")
    mu = scaler.inverse_transform(best.means_[[k]])[0]
    # ellipses live in standardised space -> map to original units
    cov = best.covariances_[k] * np.outer(scaler.scale_, scaler.scale_)
    plot_ellipse(axes[1], mu, cov, PALETTE[k])
axes[1].set(xlabel=FEATURES[0], ylabel=FEATURES[1], title=f"GMM-EM segmentation (K={K_bic}, BIC)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "mall_segmentation.png", dpi=150)
