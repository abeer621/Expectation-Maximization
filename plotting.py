"""Plot helpers."""
from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np

PALETTE = ["#e63946", "#2a9d8f", "#f4a261", "#6a4c93",
           "#457b9d", "#e9c46a", "#8d99ae", "#b5838d"]


def plot_ellipse(ax, mean, cov, color, n_std=2.0, alpha=0.25):
    """Draw the n_std confidence ellipse of a 2-D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(mpatches.Ellipse(mean, w, h, angle=angle, facecolor=color,
                                  edgecolor=color, alpha=alpha, lw=1.5))


def plot_clusters(ax, X, labels, model=None, title=""):
    """Scatter coloured by label, with fitted ellipses/means if a model is given."""
    for k in np.unique(labels):
        m = labels == k
        ax.scatter(X[m, 0], X[m, 1], c=PALETTE[k % len(PALETTE)], s=12, alpha=0.6, linewidths=0)
    if model is not None:
        for k in range(model.n_components):
            c = PALETTE[k % len(PALETTE)]
            plot_ellipse(ax, model.means_[k], model.covariances_[k], c)
            ax.scatter(*model.means_[k], marker="x", s=80, c=c, linewidths=2, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
