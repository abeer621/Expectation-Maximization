"""Clustering evaluation against known labels."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def matched_accuracy(y_true, y_pred):
    """Accuracy under the best one-to-one cluster/label mapping (Hungarian algorithm).

    Returns (accuracy, mapping) where mapping is a list of (true_label, predicted_cluster).
    Works for rectangular confusion matrices (K_pred != K_true); unmatched
    clusters simply count as errors.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    t_labels, p_labels = np.unique(y_true), np.unique(y_pred)
    confusion = np.zeros((len(t_labels), len(p_labels)), dtype=int)
    for i, t in enumerate(t_labels):
        for j, p in enumerate(p_labels):
            confusion[i, j] = np.sum((y_true == t) & (y_pred == p))
    rows, cols = linear_sum_assignment(-confusion)
    acc = confusion[rows, cols].sum() / len(y_true)
    return acc, list(zip(t_labels[rows], p_labels[cols]))


def ari(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)
