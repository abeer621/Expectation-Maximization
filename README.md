# Expectation-Maximization for Gaussian Mixture Models

A from-scratch NumPy implementation of the EM algorithm for full-covariance Gaussian mixtures, with a study of **initialization strategies**, **model selection (AIC/BIC)**, and the **relationship to K-means**. It is verified against scikit-learn and applied to synthetic data and a real customer-segmentation dataset.

![Initialization comparison](figures/init_comparison.png)

## What's here

| | |
|---|---|
| `em_gmm/gmm.py` | `GaussianMixtureEM`: fit / predict / predict_proba / score / AIC / BIC |
| `em_gmm/data.py` | Synthetic overlapping-Gaussian generator; Mall Customers loader |
| `em_gmm/metrics.py` | Hungarian-matched accuracy and Adjusted Rand Index |
| `experiments/` | Scripts that reproduce every figure and number below |
| `tests/` | Unit tests, including a cross-check against `sklearn.mixture.GaussianMixture` |

### Implementation notes
- **Numerically stable:** E-step in log space (`logsumexp`), Gaussian densities via Cholesky factorisation.
- **Singularity guard:** a covariance ridge (`reg_covar`) prevents the likelihood blow-up when a component collapses onto a point. The GMM likelihood is unbounded, so this matters.
- **Initialization options:** `random`, `kmeans`, `kmeans++`, plus `n_init` restarts keeping the best log-likelihood.
- **Tested:** the log-likelihood is checked to be monotone non-decreasing, and the converged log-likelihood agrees with scikit-learn's when both start from the same initial state.

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests
python experiments/init_comparison.py
python experiments/select_k.py
python experiments/mall_customers.py
```

```python
from em_gmm import GaussianMixtureEM

gmm = GaussianMixtureEM(n_components=4, init="kmeans++", n_init=5, random_state=0).fit(X)
labels = gmm.predict(X)
gmm.bic(X)
```

## Theory (short version)

For `p(x) = Σ_k π_k N(x | μ_k, Σ_k)` the log-likelihood contains a log of a sum, so there is no closed-form maximiser. EM maximises the evidence lower bound (ELBO) by coordinate ascent:

- **E-step:** set `q(z) = p(z | x, θ)`, giving responsibilities `r_nk`. This makes the bound tight at the current parameters.
- **M-step:** with `r_nk` fixed, closed-form updates for `π_k, μ_k, Σ_k`.

Chaining `ℓ(θ_new) ≥ L(q, θ_new) ≥ L(q, θ_old) = ℓ(θ_old)` shows the likelihood never decreases. This guarantees convergence to a *stationary point*, not the global optimum, which is why initialization matters. K-means is recovered as the `σ² → 0` limit of EM on an isotropic mixture (Bishop, PRML §9.3.2): responsibilities collapse to hard assignments and the M-step becomes the centroid update.

## Experiments

### 1. Initialization (synthetic data, 4 overlapping Gaussians, N = 600, K = 4)

Single run, same seed for all methods:

| | Random | K-means | K-means++ |
|---|---:|---:|---:|
| Matched accuracy | 0.637 | 0.832 | 0.832 |
| Adjusted Rand Index | 0.456 | 0.608 | 0.608 |
| Final log-likelihood | -2166.7 | -2128.8 | -2128.8 |
| Iterations (max 200) | 201 (not converged) | 170 | 170 |

Over 50 seeds (mean ± std):

| init | accuracy | ARI | log-likelihood | iterations | converged |
|---|---:|---:|---:|---:|---:|
| random | 0.812 ± 0.060 | 0.587 ± 0.065 | -2131.7 ± 9.4 | 174 ± 31 | 70% |
| kmeans | 0.832 ± 0.000 | 0.608 ± 0.000 | -2128.8 ± 0.0 | 171 ± 1 | 100% |
| kmeans++ | 0.832 ± 0.000 | 0.608 ± 0.000 | -2128.8 ± 0.0 | 171 ± 2 | 100% |

**Takeaway:** random initialization *can* land in a clearly worse solution (the single-run row above), but on average the gap is modest. The main benefit of K-means-based initialization is **reliability**: zero variance across seeds versus a 30% non-convergence rate for random starts. Raw convergence speed is similar once converged.

### 2. Choosing K (synthetic data, true K = 4)

![Model selection](figures/select_k.png)

On the primary dataset BIC selects K = 4 and AIC selects K = 7, consistent with BIC's heavier `p·ln N` penalty versus AIC's `2p`. Repeating over 30 freshly sampled datasets gives a more nuanced picture:

| criterion | selected K (30 datasets) |
|---|---|
| AIC | K=4: 17 · K=5: 5 · K=6: 3 · K=7: 4 · K=8: 1 |
| BIC | K=3: 18 · K=4: 12 |

The clusters overlap heavily by design, so BIC tends to *under*-fit here (merging two clusters), while AIC over-fits when it errs. Neither criterion is uniformly "more reliable"; which is better depends on how well separated the components are.

### 3. Mall Customers segmentation (Annual Income × Spending Score)

![Mall segmentation](figures/mall_segmentation.png)

With 5 restarts per K, BIC selects **K = 4** (BIC 1055.5 vs 1058.5 for K = 5, a weak preference). Segments: low-income/high-spend, mid-income/mid-spend (the largest group, 41%), high-income/high-spend, and high-income/low-spend.

**Caveat on restarts:** with a *single* run per K, K = 4 falls into a poor local optimum (log-lik -471.6 vs -466.8 with restarts), which makes K = 5 look best. This is a concrete example of initialization sensitivity corrupting model selection, and a reason to always use `n_init > 1` when comparing values of K.

## Limitations / future work
- Full-covariance GMM only; no diagonal/tied/spherical variants.
- Initialization sets only the means (covariances start as identity, weights uniform); initializing from K-means cluster statistics would likely speed convergence.
- Rival-penalised EM (Cheung, 2005) for automatic K selection was studied but not implemented.
- Evaluation against ground truth is only possible on the synthetic data.

## References
- Dempster, Laird & Rubin (1977). *Maximum likelihood from incomplete data via the EM algorithm.* JRSS-B.
- Bishop (2006). *Pattern Recognition and Machine Learning*, ch. 9.
- Arthur & Vassilvitskii (2007). *k-means++: the advantages of careful seeding.*
- Hubert & Arabie (1985). *Comparing partitions.*
