"""Compare OLS vs LASSO factor loadings in the MFDFM.

Fits the model twice on the **real Swiss data set** (``data/swiss.pkl``,
produced by ``examples/fetch_data.py``):

  1. Standard OLS loadings (the default ``MFDFM``).
  2. LASSO loadings at a fixed, reasonable alpha.

It compares, side by side:
  - BCI correlation with actual quarterly GDP (the paper's headline metric).
  - Number of loading rows shrunk entirely to zero (sparsity).
  - RMSE of the estimated BCI vs actual GDP at quarter-end months.

There is no ground-truth latent business cycle for real data, so the
benchmark is observed quarterly GDP, evaluated at the quarter-end months where
GDP is actually published. Both series are z-scored before computing RMSE so
the comparison is on a common scale (the model's BCI is in standardized GDP
units; correlation is scale-invariant, RMSE is not).

Run:
    python3 examples/lasso_comparison.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfdfm import MFDFM
from lasso_loadings import fit_lasso_model, n_zero_loading_rows, load_swiss_data

# Fixed, reasonable L1 penalty for the comparison. Chosen large enough that
# some loading rows are driven entirely to zero (visible sparsity) while the
# BCI still tracks GDP well on the real data set.
ALPHA = 0.15


def _zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _evaluate(model, gdp_obs):
    """Return (correlation, n_zero_rows, rmse) versus observed GDP.

    ``gdp_obs`` is the actual quarterly GDP series (NaN off quarter-ends). The
    BCI is sampled at the quarter-end months where GDP is published.
    """
    bci = model.business_cycle_index
    gdp = gdp_obs.dropna()
    bci_at_q = bci.reindex(gdp.index)
    corr = float(np.corrcoef(bci_at_q.values, gdp.values)[0, 1])
    nzero = n_zero_loading_rows(model.Lambda_)
    rmse = _rmse(_zscore(bci_at_q.values), _zscore(gdp.values))
    return corr, nzero, rmse


def main():
    print("=" * 64)
    print("MFDFM — OLS vs LASSO loadings comparison (real Swiss data)")
    print("=" * 64)

    print("\n1. Loading real Swiss data...")
    data, quarterly_vars, gdp_var = load_swiss_data()
    gdp_obs = data[gdp_var]
    print(f"   {data.shape[1]} variables, {data.shape[0]} months "
          f"({len(quarterly_vars)} quarterly), "
          f"{data.index[0]:%Y-%m}–{data.index[-1]:%Y-%m}")

    print("\n2. Fitting model with OLS loadings...")
    ols_model = MFDFM(n_factors=4, n_lags=1)
    ols_model.fit(data, quarterly_vars=quarterly_vars, gdp_var=gdp_var)

    print(f"3. Fitting model with LASSO loadings (alpha={ALPHA})...")
    lasso_model = fit_lasso_model(
        data, quarterly_vars, gdp_var, alpha=ALPHA, n_factors=4, n_lags=1
    )

    ols_corr, ols_nzero, ols_rmse = _evaluate(ols_model, gdp_obs)
    las_corr, las_nzero, las_rmse = _evaluate(lasso_model, gdp_obs)

    n_total = ols_model.Lambda_.shape[0]

    # --- Side-by-side results table ---
    print("\n4. Results")
    print("   " + "-" * 56)
    print(f"   {'Metric':<32}{'OLS':>10}{'LASSO':>14}")
    print("   " + "-" * 56)
    print(f"   {'BCI-GDP correlation':<32}{ols_corr:>10.4f}{las_corr:>14.4f}")
    print(f"   {'Zero loading rows':<32}{ols_nzero:>7d}/{n_total:<2d}"
          f"{las_nzero:>11d}/{n_total:<2d}")
    print(f"   {'RMSE vs GDP (z-scored)':<32}{ols_rmse:>10.4f}"
          f"{las_rmse:>14.4f}")
    print("   " + "-" * 56)

    # --- Comparison plot ---
    print("\n5. Generating comparison plot...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: BCI time series (z-scored for a common scale) with observed GDP.
    ax = axes[0]
    dates = data.index
    gdp_q = gdp_obs.dropna()
    ax.scatter(gdp_q.index, _zscore(gdp_q.values), label="Observed GDP (QoQ)",
               color="black", s=14, alpha=0.6, zorder=5)
    ax.plot(dates, _zscore(ols_model.business_cycle_index.values),
            label=f"OLS (corr={ols_corr:.3f})", linewidth=1.3)
    ax.plot(dates, _zscore(lasso_model.business_cycle_index.values),
            label=f"LASSO (corr={las_corr:.3f})", linewidth=1.3)
    ax.set_title("Business Cycle Index (z-scored) vs observed GDP: OLS vs LASSO")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()

    # Panels 2 & 3: loading heatmaps on a shared colour scale.
    vmax = float(np.abs(ols_model.Lambda_).max())
    var_names = ols_model.data_.var_names
    for ax, Lam, title in (
        (axes[1], ols_model.Lambda_,
         f"OLS loadings |Lambda|  ({ols_nzero}/{n_total} rows zeroed)"),
        (axes[2], lasso_model.Lambda_,
         f"LASSO loadings |Lambda|  ({las_nzero}/{n_total} rows zeroed)"),
    ):
        im = ax.imshow(
            np.abs(Lam).T, aspect="auto", cmap="viridis",
            vmin=0.0, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_ylabel("Factor")
        ax.set_yticks(range(Lam.shape[1]))
        ax.set_yticklabels([f"f{j + 1}" for j in range(Lam.shape[1])])
        ax.set_xticks(range(len(var_names)))
        ax.set_xticklabels(var_names, rotation=90, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    plt.tight_layout()
    out = "lasso_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"   Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
