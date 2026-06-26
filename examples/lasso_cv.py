"""Tune the LASSO penalty (alpha) by expanding-window time-series CV.

The score is the one-quarter-ahead GDP forecast RMSE. For each alpha in a grid,
and for each expanding-window fold ending at a quarter-end month Q:

  1. Estimate parameters (PCA + LASSO loadings + VAR) on the training data
     up to and including month Q.
  2. Run the Kalman filter to obtain the filtered state at Q:  xi_{Q|Q}.
  3. Forecast one quarter (3 monthly steps) ahead via the transition equation:
        xi_{Q+1|Q} = F^3 @ xi_{Q|Q}
     (one quarter ahead = 3 applications of the monthly transition F).
  4. Form the BCI forecast:  Lambda_gdp @ f_{Q+1|Q}.
  5. Compare to the actual GDP observed at the next quarter-end Q+1,
     standardized with the *training* mean/std (no look-ahead).

The alpha with the lowest pooled RMSE across folds is selected and a plot of
RMSE vs alpha is saved.

Run:
    python3 examples/lasso_cv.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lasso_loadings import fit_lasso_model, load_swiss_data

# --- CV configuration ---
ALPHAS = np.logspace(-3.0, -0.3, 7)   # ~0.001 ... 0.5
# First training cut-off. Real Swiss GDP starts ~1995-06, so the first fold
# must leave enough quarter-end GDP observations to estimate GDP loadings;
# month index 300 is ~2000-01, giving ~5 years of GDP in the first window.
FIRST_FOLD_MONTH = 300
FOLD_STEP_QUARTERS = 2                 # advance the window by N quarters per fold
N_FACTORS = 4
N_LAGS = 1


def _quarter_end_indices(dates):
    return [t for t, d in enumerate(dates) if d.month in (3, 6, 9, 12)]


def _build_folds(dates, T):
    """Quarter-end cut-off months Q with a valid next quarter-end Q+3."""
    qends = _quarter_end_indices(dates)
    folds = []
    for q in qends:
        if q < FIRST_FOLD_MONTH:
            continue
        q_next = q + 3
        if q_next > T - 1:
            continue
        folds.append((q, q_next))
    return folds[::FOLD_STEP_QUARTERS]


def _forecast_one_quarter(model):
    """One-quarter-ahead factor forecast f_{Q+1|Q} from xi_{Q|Q}."""
    xi = model.kf_result_.xi_filt[-1].copy()  # xi_{Q|Q}: last training month
    for _ in range(3):                          # 3 monthly steps = one quarter
        xi = model.F_ @ xi
    r = model.r_
    return xi[:r]


def main():
    print("=" * 64)
    print("MFDFM — LASSO alpha tuning via expanding-window TS-CV")
    print("=" * 64)

    print("\n1. Loading real Swiss data...")
    data, quarterly_vars, gdp_var = load_swiss_data()
    dates = data.index
    T = len(dates)
    gdp_raw = data[gdp_var].values  # raw quarterly GDP (NaN off quarter-ends)
    print(f"   {data.shape[1]} variables, {T} months "
          f"({len(quarterly_vars)} quarterly), "
          f"{dates[0]:%Y-%m}–{dates[-1]:%Y-%m}")

    folds = _build_folds(dates, T)
    print(f"   {len(folds)} expanding-window folds, "
          f"{len(ALPHAS)} alpha values")
    print(f"   {len(folds) * len(ALPHAS)} model fits total")

    rmse_per_alpha = np.full(len(ALPHAS), np.nan)

    print("\n2. Running cross-validation...")
    for a_idx, alpha in enumerate(ALPHAS):
        sq_errors = []
        for (q, q_next) in folds:
            actual_raw = gdp_raw[q_next]
            if np.isnan(actual_raw):
                continue

            train = data.iloc[: q + 1]
            try:
                model = fit_lasso_model(
                    train, quarterly_vars, gdp_var,
                    alpha=alpha, n_factors=N_FACTORS, n_lags=N_LAGS,
                )
            except Exception as exc:
                # A fold may fail (e.g. too short a balanced panel); skip it.
                print(f"     [alpha={alpha:.4g}] fold @ {dates[q]:%Y-%m} "
                      f"skipped: {exc}")
                continue

            f_fore = _forecast_one_quarter(model)
            bci_fore = float(model.Lambda_[0] @ f_fore)

            # Standardize the actual with TRAINING parameters (no look-ahead).
            gdp_mean = model.data_.means[0]
            gdp_std = model.data_.stds[0]
            actual_std = (actual_raw - gdp_mean) / gdp_std

            sq_errors.append((bci_fore - actual_std) ** 2)

        if sq_errors:
            rmse_per_alpha[a_idx] = float(np.sqrt(np.mean(sq_errors)))
        print(f"   alpha={alpha:>9.4g}   "
              f"RMSE={rmse_per_alpha[a_idx]:.4f}   "
              f"(n_folds={len(sq_errors)})")

    # --- Select best alpha ---
    if np.all(np.isnan(rmse_per_alpha)):
        raise RuntimeError("All folds failed; cannot select alpha.")
    best_idx = int(np.nanargmin(rmse_per_alpha))
    best_alpha = ALPHAS[best_idx]

    print("\n3. Best alpha")
    print("   " + "-" * 40)
    print(f"   alpha       = {best_alpha:.4g}")
    print(f"   CV RMSE     = {rmse_per_alpha[best_idx]:.4f}")
    print("   " + "-" * 40)

    # --- Plot RMSE vs alpha ---
    print("\n4. Generating plot...")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ALPHAS, rmse_per_alpha, "o-", color="C0")
    ax.scatter([best_alpha], [rmse_per_alpha[best_idx]],
               color="red", zorder=5, s=80,
               label=f"best alpha = {best_alpha:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel("LASSO alpha (log scale)")
    ax.set_ylabel("One-quarter-ahead GDP forecast RMSE")
    ax.set_title("LASSO alpha tuning — expanding-window time-series CV")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = "lasso_cv.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"   Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
