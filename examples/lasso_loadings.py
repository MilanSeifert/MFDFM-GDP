"""Shared LASSO loading estimation for the MFDFM examples.

Provides a drop-in replacement for ``mfdfm.estimation.estimate_loadings`` that
uses an L1-penalized regression (sklearn's ``Lasso``) instead of plain OLS
(``np.linalg.lstsq``). The LASSO shrinks small loadings to exactly zero, which
sparsifies the factor loading matrix Lambda (whole indicators can drop out).

The estimation mirrors ``estimate_loadings`` exactly:
  - Monthly variables regress on the PCA factors f^PC directly.
  - Quarterly variables regress on the G(L)-aggregated factors at quarter-end
    months, where G(L) f_t = (1/3)(f_t + f_{t-1} + f_{t-2}).
  - The idiosyncratic variance for quarterly variables uses the 9/3 = 3
    correction:  sigma^2 = 3 * var(residual).

A helper ``fit_lasso_model`` fits a full ``MFDFM`` with LASSO loadings by
temporarily monkeypatching the loading estimator used inside ``model.fit``,
so the rest of the pipeline (PCA, VAR, Kalman filter/smoother) is reused
unchanged.
"""

import sys
import os
import pickle

# Make the mfdfm package importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import Lasso

import mfdfm.model as _model_mod
from mfdfm import MFDFM
from mfdfm.data import MFData

# Real Swiss data set produced by ``examples/fetch_data.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWISS_PKL = os.path.join(_REPO_ROOT, "data", "swiss.pkl")


def load_swiss_data(path: str = SWISS_PKL):
    """Load the real Swiss mixed-frequency data set.

    Returns ``(data, quarterly_vars, gdp_var)`` exactly as produced by
    ``examples/fetch_data.py`` (which pickles a dict with those keys). Raises
    a clear error if the data set has not been fetched yet.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Real data set not found at {path}.\n"
            "Fetch it first with:\n"
            "    export FRED_API_KEY=your_key\n"
            "    python3 examples/fetch_data.py"
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["data"], payload["quarterly_vars"], payload["gdp_var"]


def estimate_loadings_lasso(
    data: MFData,
    f_pc: np.ndarray,
    n_factors: int,
    alpha: float,
    max_iter: int = 20000,
):
    """Estimate factor loadings and idiosyncratic variances via LASSO.

    Mirrors ``mfdfm.estimation.estimate_loadings`` but replaces the
    ``np.linalg.lstsq`` OLS step with sklearn's ``Lasso`` (L1 penalty
    ``alpha``). ``fit_intercept=False`` matches the original OLS, which has no
    intercept (data and factors are mean-zero / standardized).

    Parameters
    ----------
    data : MFData
        Mixed-frequency data container.
    f_pc : (T, r) array
        PCA factor scores (NaN outside the balanced panel).
    n_factors : int
        Number of factors r.
    alpha : float
        L1 penalty strength. Larger -> sparser loadings.
    max_iter : int
        Maximum coordinate-descent iterations for the Lasso solver.

    Returns
    -------
    Lambda : (n, r) array
        Factor loadings (rows may be all-zero where LASSO drops a variable).
    Sigma_ww : (n,) array
        Diagonal of the idiosyncratic covariance.
    """
    r = n_factors
    n = data.n
    Lambda = np.zeros((n, r))
    Sigma_ww = np.zeros(n)

    # Precompute temporally aggregated factors for the quarterly variables.
    f_agg = np.full((data.T, r), np.nan)
    for t in range(2, data.T):
        if not np.any(np.isnan(f_pc[t - 2 : t + 1])):
            f_agg[t] = (f_pc[t] + f_pc[t - 1] + f_pc[t - 2]) / 3.0

    # --- Quarterly variables ---
    for i in range(data.n_Q):
        y_i = data.Y[:, i]
        valid = (
            data.is_quarter_end
            & ~np.isnan(y_i)
            & ~np.isnan(f_agg[:, 0])
        )
        idx = np.where(valid)[0]
        if len(idx) <= r:
            continue

        F_sub = f_agg[idx]  # (m, r)
        y_sub = y_i[idx]    # (m,)
        las = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        las.fit(F_sub, y_sub)
        Lambda[i] = las.coef_
        resid = y_sub - F_sub @ Lambda[i]
        # Quarterly variance correction: sigma^2 = (9/3) * var(u_tilde).
        Sigma_ww[i] = 3.0 * np.var(resid, ddof=0)

    # --- Monthly variables ---
    for j in range(data.n_M):
        i = data.n_Q + j  # index in full variable ordering
        y_i = data.Y[:, i]
        valid = ~np.isnan(y_i) & ~np.isnan(f_pc[:, 0])
        idx = np.where(valid)[0]
        if len(idx) <= r:
            continue

        F_sub = f_pc[idx]
        y_sub = y_i[idx]
        las = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        las.fit(F_sub, y_sub)
        Lambda[i] = las.coef_
        resid = y_sub - F_sub @ Lambda[i]
        Sigma_ww[i] = np.var(resid, ddof=0)

    # Ensure positive variances.
    Sigma_ww = np.maximum(Sigma_ww, 1e-10)

    return Lambda, Sigma_ww


def fit_lasso_model(
    data,
    quarterly_vars,
    gdp_var,
    alpha,
    n_factors=4,
    n_lags=1,
    min_balanced_years=15,
):
    """Fit a full MFDFM using LASSO loadings.

    Temporarily replaces the loading estimator that ``model.fit`` calls so the
    entire downstream pipeline (PCA factors, VAR, Kalman filter + smoother) is
    reused without modification. The patch is always restored, even on error.
    """
    orig = _model_mod.estimate_loadings

    def patched(d, f, rr):
        return estimate_loadings_lasso(d, f, rr, alpha=alpha)

    _model_mod.estimate_loadings = patched
    try:
        model = MFDFM(
            n_factors=n_factors,
            n_lags=n_lags,
            min_balanced_years=min_balanced_years,
        )
        model.fit(data, quarterly_vars=quarterly_vars, gdp_var=gdp_var)
    finally:
        _model_mod.estimate_loadings = orig
    return model


def n_zero_loading_rows(Lambda: np.ndarray, tol: float = 1e-12) -> int:
    """Count variables whose entire loading row was shrunk to zero."""
    return int(np.sum(np.all(np.abs(Lambda) < tol, axis=1)))
