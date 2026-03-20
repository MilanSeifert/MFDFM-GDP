"""Two-step parameter estimation (PCA + OLS) for the MFDFM.

Step 1: PCA on a balanced monthly panel to extract r principal components.
Step 2: OLS to estimate factor loadings, idiosyncratic variances, and
        VAR dynamics for the factors.

References:
    Giannone, Reichlin, Small (2008)
    Doz, Giannone, Reichlin (2011)
"""

import numpy as np
from typing import Tuple

from mfdfm.data import MFData


def estimate_pca_factors(
    data: MFData, n_factors: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Extract principal component factors from the balanced monthly panel.

    Returns
    -------
    f_pc : (T, r) array
        Factor scores on the full time range (NaN outside balanced panel).
    eigenvalues : (r,) array
        Eigenvalues of the covariance matrix for the selected factors.
    eigenvectors : (n_bal, r) array
        Eigenvectors (loadings on balanced panel).
    t_start, t_end : int
        Time range of the balanced panel.
    """
    panel, var_mask, t_start, t_end = data.get_balanced_monthly_panel()
    T_bal, n_bal = panel.shape
    r = n_factors

    if r > n_bal:
        raise ValueError(
            f"n_factors={r} > number of balanced variables={n_bal}"
        )

    # Eigendecomposition of covariance matrix
    cov = panel.T @ panel / T_bal
    eigenvalues, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx[:r]]
    V = eigvecs[:, idx[:r]]  # (n_bal, r)

    # Factor scores on balanced panel
    f_bal = panel @ V  # (T_bal, r)

    # Embed in full time range
    f_pc = np.full((data.T, r), np.nan)
    f_pc[t_start:t_end] = f_bal

    return f_pc, eigenvalues, V, t_start, t_end


def estimate_loadings(
    data: MFData, f_pc: np.ndarray, n_factors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate factor loadings and idiosyncratic variances via OLS.

    For monthly variables:
        x^M_{i,t} = Lambda_i f_t + u_{i,t}

    For quarterly variables (at quarter-end months):
        x^Q_{i,t} = Lambda_i G(L) f_t + u_tilde_{i,t}
        where G(L) f_t = (1/3)(f_t + f_{t-1} + f_{t-2})
        and sigma^2_i = (9/3) * var(u_tilde)  [quarterly correction]

    Returns
    -------
    Lambda : (n, r) array
        Factor loadings.
    Sigma_ww : (n,) array
        Diagonal of idiosyncratic covariance.
    """
    r = n_factors
    n = data.n
    Lambda = np.zeros((n, r))
    Sigma_ww = np.zeros(n)

    # Precompute temporally aggregated factors for quarterly vars
    f_agg = np.full((data.T, r), np.nan)
    for t in range(2, data.T):
        if not np.any(np.isnan(f_pc[t - 2 : t + 1])):
            f_agg[t] = (f_pc[t] + f_pc[t - 1] + f_pc[t - 2]) / 3.0

    # --- Quarterly variables ---
    for i in range(data.n_Q):
        y_i = data.Y[:, i]
        # Available: quarter-end, variable observed, aggregated factors exist
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
        Lambda[i] = np.linalg.lstsq(F_sub, y_sub, rcond=None)[0]
        resid = y_sub - F_sub @ Lambda[i]
        # Quarterly variance correction: sigma^2 = (9/3) * var(u_tilde)
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
        Lambda[i] = np.linalg.lstsq(F_sub, y_sub, rcond=None)[0]
        resid = y_sub - F_sub @ Lambda[i]
        Sigma_ww[i] = np.var(resid, ddof=0)

    # Ensure positive variances
    Sigma_ww = np.maximum(Sigma_ww, 1e-10)

    return Lambda, Sigma_ww


def estimate_var(
    f_pc: np.ndarray, n_lags: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(p) on the PCA factors.

    Parameters
    ----------
    f_pc : (T, r) array
        Factor scores (NaN outside balanced panel).
    n_lags : int
        Number of VAR lags (p).

    Returns
    -------
    Phi : (r, r*p) array
        VAR coefficient matrices [Phi_1 | ... | Phi_p].
    Sigma_vv : (r, r) array
        Innovation covariance.
    """
    # Extract contiguous non-NaN segment
    valid = ~np.isnan(f_pc[:, 0])
    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise ValueError("No valid factor observations for VAR estimation")

    f = f_pc[idx[0] : idx[-1] + 1]  # contiguous segment
    T_f, r = f.shape
    p = n_lags

    if T_f <= p + r:
        raise ValueError(
            f"Not enough observations ({T_f}) for VAR({p}) with {r} factors"
        )

    # Build regression matrices
    # Y = [f_{p+1}, ..., f_T]'  (T_f - p, r)
    # X = [f_p ... f_1, f_{p+1} ... f_2, ...]  (T_f - p, r*p)
    Y = f[p:]
    X = np.column_stack([f[p - lag - 1 : T_f - lag - 1] for lag in range(p)])

    # OLS: Y = X @ B + V, where B = [Phi_1' | ... | Phi_p']' (r*p, r)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]  # (r*p, r)
    Phi = B.T  # (r, r*p)

    residuals = Y - X @ B  # (T_f - p, r)
    Sigma_vv = residuals.T @ residuals / len(residuals)

    return Phi, Sigma_vv


def select_n_factors(
    data: MFData, max_r: int = 8
) -> Tuple[int, np.ndarray]:
    """Select number of factors via BIC on the GDP measurement equation.

    BIC(r) = ln(V[G_t(L) u_hat_{t,gdp}]) + ln(T)/T * r

    Returns
    -------
    best_r : int
        Optimal number of factors.
    bic_values : (max_r,) array
        BIC values for r = 1, ..., max_r.
    """
    bic_values = np.full(max_r, np.inf)

    for r in range(1, max_r + 1):
        try:
            f_pc, _, _, t_start, t_end = estimate_pca_factors(data, r)
            Lambda, Sigma_ww = estimate_loadings(data, f_pc, r)

            # Compute GDP residuals with temporal aggregation
            f_agg = np.full((data.T, r), np.nan)
            for t in range(2, data.T):
                if not np.any(np.isnan(f_pc[t - 2 : t + 1])):
                    f_agg[t] = (f_pc[t] + f_pc[t - 1] + f_pc[t - 2]) / 3.0

            gdp = data.Y[:, 0]
            valid = (
                data.is_quarter_end
                & ~np.isnan(gdp)
                & ~np.isnan(f_agg[:, 0])
            )
            idx = np.where(valid)[0]
            if len(idx) <= r:
                continue

            resid = gdp[idx] - f_agg[idx] @ Lambda[0]
            T_eff = len(idx)
            var_resid = np.var(resid, ddof=0)

            bic_values[r - 1] = np.log(var_resid) + np.log(T_eff) / T_eff * r
        except Exception:
            continue

    best_r = int(np.argmin(bic_values)) + 1
    return best_r, bic_values


def select_n_lags(
    data: MFData, n_factors: int, max_p: int = 4
) -> Tuple[int, np.ndarray]:
    """Select number of VAR lags via BIC.

    BIC(p) = ln|Sigma_vv(p)| + ln(T)/T * p * r^2

    Returns
    -------
    best_p : int
        Optimal number of lags.
    bic_values : (max_p,) array
        BIC values for p = 1, ..., max_p.
    """
    f_pc, _, _, _, _ = estimate_pca_factors(data, n_factors)
    r = n_factors

    # Get contiguous factor segment length
    valid = ~np.isnan(f_pc[:, 0])
    idx = np.where(valid)[0]
    T_eff = idx[-1] - idx[0] + 1

    bic_values = np.full(max_p, np.inf)

    for p in range(1, max_p + 1):
        try:
            Phi, Sigma_vv = estimate_var(f_pc, p)
            sign, logdet = np.linalg.slogdet(Sigma_vv)
            if sign > 0:
                bic_values[p - 1] = (
                    logdet + np.log(T_eff) / T_eff * p * r**2
                )
        except Exception:
            continue

    best_p = int(np.argmin(bic_values)) + 1
    return best_p, bic_values
