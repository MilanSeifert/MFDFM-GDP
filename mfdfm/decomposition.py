"""Observation weights and news decomposition.

Observation weights: Koopman & Harvey (2003) algorithm for decomposing
smoothed factors into weighted sums of all observations.

News decomposition: Banbura & Modugno (2014) algorithm for decomposing
BCI revisions into news contributions from individual data releases.
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from mfdfm.model import MFDFM


def compute_bci_weights(
    model: "MFDFM", target_period: Optional[pd.Timestamp] = None
) -> Dict[str, np.ndarray]:
    """Compute observation weights for the BCI at a single target month.

    The smoothed factors can be written as a weighted sum of all observations:
        f_{t|T} = sum_k w_k(t,T) y_k

    The BCI weights are that linear combination scaled by the GDP loadings:
        BCI_t = Lambda_gdp @ f_{t|T} = sum_k (Lambda_gdp @ w_k(t,T)) y_k

    Uses the exact Koopman & Harvey (2003) observation-weight algorithm, so the
    returned weights reconstruct the BCI to numerical precision (because the
    filter is initialised at xi_0 = 0, the smoothed state is an exact linear
    function of the observations).

    Parameters
    ----------
    target_period : Timestamp, optional
        Target month t. Defaults to the latest quarter-end month.

    Returns
    -------
    dict with:
        'bci_weights' : (T, n) array
            bci_weights[k, i] = weight of observation y_{i,k} on BCI_t.
            Unobserved (k, i) entries are zero.
        'target_time' : int — the target time index t.
        'target_date' : Timestamp — the target month.
    """
    kf = model.kf_result_
    T = model.data_.T
    r = model.r_
    n = model.data_.n
    Lambda_gdp = model.Lambda_[0]  # (r,)

    t_target = _resolve_target_time(model, target_period)

    weights = _compute_smoother_weights_for_t(model, t_target)

    # BCI weight per observation: Lambda_gdp @ w_k(t,T)[:r, :]
    bci_w = np.zeros((T, n))
    for k in range(T):
        obs = kf.obs_idx[k]
        if len(obs) == 0:
            continue
        w_k = weights[k]                  # (s, m_k)
        bci_w[k, obs] = Lambda_gdp @ w_k[:r, :]   # (m_k,)

    return {
        "bci_weights": bci_w,
        "target_time": t_target,
        "target_date": model.data_.dates[t_target],
    }


def _resolve_target_time(
    model: "MFDFM", target_period: Optional[pd.Timestamp]
) -> int:
    """Resolve a target month to a time index (default: latest quarter-end)."""
    if target_period is not None:
        dates = model.data_.dates
        return int(np.argmin(np.abs(dates - pd.Timestamp(target_period))))
    qe_idx = np.where(model.data_.is_quarter_end)[0]
    return int(qe_idx[-1]) if len(qe_idx) else model.data_.T - 1


def _compute_smoother_weights_for_t(
    model: "MFDFM", t_target: int
) -> List[np.ndarray]:
    """Exact observation weights W_k(t,T) for the smoothed state at t_target.

    Writes the smoothed state as xi_{t|T} = sum_k W_k(t,T) y_k, following
    Koopman & Harvey (2003). The derivation works in two stages:

    1. Innovation-space weights A_k, where xi_{t|T} = sum_k A_k v_k and v_k is
       the Kalman innovation (prediction error) at step k. Because innovations
       are mutually uncorrelated, smoothing leaves the weights on *past*
       innovations (k < t) equal to the predicted-state weights, while
       current/future innovations (k >= t) enter through the smoothing cumulant
       r_{t-1} (Durbin & Koopman, 2012):
           A_k = F^{t-k} K_k                            for k < t
           A_k = P_{t|t-1} (prod_{l=t}^{k-1} L_l') H_k' S_k^{-1}   for k >= t
       with L_l = F (I - K_l H_l) and K_l the (update) Kalman gain.

    2. Convert innovation weights to observation weights via v_k = y_k -
       H_k xi_{k|k-1} and xi_{k|k-1} = sum_{j<k} Pi_{k,j} y_j, which gives
           W_j = A_j - b_j F K_j,   b_{j-1} = A_j H_j + b_j L_j,  b_{T-1} = 0.

    Returns a list of (s, m_k) arrays, one per time step k (m_k = number of
    observed variables at k; empty for fully-missing months).
    """
    kf = model.kf_result_
    F = model.F_
    T = model.data_.T
    s = 3 * model.r_
    I = np.eye(s)

    def _Ht_Sinv(k: int) -> np.ndarray:
        # H_k' S_k^{-1}  (s, m_k), computed via a solve for stability.
        H_k = kf.H_obs_list[k]            # (m_k, s)
        S_k = kf.inn_cov[k]               # (m_k, m_k)
        return np.linalg.solve(S_k, H_k).T

    # --- Stage 1: innovation-space weights A_k ---
    A = [None] * T
    P_tau = kf.P_pred[t_target]           # (s, s)

    # Current/future innovations k >= t_target.
    P_acc = I.copy()                      # prod_{l=t}^{k-1} L_l'  (built left-to-right)
    for k in range(t_target, T):
        obs = kf.obs_idx[k]
        if len(obs) == 0:
            A[k] = np.zeros((s, 0))
            L_k = F
        else:
            A[k] = P_tau @ (P_acc @ _Ht_Sinv(k))      # (s, m_k)
            L_k = F @ (I - kf.kalman_gains[k] @ kf.H_obs_list[k])
        P_acc = P_acc @ L_k.T

    # Past innovations j < t_target:  A_j = F^{t-j} K_j.
    F_pow = I.copy()
    for j in range(t_target - 1, -1, -1):
        F_pow = F @ F_pow                 # F^{t-j}
        obs = kf.obs_idx[j]
        if len(obs) == 0:
            A[j] = np.zeros((s, 0))
        else:
            A[j] = F_pow @ kf.kalman_gains[j]

    # --- Stage 2: innovation -> observation weights ---
    W = [None] * T
    b = np.zeros((s, s))                  # b_{T-1} = 0
    for j in range(T - 1, -1, -1):
        obs = kf.obs_idx[j]
        if len(obs) == 0:
            W[j] = np.zeros((s, 0))
            b = b @ F                     # b_{j-1} = b_j F  (L_j = F when no update)
        else:
            K_j = kf.kalman_gains[j]
            H_j = kf.H_obs_list[j]
            W[j] = A[j] - b @ (F @ K_j)
            L_j = F @ (I - K_j @ H_j)
            b = A[j] @ H_j + b @ L_j
    return W


def compute_group_contributions(
    model: "MFDFM",
    groups: Dict[str, str],
    target_period: Optional[pd.Timestamp] = None,
    other_label: str = "other",
) -> Dict:
    """Decompose a single month's BCI into additive group contributions.

    Following Galli (2017, Section 4.2), the smoothed BCI is a weighted sum of
    every observation:
        BCI_t = sum_k sum_i w_{k,i}(t) y_{k,i}
    Summing the per-observation contributions w_{k,i}(t) y_{k,i} over all months
    k and over the indicators i belonging to a group g gives that group's
    contribution to BCI_t. The contributions sum exactly to BCI_t.

    Parameters
    ----------
    groups : dict {indicator name -> group label}
        Maps each indicator to its group (e.g. one of the paper's 17 indicator
        categories). Indicators not listed are pooled under ``other_label``.
    target_period : Timestamp, optional
        Target month. Defaults to the latest quarter-end.
    other_label : str
        Group label for indicators absent from ``groups``.

    Returns
    -------
    dict with:
        'contributions' : Series indexed by group label (sums to BCI_t).
        'by_indicator'  : Series indexed by indicator name (sums to BCI_t).
        'bci'           : float — BCI_t (== contributions.sum()).
        'target_time'   : int — target time index.
        'target_date'   : Timestamp — target month.
    """
    res = compute_bci_weights(model, target_period)
    bci_w = res["bci_weights"]                      # (T, n)
    Y = np.nan_to_num(model.data_.Y, nan=0.0)       # (T, n)

    # Per-indicator contribution: sum over all months k of w_{k,i} y_{k,i}.
    contrib_by_var = (bci_w * Y).sum(axis=0)        # (n,)
    var_names = model.data_.var_names

    by_indicator = pd.Series(contrib_by_var, index=var_names, name="contribution")

    # Aggregate to groups.
    labels = [groups.get(v, other_label) for v in var_names]
    contributions = by_indicator.groupby(pd.Index(labels, name="group")).sum()
    contributions.name = "contribution"
    contributions = contributions.sort_values(key=np.abs, ascending=False)

    return {
        "contributions": contributions,
        "by_indicator": by_indicator,
        "bci": float(contrib_by_var.sum()),
        "target_time": res["target_time"],
        "target_date": res["target_date"],
    }


def compute_news_revision(
    model: "MFDFM",
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
    target_period: Optional[pd.Timestamp] = None,
) -> Dict:
    """Decompose BCI revision into news contributions.

    E[BCI_t | Omega_{v+1}] = E[BCI_t | Omega_v] + sum_j b_j * I_j

    where I_j = y_{i_j, t_j} - E[y_{i_j, t_j} | Omega_v] is the news content.

    Parameters
    ----------
    model : fitted MFDFM
        Current model (fitted on some data vintage).
    old_data : DataFrame
        Previous data vintage.
    new_data : DataFrame
        Updated data vintage.
    target_period : Timestamp, optional
        Target month. Defaults to latest quarter-end in new_data.
    """
    from mfdfm.model import MFDFM

    mfd_old = model.data_
    quarterly_vars = mfd_old.quarterly_names
    gdp_var = mfd_old.gdp_var

    # Fit model on old data
    model_old = MFDFM(n_factors=model.r_, n_lags=model.p_)
    model_old.fit(old_data, quarterly_vars, gdp_var)

    # Fit model on new data
    model_new = MFDFM(n_factors=model.r_, n_lags=model.p_)
    model_new.fit(new_data, quarterly_vars, gdp_var)

    # Find target period
    if target_period is None:
        qe = np.where(model_new.data_.is_quarter_end)[0]
        if len(qe) == 0:
            raise ValueError("No quarter-end months in new data")
        t_target = qe[-1]
    else:
        dates_new = model_new.data_.dates
        t_target = np.argmin(np.abs(dates_new - target_period))

    # BCI values
    bci_old = model_old.business_cycle_index
    bci_new = model_new.business_cycle_index
    target_date = model_new.data_.dates[t_target]

    bci_old_val = bci_old.loc[target_date] if target_date in bci_old.index else np.nan
    bci_new_val = bci_new.loc[target_date]
    revision = bci_new_val - bci_old_val

    # Identify new data releases
    Y_old = model_old.data_.Y
    Y_new = model_new.data_.Y

    # Align time indices
    old_dates = model_old.data_.dates
    new_dates = model_new.data_.dates
    common_dates = old_dates.intersection(new_dates)

    news_items = []
    for date in common_dates:
        t_old = np.searchsorted(old_dates, date)
        t_new = np.searchsorted(new_dates, date)
        if t_old >= len(old_dates) or t_new >= len(new_dates):
            continue

        for i in range(model_old.data_.n):
            was_missing = np.isnan(Y_old[t_old, i])
            is_available = not np.isnan(Y_new[t_new, i])
            if was_missing and is_available:
                var_name = model_old.data_.var_names[i]
                news_items.append({
                    "indicator": var_name,
                    "date": date,
                    "value": Y_new[t_new, i],
                    "t_new": t_new,
                    "i": i,
                })

    # Compute news content and weights
    # For each new release j: I_j = y_{i_j, t_j} - E[y_{i_j, t_j} | Omega_v]
    # The forecast E[y | Omega_v] comes from the old model's Kalman filter
    indicators = []
    news_values = []
    news_weights = []
    contributions = []

    kf_old = model_old.kf_result_
    Lambda_gdp = model_new.Lambda_[0]
    r = model_new.r_

    for item in news_items:
        t = item["t_new"]
        i = item["i"]

        # Expected value from old model at this time/variable
        # Use the predicted observation: H_t @ xi_{t|t-1}
        t_old_idx = np.searchsorted(old_dates, item["date"])
        if t_old_idx >= len(old_dates):
            continue

        H_full_old = _get_H_row(model_old, t_old_idx, i)
        if H_full_old is None:
            continue

        xi_pred_old = kf_old.xi_pred[t_old_idx]
        expected = H_full_old @ xi_pred_old

        actual = item["value"]
        news = actual - expected

        # Weight on BCI: proportional to GDP loadings and factor covariance
        P_pred_old = kf_old.P_pred[t_old_idx]
        # b_j = Lambda_gdp @ Cov(f_t, f_{t_j}) @ Lambda_{i_j}' / Var(I_j)
        H_full_new = _get_H_row(model_new, t, i)
        if H_full_new is None:
            continue

        cov_state = P_pred_old  # Cov(xi_{t|t-1})
        var_innovation = H_full_old @ cov_state @ H_full_old + model_old.Sigma_ww_[i]
        if var_innovation < 1e-15:
            continue

        # Simplified weight computation
        weight = (Lambda_gdp @ cov_state[:r, :] @ H_full_old) / var_innovation

        indicators.append(f"{item['indicator']} ({item['date'].strftime('%Y-%m')})")
        news_values.append(news)
        news_weights.append(weight)
        contributions.append(weight * news)

    return {
        "revision": revision,
        "news": np.array(news_values) if news_values else np.array([]),
        "weights": np.array(news_weights) if news_weights else np.array([]),
        "contributions": np.array(contributions) if contributions else np.array([]),
        "indicators": indicators,
        "target_date": target_date,
        "bci_old": bci_old_val,
        "bci_new": bci_new_val,
    }


def _get_H_row(model: "MFDFM", t: int, i: int) -> Optional[np.ndarray]:
    """Get the measurement equation row for variable i at time t."""
    r = model.r_
    s = 3 * r
    n_Q = model.data_.n_Q
    Lambda = model.Lambda_

    H_row = np.zeros(s)

    if i < n_Q:
        # Quarterly variable — only meaningful at quarter-end
        if not model.data_.is_quarter_end[t]:
            return None
        H_row[:r] = Lambda[i] / 3.0
        H_row[r : 2 * r] = Lambda[i] / 3.0
        H_row[2 * r : 3 * r] = Lambda[i] / 3.0
    else:
        # Monthly variable
        H_row[:r] = Lambda[i]

    return H_row
