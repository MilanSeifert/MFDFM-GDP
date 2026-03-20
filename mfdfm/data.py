"""Mixed-frequency data container and preprocessing."""

import numpy as np
import pandas as pd
from typing import List, Tuple


class MFData:
    """Container for mixed-frequency panel data.

    Reorders variables internally as [GDP, other quarterly, monthly] and
    computes observation masks for the Kalman filter.

    Parameters
    ----------
    data : DataFrame
        Panel data with DatetimeIndex (monthly frequency).
        NaN for missing observations.
    quarterly_vars : list of str
        Column names of quarterly variables (observed every 3rd month).
    gdp_var : str
        Name of the GDP column (must be in quarterly_vars).
        GDP is placed first in the internal ordering.
    min_balanced_years : int
        Minimum number of years for the balanced PCA panel.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        quarterly_vars: List[str],
        gdp_var: str,
        min_balanced_years: int = 15,
    ):
        if gdp_var not in quarterly_vars:
            raise ValueError(f"gdp_var '{gdp_var}' must be in quarterly_vars")
        for v in quarterly_vars:
            if v not in data.columns:
                raise ValueError(f"Quarterly variable '{v}' not found in data")

        # Convert PeriodIndex to DatetimeIndex if needed
        if isinstance(data.index, pd.PeriodIndex):
            data = data.to_timestamp()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("data must have DatetimeIndex or PeriodIndex")

        # Identify monthly vars
        monthly_vars = [v for v in data.columns if v not in quarterly_vars]

        # Reorder: [GDP, other quarterly, monthly]
        other_q = [v for v in quarterly_vars if v != gdp_var]
        self.var_names = [gdp_var] + other_q + monthly_vars
        self.quarterly_names = [gdp_var] + other_q
        self.monthly_names = monthly_vars
        self.gdp_var = gdp_var

        self.n_Q = len(quarterly_vars)
        self.n_M = len(monthly_vars)
        self.n = self.n_Q + self.n_M
        self.T = len(data)
        self.dates = data.index
        self.min_balanced_years = min_balanced_years

        # Quarter-end mask
        self.is_quarter_end = np.array(
            [d.month in (3, 6, 9, 12) for d in self.dates]
        )

        # Build observation matrix (T x n)
        self.Y_raw = data[self.var_names].values.astype(np.float64)

        # Force quarterly vars to NaN at non-quarter-end months
        for t in range(self.T):
            if not self.is_quarter_end[t]:
                self.Y_raw[t, : self.n_Q] = np.nan

        # Compute standardization parameters from available obs
        self.means = np.nanmean(self.Y_raw, axis=0)
        self.stds = np.nanstd(self.Y_raw, axis=0, ddof=0)
        self.stds[self.stds < 1e-12] = 1.0

        # Standardized data
        self.Y = (self.Y_raw - self.means) / self.stds

    def get_balanced_monthly_panel(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Find the largest balanced panel of monthly variables for PCA.

        Uses a greedy strategy: iteratively drops the variable that most
        constrains the balanced panel length until a panel of at least
        min_balanced_years * 12 months is achieved.

        Returns
        -------
        panel : (T_bal, n_bal) array
            Standardized balanced panel (no NaN).
        var_mask : (n_M,) bool array
            Which monthly variables are included.
        t_start, t_end : int
            Time indices [t_start, t_end) of the balanced panel.
        """
        monthly = self.Y[:, self.n_Q :]  # (T, n_M)
        min_T = self.min_balanced_years * 12
        valid = ~np.isnan(monthly)

        # Start with variables that have enough total observations
        obs_counts = valid.sum(axis=0)
        included = np.where(obs_counts >= min_T)[0]

        if len(included) == 0:
            raise ValueError(
                f"No monthly variables have >= {min_T} observations. "
                f"Max: {obs_counts.max()}"
            )

        # Iteratively find balanced panel, dropping problematic vars
        while len(included) > 1:
            all_present = valid[:, included].all(axis=1)
            t_start, best_len = _longest_true_run(all_present)

            if best_len >= min_T:
                break

            # Drop the variable whose contiguous valid range is shortest
            # within the current set's valid time window
            worst_idx = -1
            worst_score = np.inf
            for k, j in enumerate(included):
                # Score: how much this variable contributes to gaps
                candidate = np.delete(included, k)
                candidate_present = valid[:, candidate].all(axis=1)
                _, cand_len = _longest_true_run(candidate_present)
                # We want to drop the variable that, when removed,
                # gives the longest balanced panel
                if cand_len > worst_score or worst_idx == -1:
                    worst_score = cand_len
                    worst_idx = k

            included = np.delete(included, worst_idx)
        else:
            # Single variable left
            all_present = valid[:, included].all(axis=1)
            t_start, best_len = _longest_true_run(all_present)

        if best_len < min_T:
            raise ValueError(
                f"Longest balanced panel is {best_len} months, need {min_T}. "
                f"Consider reducing min_balanced_years or cleaning data."
            )

        t_end = t_start + best_len
        var_mask = np.zeros(self.n_M, dtype=bool)
        var_mask[included] = True
        panel = monthly[t_start:t_end][:, var_mask]

        assert not np.any(np.isnan(panel)), "BUG: balanced panel contains NaN"
        return panel, var_mask, t_start, t_end


def _longest_true_run(arr: np.ndarray) -> Tuple[int, int]:
    """Find start and length of the longest contiguous True run in a bool array."""
    best_start = 0
    best_len = 0
    curr_start = 0
    curr_len = 0
    for i, val in enumerate(arr):
        if val:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > best_len:
                best_start = curr_start
                best_len = curr_len
        else:
            curr_len = 0
    return best_start, best_len
