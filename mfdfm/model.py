"""Main MFDFM class — the public API.

Implements the mixed-frequency dynamic factor model of Galli (2017) using
a two-step estimation procedure (PCA + OLS, then Kalman smoother).
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from mfdfm.data import MFData
from mfdfm.kalman import KalmanResult, kalman_filter, kalman_smoother
from mfdfm.estimation import (
    estimate_loadings,
    estimate_pca_factors,
    estimate_var,
    select_n_factors,
    select_n_lags,
)
from mfdfm.decomposition import compute_bci_weights, compute_news_revision

logger = logging.getLogger(__name__)


class MFDFM:
    """Mixed-Frequency Dynamic Factor Model.

    Produces a monthly business cycle index (BCI) by combining monthly and
    quarterly indicators via a state-space DFM estimated with the Kalman
    smoother, following Galli (2017).

    Parameters
    ----------
    n_factors : int or "auto"
        Number of latent factors (r). Default 4. If "auto", selected via BIC.
    n_lags : int or "auto"
        Number of VAR lags (p). Default 1. If "auto", selected via BIC.
    min_balanced_years : int
        Minimum years for the balanced PCA panel. Default 15.

    Examples
    --------
    >>> model = MFDFM(n_factors=4, n_lags=1)
    >>> model.fit(data, quarterly_vars=['GDP', ...], gdp_var='GDP')
    >>> bci = model.business_cycle_index
    """

    def __init__(
        self,
        n_factors: Union[int, str] = 4,
        n_lags: Union[int, str] = 1,
        min_balanced_years: int = 15,
    ):
        self.n_factors_input = n_factors
        self.n_lags_input = n_lags
        self.min_balanced_years = min_balanced_years

        # Populated after fit()
        self.data_: Optional[MFData] = None
        self.r_: Optional[int] = None
        self.p_: Optional[int] = None
        self.Lambda_: Optional[np.ndarray] = None
        self.Sigma_ww_: Optional[np.ndarray] = None
        self.Phi_: Optional[np.ndarray] = None
        self.Sigma_vv_: Optional[np.ndarray] = None
        self.F_: Optional[np.ndarray] = None
        self.Q_: Optional[np.ndarray] = None
        self.kf_result_: Optional[KalmanResult] = None
        self.f_pc_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.kf_result_ is not None

    def fit(
        self,
        data: pd.DataFrame,
        quarterly_vars: List[str],
        gdp_var: str,
    ) -> "MFDFM":
        """Fit the model.

        Parameters
        ----------
        data : DataFrame
            Panel data with DatetimeIndex (monthly frequency).
        quarterly_vars : list of str
            Column names of quarterly indicators.
        gdp_var : str
            Name of the GDP column (must be in quarterly_vars).

        Returns
        -------
        self
        """
        # --- Prepare data ---
        self.data_ = MFData(
            data, quarterly_vars, gdp_var, self.min_balanced_years
        )
        mfd = self.data_

        # --- Select hyperparameters ---
        if self.n_factors_input == "auto":
            self.r_, bic_r = select_n_factors(mfd, max_r=8)
            logger.info("BIC selected r=%d factors", self.r_)
        else:
            self.r_ = int(self.n_factors_input)

        if self.n_lags_input == "auto":
            self.p_, bic_p = select_n_lags(mfd, self.r_, max_p=6)
            logger.info("BIC selected p=%d lags", self.p_)
        else:
            self.p_ = int(self.n_lags_input)

        r, p = self.r_, self.p_

        # --- Step 1: Parameter estimation via PCA + OLS ---
        logger.info("Step 1: PCA + OLS estimation (r=%d, p=%d)", r, p)
        self.f_pc_, eigenvalues, eigvecs, t_start, t_end = estimate_pca_factors(
            mfd, r
        )
        self.Lambda_, self.Sigma_ww_ = estimate_loadings(mfd, self.f_pc_, r)
        self.Phi_, self.Sigma_vv_ = estimate_var(self.f_pc_, p)

        # Sign convention: ensure GDP loading on first factor is positive
        if self.Lambda_[0, 0] < 0:
            self.Lambda_[:, 0] *= -1
            self.f_pc_[:, 0] *= -1
            self.Phi_[0, :] *= -1
            self.Phi_[:, 0] *= -1

        # --- Build state-space matrices ---
        self.F_, self.Q_ = self._build_transition(r, p)

        # --- Step 2: Kalman filter + smoother ---
        logger.info(
            "Step 2: Kalman smoother (T=%d, n=%d, state_dim=%d)",
            mfd.T,
            mfd.n,
            3 * r,
        )
        H_func, R_func = self._build_measurement(r)
        xi_0, P_0 = self._initial_state(r)

        self.kf_result_ = kalman_filter(
            mfd.Y, H_func, R_func, self.F_, self.Q_, xi_0, P_0
        )
        self.kf_result_ = kalman_smoother(self.kf_result_, self.F_)

        logger.info("Model fitted. Log-likelihood: %.2f", self.kf_result_.log_lik)
        return self

    def _build_transition(
        self, r: int, p: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build companion-form transition matrix F and noise covariance Q.

        State vector: xi_t = [f_t; f_{t-1}; f_{t-2}], dimension 3r.

        F = [Phi_1 ... Phi_p  0_{r x (3-p)r}]
            [I_{2r}           0_{2r x r}     ]

        Q = [Sigma_vv  0      0    ]
            [0         0_{rxr} 0   ]
            [0         0      0_{rxr}]
        """
        s = 3 * r
        F = np.zeros((s, s))

        # Top block: [Phi_1 ... Phi_p | zeros]
        rp = r * p
        F[:r, :rp] = self.Phi_  # (r, r*p)

        # Identity blocks for lags
        F[r : 3 * r, : 2 * r] = np.eye(2 * r)

        Q = np.zeros((s, s))
        Q[:r, :r] = self.Sigma_vv_

        return F, Q

    def _build_measurement(self, r: int):
        """Build H_func(t) and R_func(t) for the measurement equation.

        For quarterly variable i at quarter-end months:
            H[i, :] = [1/3 Lambda_i, 1/3 Lambda_i, 1/3 Lambda_i]
            R[i]    = (1/3) sigma^2_i

        For monthly variable i:
            H[i, :] = [Lambda_i, 0, 0]
            R[i]    = sigma^2_i

        Returns callables so the Kalman filter can evaluate H_t, R_t per step.
        """
        n = self.data_.n
        n_Q = self.data_.n_Q
        s = 3 * r
        Lambda = self.Lambda_
        Sigma_ww = self.Sigma_ww_

        # Pre-build H (constant since NaN selection handles time variation)
        H = np.zeros((n, s))
        # Quarterly rows: temporal aggregation
        H[:n_Q, :r] = Lambda[:n_Q] / 3.0
        H[:n_Q, r : 2 * r] = Lambda[:n_Q] / 3.0
        H[:n_Q, 2 * r : 3 * r] = Lambda[:n_Q] / 3.0
        # Monthly rows: direct observation
        H[n_Q:, :r] = Lambda[n_Q:]

        # Pre-build R diagonal (constant for same reason)
        R_diag = np.empty(n)
        R_diag[:n_Q] = Sigma_ww[:n_Q] / 3.0  # quarterly: (1/3) * sigma^2
        R_diag[n_Q:] = Sigma_ww[n_Q:]         # monthly: sigma^2

        # Ensure positive for numerical stability
        R_diag = np.maximum(R_diag, 1e-10)

        def H_func(t: int) -> np.ndarray:
            return H

        def R_func(t: int) -> np.ndarray:
            return R_diag

        return H_func, R_func

    def _initial_state(self, r: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute initial state for the Kalman filter.

        xi_0 = 0 (factors are mean-zero).
        P_0  = solution of discrete Lyapunov equation F P F' + Q = P.
        """
        s = 3 * r
        xi_0 = np.zeros(s)

        try:
            P_0 = solve_discrete_lyapunov(self.F_, self.Q_)
        except Exception:
            # Fallback: large diagonal
            logger.warning(
                "Lyapunov solver failed, using diffuse initialization"
            )
            P_0 = 10.0 * np.eye(s)

        return xi_0, P_0

    # --- Properties for accessing results ---

    @property
    def smoothed_factors(self) -> np.ndarray:
        """(T, r) array of smoothed factor estimates f_{t|T}."""
        self._check_fitted()
        return self.kf_result_.xi_smooth[:, : self.r_]

    @property
    def smoothed_states(self) -> np.ndarray:
        """(T, 3r) array of full smoothed state vector xi_{t|T}."""
        self._check_fitted()
        return self.kf_result_.xi_smooth

    @property
    def business_cycle_index(self) -> pd.Series:
        """Monthly BCI = Lambda_gdp @ f_{t|T}.

        The BCI is the monthly fitted value for GDP from the Kalman smoother.
        It is in standardized units of quarterly GDP.
        """
        self._check_fitted()
        Lambda_gdp = self.Lambda_[0]  # (r,) GDP loadings
        f_smooth = self.smoothed_factors  # (T, r)
        bci = f_smooth @ Lambda_gdp
        return pd.Series(bci, index=self.data_.dates, name="BCI")

    @property
    def bci_variance(self) -> pd.Series:
        """V(BCI_t) = Lambda_gdp P^f_{t|T} Lambda_gdp'.

        Measures the uncertainty / finality of the BCI estimate at each t.
        """
        self._check_fitted()
        Lambda_gdp = self.Lambda_[0]  # (r,)
        r = self.r_
        P_f = self.kf_result_.P_smooth[:, :r, :r]  # (T, r, r)
        var_bci = np.array(
            [Lambda_gdp @ P_f[t] @ Lambda_gdp for t in range(self.data_.T)]
        )
        return pd.Series(var_bci, index=self.data_.dates, name="BCI_variance")

    @property
    def bci_accuracy(self) -> pd.Series:
        """Relative accuracy = 1 - V(BCI_t) / V(BCI_t) at final.

        Values near 1 indicate high finality (most information incorporated).
        """
        var = self.bci_variance
        final = var.iloc[-1]
        if final < 1e-15:
            return pd.Series(
                np.ones(len(var)), index=var.index, name="BCI_accuracy"
            )
        return pd.Series(
            1.0 - var.values / var.values[-1],
            index=var.index,
            name="BCI_accuracy",
        )

    def fitted_values(self, var_name: Optional[str] = None) -> pd.DataFrame:
        """Compute fitted values y_hat = H xi_{t|T} for all or a given variable.

        Parameters
        ----------
        var_name : str, optional
            If given, return fitted values for this variable only.
        """
        self._check_fitted()
        xi = self.kf_result_.xi_smooth  # (T, s)
        r = self.r_
        n_Q = self.data_.n_Q
        Lambda = self.Lambda_

        # Compute fitted latent monthly values: y*_hat = Lambda @ f_{t|T}
        f = xi[:, :r]  # (T, r)
        y_star_hat = f @ Lambda.T  # (T, n) = fitted latent monthly

        # For quarterly variables at quarter-end months, apply G(L)
        fitted = np.full_like(y_star_hat, np.nan)
        for t in range(self.data_.T):
            # Monthly variables: direct
            fitted[t, n_Q:] = y_star_hat[t, n_Q:]
            # Quarterly variables: only at quarter-end, with aggregation
            if self.data_.is_quarter_end[t] and t >= 2:
                fitted[t, :n_Q] = (
                    y_star_hat[t, :n_Q]
                    + y_star_hat[t - 1, :n_Q]
                    + y_star_hat[t - 2, :n_Q]
                ) / 3.0

        result = pd.DataFrame(
            fitted, index=self.data_.dates, columns=self.data_.var_names
        )

        if var_name is not None:
            return result[[var_name]]
        return result

    def observation_weights(self) -> Dict[str, np.ndarray]:
        """Compute observation weights for the BCI (Koopman & Harvey, 2003).

        Returns
        -------
        dict with:
            'factor_weights': (T, T, n) — weight of indicator (k, i) on f_{t|T}
            'bci_weights': (T, n) — weight of each indicator on the BCI at t
        """
        self._check_fitted()
        return compute_bci_weights(self)

    def news_decomposition(
        self,
        old_data: pd.DataFrame,
        new_data: pd.DataFrame,
        target_period: Optional[pd.Timestamp] = None,
    ) -> Dict[str, np.ndarray]:
        """Decompose BCI revision into news contributions.

        Parameters
        ----------
        old_data : DataFrame
            Previous vintage of data.
        new_data : DataFrame
            Updated vintage of data.
        target_period : Timestamp, optional
            Target month for decomposition. Defaults to the latest
            quarter-end month in the data.

        Returns
        -------
        dict with:
            'revision': float — total BCI revision
            'news': (J,) — news content of each new data release
            'weights': (J,) — weight of each news item on the BCI
            'contributions': (J,) — weighted news (revision breakdown)
            'indicators': list — names of the new releases
        """
        self._check_fitted()
        return compute_news_revision(self, old_data, new_data, target_period)

    def predict(self, h: int = 1) -> pd.DataFrame:
        """Forecast h periods ahead using the smoothed state at T.

        Parameters
        ----------
        h : int
            Number of months to forecast.

        Returns
        -------
        DataFrame of forecasted values (h, n).
        """
        self._check_fitted()
        xi = self.kf_result_.xi_smooth[-1]  # (s,)
        r = self.r_

        forecasts = []
        for step in range(1, h + 1):
            xi = self.F_ @ xi
            f = xi[:r]
            y_hat = self.Lambda_ @ f  # (n,)
            forecasts.append(y_hat)

        dates = pd.date_range(
            start=self.data_.dates[-1] + pd.DateOffset(months=1),
            periods=h,
            freq="MS",
        )
        return pd.DataFrame(
            np.array(forecasts),
            index=dates,
            columns=self.data_.var_names,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted model to disk."""
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Union[str, Path]) -> "MFDFM":
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> str:
        """Return a summary string of the fitted model."""
        self._check_fitted()
        lines = [
            "Mixed-Frequency Dynamic Factor Model",
            "=" * 45,
            f"Factors (r):          {self.r_}",
            f"VAR lags (p):         {self.p_}",
            f"State dimension:      {3 * self.r_}",
            f"Quarterly variables:  {self.data_.n_Q}",
            f"Monthly variables:    {self.data_.n_M}",
            f"Total variables:      {self.data_.n}",
            f"Time periods:         {self.data_.T}",
            f"Sample:               {self.data_.dates[0].strftime('%Y-%m')} "
            f"to {self.data_.dates[-1].strftime('%Y-%m')}",
            f"Log-likelihood:       {self.kf_result_.log_lik:.2f}",
            "",
            "GDP loadings (Lambda_gdp):",
            f"  {self.Lambda_[0]}",
            "",
            "VAR(1) coefficient eigenvalues:",
            f"  {np.abs(np.linalg.eigvals(self.Phi_))}",
        ]
        return "\n".join(lines)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
