"""Kalman filter and Rauch-Tung-Striebel smoother.

Handles the linear Gaussian state-space model with time-varying measurement
matrices and missing observations:

    xi_t = F xi_{t-1} + e_t,    e_t ~ N(0, Q)
    y_t  = H_t xi_t  + eps_t,  eps_t ~ N(0, R_t)

Missing observations (NaN in y) are excluded from the update step at each t.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class KalmanResult:
    """Output from the Kalman filter and smoother."""

    # Filter output (T, s) and (T, s, s)
    xi_pred: np.ndarray
    P_pred: np.ndarray
    xi_filt: np.ndarray
    P_filt: np.ndarray
    log_lik: float

    # Per-step storage (variable length due to missing obs)
    innovations: List[np.ndarray]
    inn_cov: List[np.ndarray]
    kalman_gains: List[np.ndarray]
    obs_idx: List[np.ndarray]
    H_obs_list: List[Optional[np.ndarray]]

    # Smoother output (filled after smoothing)
    xi_smooth: Optional[np.ndarray] = None
    P_smooth: Optional[np.ndarray] = None
    J: Optional[np.ndarray] = None


def kalman_filter(
    y: np.ndarray,
    H_func: Callable[[int], np.ndarray],
    R_func: Callable[[int], np.ndarray],
    F: np.ndarray,
    Q: np.ndarray,
    xi_0: np.ndarray,
    P_0: np.ndarray,
) -> KalmanResult:
    """Run the Kalman filter.

    Parameters
    ----------
    y : (T, n) array
        Observations. NaN = missing.
    H_func : callable(t) -> (n, s) array
        Measurement matrix at time t.
    R_func : callable(t) -> (n,) array
        Diagonal of measurement noise covariance at time t.
    F : (s, s) array
        Transition matrix.
    Q : (s, s) array
        Transition noise covariance.
    xi_0 : (s,) array
        Initial state mean.
    P_0 : (s, s) array
        Initial state covariance.
    """
    T, n = y.shape
    s = len(xi_0)

    xi_pred = np.zeros((T, s))
    P_pred = np.zeros((T, s, s))
    xi_filt = np.zeros((T, s))
    P_filt = np.zeros((T, s, s))

    innovations: List[np.ndarray] = []
    inn_cov: List[np.ndarray] = []
    kalman_gains: List[np.ndarray] = []
    obs_idx: List[np.ndarray] = []
    H_obs_list: List[Optional[np.ndarray]] = []

    log_lik = 0.0
    xi_prev = xi_0.copy()
    P_prev = P_0.copy()

    for t in range(T):
        # --- Prediction ---
        xi_pred[t] = F @ xi_prev
        P_pred[t] = F @ P_prev @ F.T + Q
        P_pred[t] = _symmetrize(P_pred[t])

        # --- Determine observed variables ---
        H_full = H_func(t)
        R_diag = R_func(t)
        obs = np.where(~np.isnan(y[t]))[0]
        obs_idx.append(obs)

        if len(obs) == 0:
            xi_filt[t] = xi_pred[t]
            P_filt[t] = P_pred[t]
            innovations.append(np.empty(0))
            inn_cov.append(np.empty((0, 0)))
            kalman_gains.append(np.zeros((s, 0)))
            H_obs_list.append(None)
        else:
            H = H_full[obs]          # (m, s)
            R = np.diag(R_diag[obs]) # (m, m)
            y_obs = y[t, obs]        # (m,)

            # Innovation
            z = y_obs - H @ xi_pred[t]
            S = H @ P_pred[t] @ H.T + R
            S = _symmetrize(S)

            # Kalman gain: K = P H' S^{-1}
            K = np.linalg.solve(S, H @ P_pred[t]).T  # (s, m)

            # Update state
            xi_filt[t] = xi_pred[t] + K @ z

            # Update covariance (Joseph form for numerical stability)
            IKH = np.eye(s) - K @ H
            P_filt[t] = IKH @ P_pred[t] @ IKH.T + K @ R @ K.T
            P_filt[t] = _symmetrize(P_filt[t])

            # Log-likelihood contribution
            sign, logdet = np.linalg.slogdet(S)
            if sign > 0:
                log_lik -= 0.5 * (
                    len(obs) * np.log(2 * np.pi)
                    + logdet
                    + z @ np.linalg.solve(S, z)
                )

            innovations.append(z)
            inn_cov.append(S)
            kalman_gains.append(K)
            H_obs_list.append(H)

        xi_prev = xi_filt[t]
        P_prev = P_filt[t]

    return KalmanResult(
        xi_pred=xi_pred,
        P_pred=P_pred,
        xi_filt=xi_filt,
        P_filt=P_filt,
        log_lik=log_lik,
        innovations=innovations,
        inn_cov=inn_cov,
        kalman_gains=kalman_gains,
        obs_idx=obs_idx,
        H_obs_list=H_obs_list,
    )


def kalman_smoother(kf: KalmanResult, F: np.ndarray) -> KalmanResult:
    """Run the RTS smoother (Rauch-Tung-Striebel).

    Fills the xi_smooth, P_smooth, and J fields of the KalmanResult.
    """
    T, s = kf.xi_filt.shape

    xi_smooth = np.zeros((T, s))
    P_smooth = np.zeros((T, s, s))
    J = np.zeros((T, s, s))

    # Terminal condition: smoothed = filtered at T
    xi_smooth[T - 1] = kf.xi_filt[T - 1]
    P_smooth[T - 1] = kf.P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        # J_t = P_{t|t} F' P_{t+1|t}^{-1}
        PF = kf.P_filt[t] @ F.T  # (s, s)
        J[t] = np.linalg.solve(kf.P_pred[t + 1], PF.T).T

        xi_smooth[t] = kf.xi_filt[t] + J[t] @ (
            xi_smooth[t + 1] - kf.xi_pred[t + 1]
        )
        P_smooth[t] = kf.P_filt[t] + J[t] @ (
            P_smooth[t + 1] - kf.P_pred[t + 1]
        ) @ J[t].T
        P_smooth[t] = _symmetrize(P_smooth[t])

    kf.xi_smooth = xi_smooth
    kf.P_smooth = P_smooth
    kf.J = J
    return kf


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)