"""Shared fixtures for MFDFM tests."""

import numpy as np
import pandas as pd
import pytest


def generate_synthetic_mfdfm_data(
    T: int = 360,
    n_Q: int = 5,
    n_M: int = 20,
    r: int = 2,
    seed: int = 42,
    ragged_edge_months: int = 2,
) -> dict:
    """Generate synthetic mixed-frequency data from a known DGP.

    Parameters
    ----------
    T : int
        Number of monthly periods.
    n_Q : int
        Number of quarterly variables (including GDP as first).
    n_M : int
        Number of monthly variables.
    r : int
        Number of true factors.
    seed : int
        Random seed.
    ragged_edge_months : int
        Number of months of missing data at the end (ragged edge).

    Returns
    -------
    dict with:
        data : DataFrame
        quarterly_vars : list
        gdp_var : str
        true_factors : (T, r) array
        true_Lambda : (n, r) array
        true_Phi : (r, r) array
        true_Sigma_vv : (r, r) array
        true_Sigma_ww : (n,) array
        true_bci : (T,) array
    """
    rng = np.random.RandomState(seed)
    n = n_Q + n_M

    # Generate stable VAR(1) coefficient matrix
    # Use a diagonal-dominant matrix for stability
    Phi = rng.randn(r, r) * 0.3
    # Make eigenvalues < 1
    eigvals = np.linalg.eigvals(Phi)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.95:
        Phi *= 0.9 / max_eig

    # Innovation covariance
    A = rng.randn(r, r) * 0.3
    Sigma_vv = A @ A.T + 0.1 * np.eye(r)

    # Generate factors from VAR(1)
    factors = np.zeros((T, r))
    for t in range(1, T):
        v = rng.multivariate_normal(np.zeros(r), Sigma_vv)
        factors[t] = Phi @ factors[t - 1] + v

    # Factor loadings
    Lambda = rng.randn(n, r) * 0.5 + 0.3
    # Ensure GDP has positive loading on factor 1
    Lambda[0, 0] = abs(Lambda[0, 0]) + 0.5

    # Idiosyncratic variances
    Sigma_ww = np.abs(rng.randn(n)) * 0.3 + 0.2

    # Generate latent monthly data: y* = Lambda @ f + u
    u = np.zeros((T, n))
    for i in range(n):
        u[:, i] = rng.normal(0, np.sqrt(Sigma_ww[i]), T)
    y_star = factors @ Lambda.T + u  # (T, n) latent monthly

    # Build observed data
    Y = np.full((T, n), np.nan)

    # Monthly variables: directly observed
    Y[:, n_Q:] = y_star[:, n_Q:]

    # Quarterly variables: temporal aggregation G(L) = 1/3(1 + L + L^2)
    for t in range(2, T):
        month = (t % 12) + 1  # 1-indexed month
        # Quarter-end months: March(3), June(6), Sept(9), Dec(12)
        if month in (3, 6, 9, 12):
            for i in range(n_Q):
                Y[t, i] = (y_star[t, i] + y_star[t - 1, i] + y_star[t - 2, i]) / 3.0

    # Add ragged edge: remove last few months of some variables
    if ragged_edge_months > 0:
        # Remove last 'ragged_edge_months' from ~30% of monthly vars
        n_ragged = max(1, n_M // 3)
        ragged_vars = rng.choice(n_M, n_ragged, replace=False)
        for j in ragged_vars:
            Y[-ragged_edge_months:, n_Q + j] = np.nan

    # Add ragged start: some monthly vars start later
    n_late_start = max(1, n_M // 5)
    late_vars = rng.choice(n_M, n_late_start, replace=False)
    for j in late_vars:
        late_months = rng.randint(12, 60)
        Y[:late_months, n_Q + j] = np.nan

    # Create DataFrame
    dates = pd.date_range("1990-01-01", periods=T, freq="MS")
    quarterly_names = [f"Q_{i}" for i in range(n_Q)]
    quarterly_names[0] = "GDP"
    monthly_names = [f"M_{j}" for j in range(n_M)]
    columns = quarterly_names + monthly_names

    df = pd.DataFrame(Y, index=dates, columns=columns)

    # True BCI = Lambda_gdp @ f_t
    true_bci = factors @ Lambda[0]

    return {
        "data": df,
        "quarterly_vars": quarterly_names,
        "gdp_var": "GDP",
        "true_factors": factors,
        "true_Lambda": Lambda,
        "true_Phi": Phi,
        "true_Sigma_vv": Sigma_vv,
        "true_Sigma_ww": Sigma_ww,
        "true_bci": true_bci,
    }


@pytest.fixture
def synthetic_data():
    """Standard synthetic dataset for testing."""
    return generate_synthetic_mfdfm_data()


@pytest.fixture
def small_synthetic_data():
    """Small dataset for fast tests."""
    return generate_synthetic_mfdfm_data(T=240, n_Q=3, n_M=10, r=2, seed=123)
