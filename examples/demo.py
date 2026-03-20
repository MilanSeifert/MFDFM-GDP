"""Demo: Mixed-Frequency Dynamic Factor Model (Galli, 2017).

Generates synthetic mixed-frequency data, fits the MFDFM, and
produces diagnostic plots and a summary.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfdfm import MFDFM


def generate_demo_data(T=480, n_Q=8, n_M=30, r=4, seed=2017):
    """Generate realistic synthetic mixed-frequency data.

    Mimics the structure of the Swiss business cycle data set:
    - Multiple quarterly variables (GDP components)
    - Many monthly variables (surveys, hard data, financial)
    - Ragged edges and ragged starts
    - 40 years of monthly data
    """
    rng = np.random.RandomState(seed)
    n = n_Q + n_M

    # 4-factor VAR(1) — as in the paper
    Phi = np.zeros((r, r))
    Phi[0, 0] = 0.85  # persistent main business cycle factor
    Phi[1, 1] = 0.70  # moderately persistent
    Phi[2, 2] = 0.60  # less persistent
    Phi[3, 3] = 0.50  # transitory factor
    # Cross-factor dynamics
    Phi[0, 3] = 0.10  # financial → business cycle (lagged)

    Sigma_vv = np.diag([0.15, 0.20, 0.25, 0.30])

    # Generate factors
    factors = np.zeros((T, r))
    for t in range(1, T):
        factors[t] = Phi @ factors[t - 1] + rng.multivariate_normal(
            np.zeros(r), Sigma_vv
        )

    # Factor loadings — structured by indicator type
    Lambda = np.zeros((n, r))

    # GDP and quarterly variables: load mainly on factor 1
    for i in range(n_Q):
        Lambda[i, 0] = rng.uniform(0.3, 0.8)
        Lambda[i, 1] = rng.uniform(-0.2, 0.3)
        Lambda[i, 2] = rng.uniform(-0.1, 0.2)
        Lambda[i, 3] = rng.uniform(-0.1, 0.1)
    Lambda[0, 0] = 0.7  # GDP has strong loading on factor 1

    # Monthly variables: diverse loadings
    for j in range(n_M):
        i = n_Q + j
        if j < 10:  # "hard data" — load on factors 1, 2
            Lambda[i, 0] = rng.uniform(0.2, 0.6)
            Lambda[i, 1] = rng.uniform(0.1, 0.4)
            Lambda[i, 2] = rng.uniform(-0.1, 0.2)
        elif j < 20:  # "soft data / surveys"
            Lambda[i, 0] = rng.uniform(0.2, 0.5)
            Lambda[i, 1] = rng.uniform(-0.1, 0.3)
            Lambda[i, 2] = rng.uniform(0.1, 0.4)
        else:  # "financial"
            Lambda[i, 0] = rng.uniform(0.1, 0.3)
            Lambda[i, 3] = rng.uniform(0.3, 0.7)

    # Idiosyncratic variances
    Sigma_ww = rng.uniform(0.2, 0.8, n)

    # Generate latent monthly data
    u = np.zeros((T, n))
    for i in range(n):
        u[:, i] = rng.normal(0, np.sqrt(Sigma_ww[i]), T)
    y_star = factors @ Lambda.T + u

    # Build observed data
    dates = pd.date_range("1980-01-01", periods=T, freq="MS")
    Y = np.full((T, n), np.nan)

    # Monthly variables
    Y[:, n_Q:] = y_star[:, n_Q:]

    # Quarterly variables with temporal aggregation
    for t in range(2, T):
        if dates[t].month in (3, 6, 9, 12):
            for i in range(n_Q):
                Y[t, i] = (
                    y_star[t, i] + y_star[t - 1, i] + y_star[t - 2, i]
                ) / 3.0

    # Ragged starts: some monthly vars start later
    start_years = [0, 0, 0, 0, 0, 5, 5, 8, 10, 10,  # hard data
                   0, 0, 2, 3, 5, 5, 8, 10, 12, 15,  # surveys
                   0, 0, 0, 0, 5, 5, 8, 10, 12, 15]  # financial
    for j in range(min(n_M, len(start_years))):
        start_months = start_years[j] * 12
        if start_months > 0:
            Y[:start_months, n_Q + j] = np.nan

    # Ragged edge: last 1-3 months missing for some vars
    for j in range(n_M):
        lag = rng.choice([0, 0, 1, 1, 2, 3])
        if lag > 0:
            Y[-lag:, n_Q + j] = np.nan

    # Variable names
    q_names = ["GDP"] + [f"GDP_comp_{i}" for i in range(1, n_Q)]
    m_names = (
        [f"hard_{j}" for j in range(10)]
        + [f"survey_{j}" for j in range(10)]
        + [f"financial_{j}" for j in range(10)]
    )
    cols = q_names + m_names[:n_M]

    df = pd.DataFrame(Y, index=dates, columns=cols)
    true_bci = factors @ Lambda[0]

    return df, q_names, true_bci, factors


def main():
    print("=" * 60)
    print("Mixed-Frequency Dynamic Factor Model — Demo")
    print("=" * 60)

    # Generate data
    print("\n1. Generating synthetic data...")
    data, quarterly_vars, true_bci, true_factors = generate_demo_data()
    print(f"   {data.shape[1]} variables, {data.shape[0]} months")
    print(f"   {len(quarterly_vars)} quarterly, "
          f"{data.shape[1] - len(quarterly_vars)} monthly")
    print(f"   Sample: {data.index[0].strftime('%Y-%m')} to "
          f"{data.index[-1].strftime('%Y-%m')}")

    # Fit model
    print("\n2. Fitting MFDFM (r=4 factors, p=1 lag)...")
    model = MFDFM(n_factors=4, n_lags=1)
    model.fit(data, quarterly_vars=quarterly_vars, gdp_var="GDP")
    print(model.summary())

    # Evaluate BCI
    bci = model.business_cycle_index
    corr = np.corrcoef(bci.values, true_bci)[0, 1]
    print(f"\n3. BCI correlation with true DGP: {corr:.4f}")

    # Variance / accuracy
    var = model.bci_variance
    print(f"   BCI variance range: [{var.min():.4f}, {var.max():.4f}]")

    # Forecasting
    forecast = model.predict(h=6)
    print(f"\n4. 6-month forecast generated ({forecast.shape})")

    # Observation weights
    weights = model.observation_weights()
    bci_w = weights["bci_weights"]
    # Sum absolute weights per variable
    abs_w = np.abs(bci_w).sum(axis=0)
    abs_w_norm = abs_w / abs_w.sum()
    top_10 = np.argsort(-abs_w_norm)[:10]
    print("\n5. Top 10 indicators by BCI weight:")
    for rank, idx in enumerate(top_10, 1):
        name = model.data_.var_names[idx]
        print(f"   {rank:2d}. {name:20s} ({abs_w_norm[idx]:.3f})")

    # Plot
    print("\n6. Generating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: BCI vs true
    ax = axes[0]
    ax.plot(data.index, true_bci, label="True BCI (DGP)", alpha=0.7)
    ax.plot(bci.index, bci.values, label="Estimated BCI", linewidth=1.5)
    ax.set_title(f"Business Cycle Index (correlation = {corr:.3f})")
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)

    # Panel 2: Smoothed factors
    ax = axes[1]
    factors_est = model.smoothed_factors
    for j in range(min(4, factors_est.shape[1])):
        ax.plot(
            data.index, factors_est[:, j],
            label=f"Factor {j+1}", alpha=0.8,
        )
    ax.set_title("Smoothed Factors")
    ax.legend()

    # Panel 3: BCI accuracy
    ax = axes[2]
    acc = model.bci_accuracy
    ax.plot(acc.index, acc.values)
    ax.set_title("BCI Accuracy (relative to final estimate)")
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig("mfdfm_demo.png", dpi=150, bbox_inches="tight")
    print("   Saved: mfdfm_demo.png")

    # Save model
    model.save("mfdfm_demo.pkl")
    print("   Saved: mfdfm_demo.pkl")

    # Verify save/load
    loaded = MFDFM.load("mfdfm_demo.pkl")
    bci_loaded = loaded.business_cycle_index
    assert np.allclose(bci.values, bci_loaded.values)
    print("   Save/load verified.")

    print("\nDone.")


if __name__ == "__main__":
    main()
