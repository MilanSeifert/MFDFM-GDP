"""Fit the MFDFM on Swiss macro data and plot the Business Cycle Index.

Usage
-----
    python3 examples/run_swiss.py

Requires data/swiss.pkl to exist (run examples/fetch_data.py first).
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfdfm import MFDFM


def main():
    # Load data
    pkl_path = "data/swiss.pkl"
    if not os.path.exists(pkl_path):
        sys.exit("data/swiss.pkl not found — run examples/fetch_data.py first.")

    with open(pkl_path, "rb") as f:
        p = pickle.load(f)

    data = p["data"]
    quarterly_vars = p["quarterly_vars"]

    # Start sample at first GDP observation
    first_gdp = data["GDP"].first_valid_index()
    data = data.loc[first_gdp:]
    print(f"Loaded {data.shape[1]} variables, "
          f"{data.index[0].strftime('%Y-%m')} → {data.index[-1].strftime('%Y-%m')}")

    # Fit model
    print("Fitting MFDFM (r=4 factors, p=1 lag) ...")
    model = MFDFM(n_factors=4, n_lags=1)
    model.fit(data, quarterly_vars=quarterly_vars, gdp_var="GDP")
    print(model.summary())

    # Plot BCI vs GDP (both in QoQ log-diff units)
    bci = model.business_cycle_index
    gdp = data["GDP"].dropna()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(bci.index, bci.values, color="steelblue", linewidth=1.5,
            label="Business Cycle Index")
    ax.plot(gdp.index, gdp.values * 100, color="darkorange", linewidth=1.5,
            label="GDP (QoQ % change)", marker="o", markersize=3, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_title("Swiss Business Cycle Index vs GDP growth")
    ax.set_ylabel("QoQ % change")
    ax.legend()

    plt.tight_layout()
    out_path = "mfdfm_swiss.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()