"""Out-of-sample nowcast evaluation of the Swiss MFDFM.

For each quarter in the test period, the model is re-fitted on data up to
that quarter's end with GDP for the quarter set to NaN (simulating real-time,
since GDP is released ~2 months after quarter end). The BCI at the quarter-end
month is the nowcast — what the model thinks GDP is based on the other
indicators — and is compared to the actual GDP release.

Usage
-----
    python3 examples/evaluate_swiss.py

Requires data/swiss.pkl (run examples/fetch_data.py first).
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfdfm import MFDFM


TEST_START = "2015-01-01"  # first test quarter (inclusive)


def main():
    pkl_path = "data/swiss.pkl"
    if not os.path.exists(pkl_path):
        sys.exit("data/swiss.pkl not found — run examples/fetch_data.py first.")

    with open(pkl_path, "rb") as f:
        p = pickle.load(f)

    data = p["data"]
    quarterly_vars = p["quarterly_vars"]

    # Start at first GDP observation (earlier rows are useless for this eval)
    data = data.loc[data["GDP"].first_valid_index():]

    # Quarter-end months in the test period where GDP is actually observed
    gdp = data["GDP"].dropna()
    test_quarters = gdp.index[gdp.index >= TEST_START]
    print(f"Evaluating on {len(test_quarters)} quarters: "
          f"{test_quarters[0].strftime('%Y-%m')} → {test_quarters[-1].strftime('%Y-%m')}")

    nowcasts = []
    actuals = []
    for q_end in test_quarters:
        # Expanding window: data up to and including the quarter-end month,
        # but with GDP for this quarter masked out (simulates pre-release).
        data_t = data.loc[:q_end].copy()
        data_t.loc[q_end, "GDP"] = np.nan

        model = MFDFM(n_factors=4, n_lags=1)
        model.fit(data_t, quarterly_vars=quarterly_vars, gdp_var="GDP")

        nowcasts.append(model.business_cycle_index.loc[q_end])
        actuals.append(gdp.loc[q_end])
        print(f"  {q_end.strftime('%Y-%m')}: "
              f"nowcast={nowcasts[-1]:+.4f}  actual={actuals[-1]:+.4f}")

    nowcasts = np.array(nowcasts)
    actuals = np.array(actuals)

    # The BCI is not calibrated in GDP units — it's Λ_gdp * f where factors are
    # from standardised data. To get comparable RMSE, rescale the BCI by
    # matching mean and std to GDP's training-period moments.
    gdp_train = gdp.loc[:TEST_START]
    mu_g, sd_g = gdp_train.mean(), gdp_train.std()
    mu_n, sd_n = nowcasts.mean(), nowcasts.std()
    nowcasts_scaled = (nowcasts - mu_n) / sd_n * sd_g + mu_g

    # Metrics (scaled and scale-free)
    mse = np.mean((nowcasts_scaled - actuals) ** 2)
    mse_base = np.mean((mu_g - actuals) ** 2)  # predict training mean
    rmse = np.sqrt(mse)
    rmse_base = np.sqrt(mse_base)
    oos_r2 = 1 - mse / mse_base  # can be negative if worse than baseline
    corr = np.corrcoef(nowcasts, actuals)[0, 1]
    dir_acc = np.mean(np.sign(nowcasts - mu_n) == np.sign(actuals - mu_g))

    print("\n" + "=" * 50)
    print("Out-of-sample nowcast evaluation")
    print("=" * 50)
    print(f"  N quarters                    : {len(actuals)}")
    print(f"  Correlation (BCI vs GDP)      : {corr:+.3f}")
    print(f"  Directional accuracy          : {dir_acc:.1%}")
    print(f"  RMSE after rescaling (pp)     : {rmse*100:.2f}")
    print(f"  RMSE baseline (constant mean) : {rmse_base*100:.2f}")
    print(f"  Out-of-sample R^2             : {oos_r2:+.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_quarters, actuals * 100, color="darkorange", linewidth=1.5,
            label="Actual GDP (QoQ %)", marker="o", markersize=5)
    ax.plot(test_quarters, nowcasts_scaled * 100, color="steelblue",
            linewidth=1.5, label="Nowcast (rescaled, QoQ %)",
            marker="s", markersize=5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_title(
        f"Out-of-sample nowcast: Swiss GDP growth  "
        f"(corr={corr:.2f}, dir-acc={dir_acc:.0%})"
    )
    ax.set_ylabel("QoQ % change")
    ax.legend()
    plt.tight_layout()
    plt.savefig("mfdfm_swiss_eval.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot: mfdfm_swiss_eval.png")


if __name__ == "__main__":
    main()
