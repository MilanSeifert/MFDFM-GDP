"""Fetch Swiss macro data from FRED for use with the MFDFM model.

Downloads Swiss GDP (quarterly target) and a set of monthly/quarterly Swiss
macro indicators from FRED, applies the transformations required by the model
(3-month log-differences for monthly levels, QoQ log-differences for quarterly
levels), and saves a ready-to-use DataFrame as a CSV and a pickled dict.

Usage
-----
    export FRED_API_KEY="your_key_here"   # get a free key at fred.stlouisfed.org
    python3 examples/fetch_fred_data.py

    # Or pass the key directly:
    python3 examples/fetch_fred_data.py --api-key YOUR_KEY

    # Change output paths:
    python3 examples/fetch_fred_data.py --out-csv data/swiss.csv --out-pkl data/swiss.pkl

Output
------
CSV / PKL containing:
  - One column per indicator, monthly DatetimeIndex (month-start freq)
  - Quarterly variables: observed at quarter-end months (Mar/Jun/Sep/Dec), NaN elsewhere
  - Data already transformed to stationarity (see TRANSFORM column in SERIES catalog)
  - Ready to pass directly to MFDFM.fit(data, quarterly_vars=..., gdp_var="GDP")

Notes
-----
- A free FRED API key is required: https://fred.stlouisfed.org/docs/api/api_key.html
- Many Swiss series on FRED come from OECD MEI; verify codes at fred.stlouisfed.org
- Series flagged transform="rate" are assumed stationary (interest/unemployment rates);
  all others are log-differenced to remove trends.
- Seasonal adjustment: most OECD MEI series are already SA; raw series are noted.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Series catalog
# ---------------------------------------------------------------------------
# Each entry:
#   fred_id   : FRED series identifier
#   label     : short name used as DataFrame column
#   freq      : "M" (monthly) or "Q" (quarterly)
#   transform : "log_diff3"  → 3-month log-difference  (monthly levels/indices)
#               "log_diff_q" → QoQ log-difference      (quarterly levels)
#               "diff3"      → 3-month arithmetic diff  (monthly rates/%-points)
#               "diff_q"     → QoQ arithmetic diff      (quarterly rates)
#               "rate"       → leave as-is (already stationary rate/index)
#   sa        : True if the FRED series is already seasonally adjusted
#   notes     : free-text description
#
# All transformations follow Galli (2017):
#   - Monthly non-stationary series: Δ₃ ln x_t  = ln(x_t) − ln(x_{t−3})
#   - Quarterly non-stationary:      Δ₁ ln x_t  = ln(x_t) − ln(x_{t−1})  (QoQ)
#   - Stationary rates: no transformation needed
# ---------------------------------------------------------------------------

SERIES = [
    # ------------------------------------------------------------------
    # QUARTERLY — GDP and components  (target + auxiliary quarterly vars)
    # ------------------------------------------------------------------
    dict(
        fred_id="CHNGDPNQDSMEI",
        label="GDP",
        freq="Q",
        transform="log_diff_q",
        sa=True,
        notes="Switzerland Real GDP, chained volumes, SA (OECD MEI)",
    ),
    dict(
        fred_id="NAEXKP02CHQ657S",
        label="GDP_CONS",
        freq="Q",
        transform="log_diff_q",
        sa=True,
        notes="Switzerland Private Final Consumption Expenditure, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP03CHQ657S",
        label="GDP_GOV",
        freq="Q",
        transform="log_diff_q",
        sa=True,
        notes="Switzerland Government Final Consumption Expenditure, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP04CHQ657S",
        label="GDP_INVEST",
        freq="Q",
        transform="log_diff_q",
        sa=True,
        notes="Switzerland Gross Fixed Capital Formation, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP06CHQ657S",
        label="GDP_EXP",
        freq="Q",
        transform="log_diff_q",
        sa=True,
        notes="Switzerland Exports of Goods and Services, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP07CHQ657S",
        label="GDP_IMP",
        freq="Q",
        transform="log_diff_q",
        sa=True,
        notes="Switzerland Imports of Goods and Services, real SA (OECD)",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Hard data / production
    # ------------------------------------------------------------------
    dict(
        fred_id="CHLAMA01CHM661S",
        label="INDPRO",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Industrial Production Index (OECD MEI), SA",
    ),
    dict(
        fred_id="CHEPROINDAISMEI",
        label="PROIND",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Production in Industry, SA (OECD MEI)",
    ),
    dict(
        fred_id="CHNFACTOISMEI",
        label="MNFORDERS",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Manufacturing New Orders (OECD MEI)",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Labor market
    # ------------------------------------------------------------------
    dict(
        fred_id="CHEUNR",
        label="UNEMP",
        freq="M",
        transform="rate",
        sa=True,
        notes="Switzerland Unemployment Rate, SA (OECD)",
    ),
    dict(
        fred_id="CHEEMPRTT01CHM156S",
        label="EMPL",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Employment (OECD MEI)",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Prices
    # ------------------------------------------------------------------
    dict(
        fred_id="CHECPIALLMINMEI",
        label="CPI",
        freq="M",
        transform="log_diff3",
        sa=False,
        notes="Switzerland CPI All Items (OECD MEI), not SA",
    ),
    dict(
        fred_id="CPALTT01CHM657N",
        label="CPITOT",
        freq="M",
        transform="log_diff3",
        sa=False,
        notes="Switzerland CPI Total (OECD), not SA",
    ),
    dict(
        fred_id="CHEPPIALLMINMEI",
        label="PPI",
        freq="M",
        transform="log_diff3",
        sa=False,
        notes="Switzerland Producer Prices Index (OECD MEI)",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Financial / monetary
    # ------------------------------------------------------------------
    dict(
        fred_id="IRLTST01CHM156N",
        label="LTRATE",
        freq="M",
        transform="rate",
        sa=False,
        notes="Switzerland Long-Term Government Bond Yields (OECD MEI)",
    ),
    dict(
        fred_id="IRSTCI01CHM156N",
        label="STRATE",
        freq="M",
        transform="rate",
        sa=False,
        notes="Switzerland Immediate Call Money/Interbank Rate (OECD MEI)",
    ),
    dict(
        fred_id="IRSTCB01CHM156N",
        label="CB_RATE",
        freq="M",
        transform="rate",
        sa=False,
        notes="Switzerland Central Bank Policy Rate (OECD MEI)",
    ),
    dict(
        fred_id="DEXSZUS",
        label="CHFUSD",
        freq="M",
        transform="log_diff3",
        sa=False,
        notes="Switzerland / U.S. Foreign Exchange Rate (CHF per USD), daily → monthly avg",
    ),
    dict(
        fred_id="MABMM301CHM189S",
        label="M1",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland M1 Money Supply, SA (OECD MEI)",
    ),
    dict(
        fred_id="MABMM302CHM189S",
        label="M2",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland M2 Money Supply, SA (OECD MEI)",
    ),
    dict(
        fred_id="MABMM303CHM189S",
        label="M3",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland M3 Money Supply, SA (OECD MEI)",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Trade
    # ------------------------------------------------------------------
    dict(
        fred_id="XTIMVA01CHM667S",
        label="EXPORTS",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Exports of Goods, value SA (OECD MEI)",
    ),
    dict(
        fred_id="XTIMVA02CHM667S",
        label="IMPORTS",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Imports of Goods, value SA (OECD MEI)",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Surveys / soft data / leading indicators
    # ------------------------------------------------------------------
    dict(
        fred_id="CHELORSGPNOSTP",
        label="CLI",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Composite Leading Indicator, amplitude-adjusted (OECD)",
    ),
    dict(
        fred_id="LOCOBSNO02CHM661S",
        label="BUS_CONF",
        freq="M",
        transform="rate",
        sa=True,
        notes="Switzerland Business Confidence: Industrial (OECD)",
    ),
    dict(
        fred_id="CSCICP02CHM460S",
        label="CONS_CONF",
        freq="M",
        transform="rate",
        sa=True,
        notes="Switzerland Consumer Confidence Indicator (OECD)",
    ),
    dict(
        fred_id="LOLITONOSTP",
        label="OECD_CLI",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="OECD Total CLI (global, used as foreign indicator)",
    ),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _apply_transform(series: pd.Series, transform: str, freq: str) -> pd.Series:
    """Apply stationarity transformation to a raw series.

    Parameters
    ----------
    series    : raw level/rate series (monthly freq)
    transform : one of "log_diff3", "log_diff_q", "diff3", "diff_q", "rate"
    freq      : "M" or "Q" (determines diff lag)
    """
    if transform == "rate":
        return series

    if transform == "log_diff3":
        # 3-month log-difference: ln(x_t) - ln(x_{t-3})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_s = np.log(series.clip(lower=1e-12))
        return log_s - log_s.shift(3)

    if transform == "log_diff_q":
        # QoQ log-difference: ln(x_t) - ln(x_{t-1}) on quarterly data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_s = np.log(series.clip(lower=1e-12))
        return log_s - log_s.shift(1)

    if transform == "diff3":
        return series - series.shift(3)

    if transform == "diff_q":
        return series - series.shift(1)

    raise ValueError(f"Unknown transform: {transform!r}")


def _to_monthly_index(series: pd.Series, freq: str) -> pd.Series:
    """Convert any DatetimeIndex series to a monthly (MS) index.

    Quarterly series: kept at quarter-end months (Mar/Jun/Sep/Dec), NaN elsewhere.
    Monthly series: resampled to MS if needed (averages daily series).
    """
    if series.empty:
        return series

    # Resample to monthly mean first (handles daily / irregular series)
    monthly = series.resample("MS").mean()

    if freq == "Q":
        # Keep only quarter-end months, set others to NaN
        mask = monthly.index.month.isin([3, 6, 9, 12])
        monthly = monthly.where(mask)

    return monthly


def fetch_series(fred, entry: dict, start_date: str, end_date: str) -> pd.Series | None:
    """Download one FRED series, resample to monthly, apply transform.

    Returns None if the series cannot be fetched.
    """
    fred_id = entry["fred_id"]
    label = entry["label"]
    freq = entry["freq"]
    transform = entry["transform"]

    try:
        raw = fred.get_series(fred_id, observation_start=start_date, observation_end=end_date)
    except Exception as exc:
        print(f"  [WARN] Could not fetch {fred_id} ({label}): {exc}")
        return None

    if raw is None or raw.empty:
        print(f"  [WARN] Empty series: {fred_id} ({label})")
        return None

    raw = raw.dropna()
    raw.index = pd.DatetimeIndex(raw.index)

    # Resample to monthly and align to quarter-end months for quarterly series
    monthly = _to_monthly_index(raw, freq)

    # Apply stationarity transform
    transformed = _apply_transform(monthly, transform, freq)

    transformed.name = label
    return transformed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    api_key: str,
    start_date: str = "1975-01-01",
    end_date: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download all series and assemble a model-ready DataFrame.

    Returns
    -------
    data          : DataFrame, monthly DatetimeIndex, columns = labels
    quarterly_vars: list of column names that are quarterly
    """
    try:
        from fredapi import Fred
    except ImportError:
        sys.exit(
            "fredapi is not installed. Add it to requirements.txt and rebuild "
            "the container, or run: pip3 install --break-system-packages fredapi"
        )

    fred = Fred(api_key=api_key)

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Fetching {len(SERIES)} series from FRED "
          f"({start_date} → {end_date}) ...")

    frames: list[pd.Series] = []
    quarterly_vars: list[str] = []

    for entry in SERIES:
        print(f"  {entry['fred_id']:30s}  [{entry['freq']}]  {entry['label']}")
        s = fetch_series(fred, entry, start_date, end_date)
        if s is not None:
            frames.append(s)
            if entry["freq"] == "Q":
                quarterly_vars.append(entry["label"])

    if not frames:
        sys.exit("No series were fetched successfully.")

    # Align all series on a common monthly index
    data = pd.concat(frames, axis=1)
    data.index.name = "date"

    # Put GDP first (required by MFDFM)
    if "GDP" in data.columns:
        other_cols = [c for c in data.columns if c != "GDP"]
        data = data[["GDP"] + other_cols]
        if "GDP" not in quarterly_vars:
            quarterly_vars.insert(0, "GDP")
        else:
            quarterly_vars.remove("GDP")
            quarterly_vars.insert(0, "GDP")

    n_obs = data.shape[0]
    n_vars = data.shape[1]
    n_q = len(quarterly_vars)
    n_m = n_vars - n_q
    coverage = data.notna().mean().mean() * 100

    print(f"\nDataset assembled:")
    print(f"  Variables  : {n_vars} ({n_q} quarterly, {n_m} monthly)")
    print(f"  Time span  : {data.index[0].strftime('%Y-%m')} "
          f"→ {data.index[-1].strftime('%Y-%m')}  ({n_obs} months)")
    print(f"  Coverage   : {coverage:.1f}% non-missing")
    print(f"  Quarterly  : {quarterly_vars}")

    return data, quarterly_vars


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--api-key", default=None,
        help="FRED API key (default: $FRED_API_KEY env var)",
    )
    p.add_argument(
        "--start", default="1975-01-01",
        help="Start date YYYY-MM-DD (default: 1975-01-01)",
    )
    p.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--out-csv", default="data/swiss_fred.csv",
        help="Output CSV path (default: data/swiss_fred.csv)",
    )
    p.add_argument(
        "--out-pkl", default="data/swiss_fred.pkl",
        help="Output pickle path (default: data/swiss_fred.pkl)",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Print summary only, do not write files",
    )
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        sys.exit(
            "FRED API key required.\n"
            "Set the FRED_API_KEY environment variable or pass --api-key.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    data, quarterly_vars = build_dataset(
        api_key=api_key,
        start_date=args.start,
        end_date=args.end,
    )

    print("\nFirst few rows (GDP + quarterly vars):")
    q_cols = [c for c in quarterly_vars if c in data.columns]
    print(data[q_cols].dropna(how="all").head(8).to_string())

    if not args.no_save:
        import os as _os
        for path in (args.out_csv, args.out_pkl):
            _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)

        data.to_csv(args.out_csv)
        print(f"\nSaved CSV : {args.out_csv}")

        import pickle
        payload = {
            "data": data,
            "quarterly_vars": quarterly_vars,
            "gdp_var": "GDP",
        }
        with open(args.out_pkl, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved PKL : {args.out_pkl}")

    print("\n--- How to use with MFDFM ---")
    print("import pickle")
    print(f"payload = pickle.load(open('{args.out_pkl}', 'rb'))")
    print("from mfdfm import MFDFM")
    print("model = MFDFM(n_factors='auto', n_lags='auto')")
    print("model.fit(payload['data'],")
    print("          quarterly_vars=payload['quarterly_vars'],")
    print("          gdp_var=payload['gdp_var'])")
    print("bci = model.business_cycle_index")


if __name__ == "__main__":
    main()
