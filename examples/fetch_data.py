"""Fetch Swiss macro data for use with the MFDFM model.

Sources
-------
- FRED (Federal Reserve Bank of St. Louis) — GDP and macro indicators
- SNB  (Swiss National Bank open data API) — CPI (no API key needed)

Usage
-----
    # Fetch CPI from SNB (no API key needed):
    python3 examples/fetch_data.py --snb-cpi

    # Fetch FRED series (API key required):
    export FRED_API_KEY="your_key_here"
    python3 examples/fetch_data.py --api-key YOUR_KEY

    # Change output paths:
    python3 examples/fetch_data.py --out-csv data/swiss.csv --out-pkl data/swiss.pkl

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

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SNB CPI fetcher
# ---------------------------------------------------------------------------

SNB_CPI_URL = "https://data.snb.ch/api/cube/plkoprinfla/data/csv/en"


def fetch_snb_cpi(transform: str = "log_diff3") -> pd.Series:
    """Download Swiss CPI from the SNB open data API.

    Returns a monthly Series (MS freq) with the requested transformation
    applied (default: 3-month log-difference, as required by the MFDFM).

    Parameters
    ----------
    transform : "log_diff3" | "rate" | "none"
    """
    import requests
    resp = requests.get(SNB_CPI_URL, timeout=30)
    resp.raise_for_status()

    # SNB CSVs have a variable number of metadata/comment rows before the
    # actual data.  The data block starts at the first line whose first field
    # looks like a date (YYYY-MM or YYYY).
    lines = resp.text.splitlines()
    data_lines = [l for l in lines if l and l[0].isdigit()]
    if not data_lines:
        raise ValueError("Could not find data rows in SNB CPI response.")

    # Find the header row (last non-data line before the data block)
    data_start = next(i for i, l in enumerate(lines) if l and l[0].isdigit())
    # SNB uses semicolons; the header is the line just before the data
    header_line = lines[data_start - 1] if data_start > 0 else None

    raw_csv = "\n".join(
        ([header_line] if header_line else []) + data_lines
    )
    df = pd.read_csv(io.StringIO(raw_csv), sep=";")

    # First column is the date, second is the value (total CPI index)
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = pd.to_datetime(df[date_col].astype(str).str[:7], format="%Y-%m")
    s = s.sort_index().resample("MS").mean()
    s.name = "CPI_SNB"

    if transform == "log_diff3":
        log_s = np.log(s.clip(lower=1e-12))
        s = log_s - log_s.shift(3)
    elif transform == "none":
        pass
    # "rate" → no transform

    print(f"  SNB CPI: {s.first_valid_index().strftime('%Y-%m')} → "
          f"{s.last_valid_index().strftime('%Y-%m')}  "
          f"({s.notna().sum()} observations)")
    return s


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
        fred_id="CLVMNACSAB1GQCH",
        label="GDP",
        freq="Q",
        transform="log_diff_q",
        sa=True,
        notes="Real Gross Domestic Product for Switzerland",
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
        fred_id="CHEPROINDAISMEI",
        label="PROIND",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Switzerland Production in Industry, SA (OECD MEI)",
    ),
    dict(
        fred_id="PRMNTO01CHQ657S",
        label="PROMANTOT",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Production: Manufacturing: Total Manufacturing for Switzerland",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Labor market
    # ------------------------------------------------------------------
    dict(
        fred_id="SLUEM1524ZSCHE",
        label="UNEMP",
        freq="M",
        transform="rate",
        sa=False,
        notes="Switzerland Unemployment Rate, SA (OECD)",
    ),
    dict(
        fred_id="LMUNRLTTCHM647S",
        label="UNEMPL",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="Infra-Annual Registered Unemployment and Job Vacancies: Total Economy: Registered Unemployment for Switzerland",
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
        notes="Consumer Price Indices (CPIs, HICPs), COICOP 1999: Consumer Price Index: Total for Switzerland ",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Financial / monetary
    # ------------------------------------------------------------------
    dict(
        fred_id="DEXSZUS",
        label="CHFUSD",
        freq="M",
        transform="log_diff3",
        sa=False,
        notes="Switzerland / U.S. Foreign Exchange Rate (CHF per USD), daily → monthly avg",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Trade
    # ------------------------------------------------------------------
    dict(
        fred_id="XTIMVA01CHM667S",
        label="IMPORTS",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="International Merchandise Trade Statistics: Imports: Commodities for Switzerland",
    ),
    dict(
        fred_id="XTEXVA01CHM664S",
        label="EXPORTS",
        freq="M",
        transform="log_diff3",
        sa=True,
        notes="International Merchandise Trade Statistics: Exports: Commodities for Switzerland",
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

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=None, help="FRED API key (default: $FRED_API_KEY)")
    args, _ = p.parse_known_args()

    api_key = args.api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        sys.exit(
            "FRED API key required.\n"
            "Set the FRED_API_KEY environment variable or pass --api-key.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    print("Fetching CPI from SNB ...")
    cpi_snb = fetch_snb_cpi()

    data, quarterly_vars = build_dataset(api_key=api_key)

    # Use SNB CPI instead of FRED CPI
    if "CPI" in data.columns:
        data = data.drop(columns=["CPI"])
    data["CPI_SNB"] = cpi_snb

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/swiss.csv")
    print("Saved CSV : data/swiss.csv")

    import pickle
    payload = {"data": data, "quarterly_vars": quarterly_vars, "gdp_var": "GDP"}
    with open("data/swiss.pkl", "wb") as f:
        pickle.dump(payload, f)
    print("Saved PKL : data/swiss.pkl")


if __name__ == "__main__":
    main()
