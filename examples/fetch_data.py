"""Fetch Swiss macro data for use with the MFDFM model.

Sources
-------
- FRED (Federal Reserve Bank of St. Louis) — GDP and macro indicators
- SNB  (Swiss National Bank open data API) — CPI (no API key needed)

Usage
-----
    # Fetch CPI from SNB (no API key needed):

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
  all others are percentage-changed (QoQ or 3-month) to remove trends.
- Seasonal adjustment: most OECD MEI series are already SA; raw series are noted.
"""

import argparse
import io
import os
import pickle
import sys
from typing import Literal

import pandas as pd
import requests
from fredapi import Fred


# ---------------------------------------------------------------------------
# SNB CPI fetcher
# ---------------------------------------------------------------------------

SNB_CPI_URL = "https://data.snb.ch/api/cube/plkoprinfla/data/csv/en"


def fetch_snb_cpi(transform: str = "rate") -> pd.Series:
    """Download Swiss CPI from the SNB open data API.

    Returns a monthly Series (MS freq) with the requested transformation
    applied (default: no transform — TLK is already a YoY inflation rate).

    Parameters
    ----------
    transform : "pct_change3" | "rate" | "none"
    """
    resp = requests.get(SNB_CPI_URL, timeout=30)
    resp.raise_for_status()

    # SNB CSV format (2024+): BOM + metadata rows, then a blank line, then
    # a header "Date";"D0";"Value" followed by data rows with quoted fields.
    # Multiple D0 codes appear per date; "TLK" is the national total inflation rate.
    lines = resp.text.lstrip("\ufeff").splitlines()

    # Find header line (contains "Date" and "Value")
    header_idx = next(
        (i for i, l in enumerate(lines) if '"Date"' in l or "Date" in l.split(";")),
        None,
    )
    if header_idx is None:
        raise ValueError("Could not find header row in SNB CPI response.")

    raw_csv = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(raw_csv), sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Filter for the total CPI YoY rate (TLK = national total, year-on-year %)
    if "D0" in df.columns:
        df = df[df["D0"].str.strip().str.strip('"') == "TLK"]

    date_col, val_col = df.columns[0], df.columns[-1]
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = pd.to_datetime(df[date_col].astype(str).str.strip('"').str[:7], format="%Y-%m")
    s = s.sort_index().resample("MS").mean()
    s.name = "CPI_BFS"

    if transform == "pct_change3":
        s = s.pct_change(3)
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
#   transform : "pct_change3" → 3-month percentage change  (monthly levels/indices)
#               "pct_change_q"→ QoQ percentage change       (quarterly levels)
#               "diff3"       → 3-month arithmetic diff     (monthly rates/%-points)
#               "diff_q"      → QoQ arithmetic diff         (quarterly rates)
#               "rate"        → leave as-is (already stationary rate/index)
#   sa        : True if the FRED series is already seasonally adjusted
#   notes     : free-text description
#
# Stationarity transformations (consistent with Galli, 2017, which allows
# either differences or growth rates; we use percentage changes, matching
# official SECO/SNB quarter-on-quarter GDP reporting conventions):
#   - Monthly non-stationary series: x_t / x_{t-3} - 1
#   - Quarterly non-stationary:      x_t / x_{t-1} - 1 (QoQ)
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
        transform="pct_change_q",
        sa=True,
        notes="Real Gross Domestic Product for Switzerland",
    ),
    dict(
        fred_id="NAEXKP02CHQ657S",
        label="GDP_CONS",
        freq="Q",
        transform="pct_change_q",
        sa=True,
        notes="Switzerland Private Final Consumption Expenditure, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP03CHQ657S",
        label="GDP_GOV",
        freq="Q",
        transform="pct_change_q",
        sa=True,
        notes="Switzerland Government Final Consumption Expenditure, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP04CHQ657S",
        label="GDP_INVEST",
        freq="Q",
        transform="pct_change_q",
        sa=True,
        notes="Switzerland Gross Fixed Capital Formation, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP06CHQ657S",
        label="GDP_EXP",
        freq="Q",
        transform="pct_change_q",
        sa=True,
        notes="Switzerland Exports of Goods and Services, real SA (OECD)",
    ),
    dict(
        fred_id="NAEXKP07CHQ657S",
        label="GDP_IMP",
        freq="Q",
        transform="pct_change_q",
        sa=True,
        notes="Switzerland Imports of Goods and Services, real SA (OECD)",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Hard data / production
    # ------------------------------------------------------------------
    dict(
        fred_id="PRMNTO01CHQ657S",
        label="PROMANTOT",
        freq="M",
        transform="pct_change3",
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
        transform="pct_change3",
        sa=True,
        notes="Infra-Annual Registered Unemployment and Job Vacancies: Total Economy: Registered Unemployment for Switzerland",
    ),
    # ------------------------------------------------------------------
    # MONTHLY — Financial / monetary
    # ------------------------------------------------------------------
    dict(
        fred_id="DEXSZUS",
        label="CHFUSD",
        freq="M",
        transform="pct_change3",
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
        transform="pct_change3",
        sa=True,
        notes="International Merchandise Trade Statistics: Imports: Commodities for Switzerland",
    ),
    dict(
        fred_id="XTEXVA01CHM664S",
        label="EXPORTS",
        freq="M",
        transform="pct_change3",
        sa=True,
        notes="International Merchandise Trade Statistics: Exports: Commodities for Switzerland",
    ),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _apply_transform(
    series: pd.Series,
    transform: Literal["pct_change3", "pct_change_q", "diff3", "diff_q", "rate"],
    freq: str,
) -> pd.Series:
    """Apply stationarity transformation to a raw series.

    Parameters
    ----------
    series    : raw level/rate series (monthly freq)
    transform : one of "pct_change3", "pct_change_q", "diff3", "diff_q", "rate"
    freq      : "M" or "Q" (determines diff lag)
    """
    if transform == "rate":
        return series

    if transform == "pct_change3":
        # 3-month percentage change: x_t / x_{t-3} - 1
        return series.pct_change(3)

    if transform == "pct_change_q":
        # QoQ percentage change: x_t / x_{t-1} - 1 on quarterly data
        return series.pct_change(1)

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

    if freq == "Q":
        # FRED dates quarterly series at quarter-start (Jan/Apr/Jul/Oct).
        # resample("QE") normalises any convention to quarter-end dates,
        # then we map e.g. 2020-03-31 → 2020-03-01 (month-start) and
        # reindex onto a full monthly MS grid with NaN in non-quarter months.
        qe = series.resample("QE").mean()
        qe.index = qe.index.to_period("M").to_timestamp()
        start = series.index.min().to_period("M").to_timestamp()
        end = series.index.max().to_period("M").to_timestamp()
        return qe.reindex(pd.date_range(start, end, freq="MS"))
    else:
        return series.resample("MS").mean()


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

    if freq == "Q":
        # Apply QoQ transform at native quarterly frequency first (so shift(1)
        # means one quarter, not one month), then expand to the monthly grid.
        transformed = _apply_transform(raw, transform, freq)
        transformed = _to_monthly_index(transformed, freq)
    else:
        transformed = _to_monthly_index(raw, freq)
        transformed = _apply_transform(transformed, transform, freq)

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

    data["CPI_BFS"] = cpi_snb

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/swiss.csv")
    print("Saved CSV : data/swiss.csv")

    payload = {"data": data, "quarterly_vars": quarterly_vars, "gdp_var": "GDP"}
    with open("data/swiss.pkl", "wb") as f:
        pickle.dump(payload, f)
    print("Saved PKL : data/swiss.pkl")


if __name__ == "__main__":
    main()
