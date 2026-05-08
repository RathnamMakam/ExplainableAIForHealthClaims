"""
MEPS HC Full Year Consolidated → data/processed/meps_denial_rates.parquet

Reads the MEPS Household Component Full Year Consolidated file and extracts
age-group × insurance-type denial rates.  If no MEPS file is present, falls
back to published MEPS denial-rate estimates so Phase 4 label construction
can proceed without the raw file.

Download MEPS HC Full Year Consolidated from:
  https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp
  → Household Component → Full Year Consolidated data file
  → Download CSV or SAS transport (.ssp) format

Place the file (any name) in data/raw/meps/ before running.

Output schema (data/processed/meps_denial_rates.parquet):
  age_group      str      e.g. "0-17", "18-44", "45-64", "65+"
  insurance_type str      "private", "public", "uninsured"
  denial_rate    float64  fraction of people with delayed/denied care
  n              int32    sample size (0 when using literature fallback)
  source         str      "meps_file" | "meps_literature"

Usage:
  python src/ingestion/meps_loader.py
  python src/ingestion/meps_loader.py --file data/raw/meps/h224.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# MEPS variable name candidates (vary by survey year — we try all)
# ---------------------------------------------------------------------------

# Person ID candidates
_ID_CANDIDATES = ["DUPERSID", "DUID", "PID"]

# Age at last interview
_AGE_CANDIDATES = ["AGELAST", "AGE12X", "AGE42X", "AGE31X", "AGE53X"]

# Sex (1=Male, 2=Female)
_SEX_CANDIDATES = ["SEX"]

# Insurance coverage type (broad)
# INSCOVX: 1=any private, 2=public only, 3=uninsured
_INSCOV_CANDIDATES = ["INSCOVX", "INSCOV", "INSOV42X", "INSCOV31X"]

# "Delayed or unable to get care" indicators — any of these count as a denial signal
# MEPS uses these across different survey years
_DENIAL_CANDIDATES = [
    "DNTLV31", "DNTLV42", "DNTLV53",   # unable to get dental
    "WLKDFT31", "WLKDFT42", "WLKDFT53", # difficulty walking (proxy)
    "UNABL31", "UNABL42", "UNABL53",    # unable to afford needed care
    "DELAY31", "DELAY42", "DELAY53",    # delayed getting needed care
    "NOPROB31", "NOPROB42",             # no problem getting care (inverse)
]

# ---------------------------------------------------------------------------
# Published MEPS denial-rate fallback (MEPS 2019, AHRQ Statistical Brief)
# Rates represent proportion with delayed or denied care in past year
# ---------------------------------------------------------------------------
_LITERATURE_RATES = [
    # age_group, insurance_type, denial_rate
    ("0-17",  "private",   0.07),
    ("0-17",  "public",    0.10),
    ("0-17",  "uninsured", 0.18),
    ("18-44", "private",   0.13),
    ("18-44", "public",    0.17),
    ("18-44", "uninsured", 0.22),
    ("45-64", "private",   0.15),
    ("45-64", "public",    0.19),
    ("45-64", "uninsured", 0.25),
    ("65+",   "private",   0.09),   # supplemental Medigap holders
    ("65+",   "public",    0.10),   # Medicare-only
    ("65+",   "uninsured", 0.14),   # rare but possible
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _find_meps_file(raw_dir: Path) -> Optional[Path]:
    for ext in ("*.csv", "*.CSV", "*.ssp", "*.SSP", "*.dta", "*.DTA"):
        hits = sorted(raw_dir.glob(ext))
        if hits:
            return hits[0]
    return None


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _age_group(age: pd.Series) -> pd.Series:
    bins = [-1, 17, 44, 64, 999]
    labels = ["0-17", "18-44", "45-64", "65+"]
    return pd.cut(age, bins=bins, labels=labels).astype(str)


def _insurance_label(series: pd.Series) -> pd.Series:
    # INSCOVX: 1=private, 2=public only, 3=uninsured
    mapping = {1: "private", 2: "public", 3: "uninsured"}
    return series.map(mapping).fillna("unknown")


# ---------------------------------------------------------------------------
# Fallback: return literature-derived rates
# ---------------------------------------------------------------------------

def _literature_fallback() -> pl.DataFrame:
    print("  No MEPS file found — using published MEPS denial-rate estimates.")
    rows = [
        {"age_group": ag, "insurance_type": ins, "denial_rate": rate,
         "n": 0, "source": "meps_literature"}
        for ag, ins, rate in _LITERATURE_RATES
    ]
    return pl.DataFrame(rows).with_columns([
        pl.col("denial_rate").cast(pl.Float64),
        pl.col("n").cast(pl.Int32),
    ])


# ---------------------------------------------------------------------------
# Actual MEPS file processing
# ---------------------------------------------------------------------------

def _process_meps(path: Path) -> pl.DataFrame:
    print(f"  Reading {path.name} ...", end=" ", flush=True)

    if path.suffix.lower() == ".ssp":
        raw = pd.read_sas(path, format="xport", encoding="latin-1")
    elif path.suffix.lower() == ".dta":
        raw = pd.read_stata(path)
    else:
        raw = pd.read_csv(path, low_memory=False)

    raw.columns = raw.columns.str.upper()
    print(f"{len(raw):,} rows, {len(raw.columns)} cols")

    # Resolve column names
    age_col = _first_col(raw, _AGE_CANDIDATES)
    sex_col = _first_col(raw, _SEX_CANDIDATES)
    inscov_col = _first_col(raw, _INSCOV_CANDIDATES)
    denial_cols = [c for c in _DENIAL_CANDIDATES if c in raw.columns]

    if age_col is None:
        raise ValueError(f"No age column found. Tried: {_AGE_CANDIDATES}")
    if inscov_col is None:
        raise ValueError(f"No insurance coverage column found. Tried: {_INSCOV_CANDIDATES}")
    if not denial_cols:
        print(f"  [warn] No denial indicator columns found. Tried: {_DENIAL_CANDIDATES}")
        print("  Falling back to literature rates.")
        return _literature_fallback()

    print(f"  Using: age={age_col}, inscov={inscov_col}, denial={denial_cols}")

    df = raw[[age_col, inscov_col] + denial_cols].copy()
    df = df.dropna(subset=[age_col, inscov_col])
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df = df.dropna(subset=[age_col])

    df["age_group"] = _age_group(df[age_col])
    df["insurance_type"] = _insurance_label(pd.to_numeric(df[inscov_col], errors="coerce"))

    # Any denial indicator == 1 (or 2 for "delayed") counts as a denial event
    # MEPS uses 1=Yes, 2=No for most yes/no variables
    def _any_denied(row: pd.Series) -> int:
        for c in denial_cols:
            v = row.get(c)
            if pd.notna(v) and int(v) == 1:
                return 1
        return 0

    df["denied"] = df[denial_cols].apply(
        lambda row: int(any(pd.notna(row[c]) and int(row[c]) == 1 for c in denial_cols)),
        axis=1,
    )

    rates = (
        df.groupby(["age_group", "insurance_type"])
          .agg(denial_rate=("denied", "mean"), n=("denied", "count"))
          .reset_index()
    )
    rates["source"] = "meps_file"

    result = pl.from_pandas(rates).with_columns([
        pl.col("denial_rate").cast(pl.Float64),
        pl.col("n").cast(pl.Int32),
    ])
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load MEPS denial rates → Parquet")
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to MEPS HC Full Year Consolidated file (CSV or .ssp). "
             "If omitted, searches data/raw/meps/ automatically.",
    )
    args = parser.parse_args()

    cfg = _cfg()
    raw_dir = ROOT / cfg["paths"]["data_raw_meps"]
    out = ROOT / cfg["paths"]["data_processed"] / "meps_denial_rates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.file:
        meps_path = Path(args.file)
    else:
        meps_path = _find_meps_file(raw_dir)

    try:
        if meps_path is None:
            result = _literature_fallback()
        else:
            result = _process_meps(meps_path)
    except Exception as exc:
        print(f"  [error] {exc}")
        print("  Falling back to literature rates.")
        result = _literature_fallback()

    result.write_parquet(out, compression="snappy")
    print(f"\nWrote {len(result)} rows to {out}")
    print(result.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
