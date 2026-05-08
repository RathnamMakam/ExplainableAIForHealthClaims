"""
SynPUF inpatient + outpatient claims → data/processed/claims_raw.parquet

Place SynPUF ZIP or CSV files in data/raw/synpuf/ before running.
Download from:
  https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF

Expected filenames per sample N (ZIP or unzipped CSV both accepted):
  DE1_0_2008_Beneficiary_Summary_File_Sample_{N}.{csv|zip}
  DE1_0_2008_to_2010_Inpatient_Claims_Sample_{N}.{csv|zip}
  DE1_0_2008_to_2010_Outpatient_Claims_Sample_{N}.{csv|zip}

Usage:
  python src/ingestion/synpuf_loader.py --samples 1 2 3
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Optional

import polars as pl
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Column subsets — missing columns in a file are skipped with a warning
# ---------------------------------------------------------------------------
_BENE_WANT = [
    "DESYNPUF_ID", "BENE_BIRTH_DT", "BENE_DEATH_DT",
    "BENE_SEX_IDENT_CD", "BENE_RACE_CD", "BENE_ESRD_IND",
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA",
    "BENE_HI_CVRAGE_TOT_MONS", "BENE_SMI_CVRAGE_TOT_MONS", "BENE_HMO_CVRAGE_TOT_MONS",
]

_IP_WANT = [
    "DESYNPUF_ID", "CLM_ID", "SEGMENT",
    "CLM_FROM_DT", "CLM_THRU_DT", "CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT",
    "PRVDR_NUM", "AT_PHYSN_NPI", "OP_PHYSN_NPI",
    "CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT",
    "CLM_UTLZTN_DAY_CNT", "CLM_DRG_CD",
    "ADMTNG_ICD9_DGNS_CD",
    "ICD9_DGNS_CD_1", "ICD9_DGNS_CD_2", "ICD9_DGNS_CD_3",
    "ICD9_PRCDR_CD_1", "ICD9_PRCDR_CD_2",
    "NCH_BENE_IP_DDCTBL_AMT", "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
]

_OP_WANT = [
    "DESYNPUF_ID", "CLM_ID",
    "CLM_FROM_DT", "CLM_THRU_DT",
    "PRVDR_NUM", "AT_PHYSN_NPI", "OP_PHYSN_NPI",
    "CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT",
    "ICD9_DGNS_CD_1", "ICD9_DGNS_CD_2", "ICD9_DGNS_CD_3",
    "ICD9_PRCDR_CD_1", "ICD9_PRCDR_CD_2",
    "HCPCS_CD_1", "HCPCS_CD_2",
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM", "NCH_BENE_PTB_DDCTBL_AMT",
]

_AMT_COLS = [
    "CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT",
    "NCH_BENE_IP_DDCTBL_AMT", "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
    "NCH_BENE_PTB_DDCTBL_AMT",
]

# ICD-9 V-codes and E-codes are alphanumeric — force all code columns to Utf8
# so Polars doesn't infer i64 from early rows and crash on V090, V5869, etc.
# Covers the full SynPUF column ranges: 10 dx, 6 px, 45 HCPCS.
_FORCE_STR: dict[str, type[pl.DataType]] = {c: pl.Utf8 for c in (
    ["DESYNPUF_ID", "CLM_ID", "PRVDR_NUM", "AT_PHYSN_NPI", "OP_PHYSN_NPI",
     "CLM_DRG_CD", "ADMTNG_ICD9_DGNS_CD", "BENE_ESRD_IND"]
    + [f"ICD9_DGNS_CD_{i}"  for i in range(1, 11)]
    + [f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)]
    + [f"HCPCS_CD_{i}"      for i in range(1, 46)]
)}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _find(raw_dir: Path, *patterns: str) -> Optional[Path]:
    """Return first file matching any glob pattern, or None."""
    for pat in patterns:
        hits = sorted(raw_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _read_csv(path: Path, want: list[str]) -> pl.DataFrame:
    """Read a CSV (or the first CSV inside a ZIP), keeping only wanted columns."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_name = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv")), None
            )
            if csv_name is None:
                raise FileNotFoundError(f"No CSV found inside {path.name}")
            data: io.BytesIO | Path = io.BytesIO(zf.read(csv_name))
    else:
        data = path

    df = pl.read_csv(data, infer_schema_length=5000, null_values=["", " ", "NA"], schema_overrides=_FORCE_STR)
    present = set(df.columns)
    missing = [c for c in want if c not in present]
    if missing:
        print(f"    [warn] {path.name}: columns not found — {missing}")
    return df.select([c for c in want if c in present])


def _date_expr(col: str) -> pl.Expr:
    """YYYYMMDD int/string → Date, nullifying zero-valued entries."""
    cleaned = (
        pl.col(col).cast(pl.Utf8)
        .str.strip_chars()
        .str.replace(r"^0+$", "")          # "0", "00000000" → ""
    )
    return cleaned.str.to_date(format="%Y%m%d", strict=False).alias(col)


def _npi_expr(col: str) -> pl.Expr:
    """Normalize NPI: strip whitespace, remove trailing .0, null out zeros."""
    cleaned = (
        pl.col(col).cast(pl.Utf8)
        .str.strip_chars()
        .str.replace(r"\.0$", "")
    )
    return (
        pl.when(cleaned.str.len_chars() == 0)
        .then(None)
        .when(cleaned == "0")
        .then(None)
        .otherwise(cleaned)
        .alias(col)
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_bene(raw_dir: Path, n: int) -> pl.DataFrame:
    path = _find(
        raw_dir,
        f"*Beneficiary_Summary*Sample_{n}.csv",
        f"*Beneficiary_Summary*Sample_{n}.zip",
        f"*Beneficiary*{n}.csv",
        f"*Beneficiary*{n}.zip",
        f"*beneficiary*{n}*.csv",
        f"*beneficiary*{n}*.zip",
    )
    if path is None:
        raise FileNotFoundError(
            f"Beneficiary summary for sample {n} not found in {raw_dir}\n"
            "  Download SynPUF files from cms.gov > Research > Downloadable "
            "Public Use Files > SynPUFs > DE_Syn_PUF"
        )
    df = _read_csv(path, _BENE_WANT)

    exprs = []
    for col in ("BENE_BIRTH_DT", "BENE_DEATH_DT"):
        if col in df.columns:
            exprs.append(_date_expr(col))
    if "BENE_ESRD_IND" in df.columns:
        exprs.append(
            (pl.col("BENE_ESRD_IND").cast(pl.Utf8).str.to_uppercase() == "Y")
            .alias("BENE_ESRD_IND")
        )
    if exprs:
        df = df.with_columns(exprs)

    return df.rename({"DESYNPUF_ID": "bene_id"})


def _load_claims(raw_dir: Path, n: int, kind: str) -> pl.DataFrame:
    assert kind in ("IP", "OP")
    label = "Inpatient" if kind == "IP" else "Outpatient"
    want = _IP_WANT if kind == "IP" else _OP_WANT

    path = _find(
        raw_dir,
        f"*{label}*Sample_{n}.csv",
        f"*{label}*Sample_{n}.zip",
        f"*{label}*{n}.csv",
        f"*{label}*{n}.zip",
        f"*{label.lower()}*{n}*.csv",
        f"*{label.lower()}*{n}*.zip",
    )
    if path is None:
        raise FileNotFoundError(f"{label} claims for sample {n} not found in {raw_dir}")

    df = _read_csv(path, want)

    # IP claims can have multiple segments per claim — keep segment 1 only
    if "SEGMENT" in df.columns:
        df = df.filter(pl.col("SEGMENT").cast(pl.Utf8).str.strip_chars() == "1").drop("SEGMENT")

    exprs = []
    for col in ("CLM_FROM_DT", "CLM_THRU_DT", "CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT"):
        if col in df.columns:
            exprs.append(_date_expr(col))
    for col in ("AT_PHYSN_NPI", "OP_PHYSN_NPI"):
        if col in df.columns:
            exprs.append(_npi_expr(col))
    for col in _AMT_COLS:
        if col in df.columns:
            exprs.append(pl.col(col).cast(pl.Float32, strict=False).alias(col))

    if exprs:
        df = df.with_columns(exprs)

    return (
        df.rename({"DESYNPUF_ID": "bene_id", "CLM_ID": "claim_id"})
          .with_columns(pl.lit(kind).alias("claim_type"))
    )


# ---------------------------------------------------------------------------
# Per-sample orchestration
# ---------------------------------------------------------------------------

def process_sample(raw_dir: Path, n: int) -> pl.DataFrame:
    print(f"  bene ...", end=" ", flush=True)
    bene = _load_bene(raw_dir, n)
    print(f"{len(bene):,} beneficiaries")

    frames = []
    for kind in ("IP", "OP"):
        print(f"  {kind}  ...", end=" ", flush=True)
        try:
            claims = _load_claims(raw_dir, n, kind)
            merged = (
                claims
                .join(bene, on="bene_id", how="left")
                .with_columns(pl.lit(n).cast(pl.Int8).alias("sample_num"))
            )
            frames.append(merged)
            print(f"{len(claims):,} claims")
        except FileNotFoundError as exc:
            print(f"\n  [skip] {exc}")

    if not frames:
        raise RuntimeError(f"No claim files found for sample {n}")
    return pl.concat(frames, how="diagonal")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load CMS SynPUF claims → Parquet")
    parser.add_argument(
        "--samples", nargs="+", type=int, required=True,
        help="Sample numbers to load, e.g. --samples 1 2 3",
    )
    args = parser.parse_args()

    cfg = _cfg()
    raw_dir = ROOT / cfg["paths"]["data_raw_synpuf"]
    out = ROOT / cfg["paths"]["data_processed"] / "claims_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    all_frames: list[pl.DataFrame] = []
    for n in tqdm(args.samples, desc="Samples"):
        print(f"\nSample {n}:")
        try:
            all_frames.append(process_sample(raw_dir, n))
        except Exception as exc:
            print(f"  [error] sample {n}: {exc}")

    if not all_frames:
        print(f"\nNo data loaded. Place SynPUF files in: {raw_dir}")
        sys.exit(1)

    combined = pl.concat(all_frames, how="diagonal")
    combined.write_parquet(out, compression="snappy")

    print(f"\nWrote {len(combined):,} rows to {out}")
    print(f"claim_type counts:\n{combined['claim_type'].value_counts(sort=True)}")
    print(f"Columns ({len(combined.columns)}): {combined.columns}")


if __name__ == "__main__":
    main()
