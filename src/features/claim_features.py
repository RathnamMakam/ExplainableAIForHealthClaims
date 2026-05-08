"""
Feature engineering + denial label construction.

Input:  data/processed/claims_raw.parquet
        data/processed/meps_denial_rates.parquet
Output: data/processed/claims_features.parquet

Denial label design
-------------------
SynPUF has no ground-truth denial flag, so we construct one from claim
characteristics known to drive real-world denials, calibrated to the
MEPS-derived base rate (~15%).  The label uses a log-odds model:

  log_odds = intercept
             + 0.80 * prior_auth_missing   ← KEY demo driver (SHAP waterfall)
             + 1.20 * zero_or_neg_payment  ← payment not made
             + 0.30 * vcode_primary        ← supplementary/admin code as primary
             + 0.50 * zero_los_inpatient   ← IP claim with 0 LOS (data quality)
             + 0.20 * high_cost            ← high-cost scrutiny
             + 0.15 * chronic_burden       ← complexity proxy
             - 0.10 * coverage_stability   ← stable coverage reduces denials
             - 0.20 * has_hmo              ← HMO typically pre-auths upfront

The intercept is tuned so sigmoid(intercept) ≈ MEPS base denial rate for the
demographic mix in the sample.

Usage:
  python src/features/claim_features.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.encoders import (
    CHRONIC_COL_MAP,
    CHRONIC_FEATURE_NAMES,
    chronic_flag_expr,
    icd9_prefix_expr,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature engineering — Polars pipeline
# ---------------------------------------------------------------------------

def _temporal(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        ((pl.col("CLM_THRU_DT") - pl.col("CLM_FROM_DT")).dt.total_days()
         .cast(pl.Int16))
        .alias("los_days"),

        pl.col("CLM_FROM_DT").dt.year().cast(pl.Int16).alias("claim_year"),
        pl.col("CLM_FROM_DT").dt.month().cast(pl.Int8).alias("claim_month"),
    ])


def _amounts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # Clip negatives: negative amounts are claim adjustments (reversal)
        pl.col("CLM_PMT_AMT").fill_null(0.0).alias("clm_pmt_amt"),
        pl.col("NCH_PRMRY_PYR_CLM_PD_AMT").fill_null(0.0).alias("primary_payer_amt"),

        # Adjustment flag: original column before clipping
        (pl.col("CLM_PMT_AMT").fill_null(0.0) < 0).cast(pl.Int8).alias("is_adjustment"),
    ]).with_columns([
        (pl.col("primary_payer_amt") > 0).cast(pl.Int8).alias("has_primary_payer"),
    ])


def _provider(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # Prior auth proxy: attending physician NPI present → auth likely on file
        pl.col("AT_PHYSN_NPI").is_not_null().cast(pl.Int8).alias("prior_auth_present"),
        pl.col("OP_PHYSN_NPI").is_not_null().cast(pl.Int8).alias("has_op_surgeon"),
    ])


def _diagnosis(df: pl.DataFrame) -> pl.DataFrame:
    # Primary diagnosis as 3-digit numeric prefix (V=900, E=950)
    primary_col = pl.when(
        pl.col("ADMTNG_ICD9_DGNS_CD").is_not_null()
    ).then(
        icd9_prefix_expr("ADMTNG_ICD9_DGNS_CD")
    ).otherwise(
        icd9_prefix_expr("ICD9_DGNS_CD_1")
    ).alias("primary_icd9_num")

    # Count filled diagnosis and procedure fields
    dx_cols  = ["ICD9_DGNS_CD_1", "ICD9_DGNS_CD_2", "ICD9_DGNS_CD_3"]
    prc_cols = ["ICD9_PRCDR_CD_1", "ICD9_PRCDR_CD_2"]

    num_dx  = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in dx_cols if c in df.columns])
    num_prc = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in prc_cols if c in df.columns])

    return df.with_columns([
        primary_col,
        num_dx.cast(pl.Int8).alias("num_diagnoses"),
        num_prc.cast(pl.Int8).alias("num_procedures"),
        pl.col("ICD9_PRCDR_CD_1").is_not_null().cast(pl.Int8).alias("has_procedure"),
        pl.col("HCPCS_CD_1").is_not_null().cast(pl.Int8).alias("has_hcpcs")
            if "HCPCS_CD_1" in df.columns
            else pl.lit(0, dtype=pl.Int8).alias("has_hcpcs"),
        pl.col("CLM_DRG_CD").cast(pl.Int16, strict=False).alias("drg_num")
            if "CLM_DRG_CD" in df.columns
            else pl.lit(None, dtype=pl.Int16).alias("drg_num"),
        pl.col("CLM_UTLZTN_DAY_CNT").cast(pl.Int16, strict=False).alias("utlztn_days")
            if "CLM_UTLZTN_DAY_CNT" in df.columns
            else pl.lit(None, dtype=pl.Int16).alias("utlztn_days"),
    ])


def _bene_demographics(df: pl.DataFrame) -> pl.DataFrame:
    age_days = (pl.col("CLM_FROM_DT") - pl.col("BENE_BIRTH_DT")).dt.total_days()
    bene_age = (age_days / 365.25).cast(pl.Int16)

    return df.with_columns([
        bene_age.alias("bene_age"),
        pl.col("BENE_SEX_IDENT_CD").cast(pl.Int8).alias("bene_sex"),
        pl.col("BENE_RACE_CD").cast(pl.Int8).alias("bene_race"),
        pl.col("BENE_ESRD_IND").cast(pl.Int8).alias("bene_esrd"),
        pl.col("BENE_HI_CVRAGE_TOT_MONS").cast(pl.Int8, strict=False).alias("bene_hi_mons"),
        pl.col("BENE_HMO_CVRAGE_TOT_MONS").cast(pl.Int8, strict=False).alias("bene_hmo_mons"),
        pl.col("BENE_DEATH_DT").is_not_null().cast(pl.Int8).alias("bene_is_deceased"),
    ])


def _chronic_conditions(df: pl.DataFrame) -> pl.DataFrame:
    chronic_exprs = [
        chronic_flag_expr(src).alias(dst)
        for src, dst in CHRONIC_COL_MAP.items()
        if src in df.columns
    ]
    df = df.with_columns(chronic_exprs)

    present = [f for f in CHRONIC_FEATURE_NAMES if f in df.columns]
    chronic_count = pl.sum_horizontal([pl.col(f) for f in present]).cast(pl.Int8)
    return df.with_columns(chronic_count.alias("chronic_count"))


def _claim_type_flag(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("claim_type") == "IP").cast(pl.Int8).alias("is_inpatient")
    )


# ---------------------------------------------------------------------------
# Final feature selection — defines training column order
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # claim
    "is_inpatient", "los_days", "claim_year", "claim_month",
    # amounts
    "clm_pmt_amt", "primary_payer_amt", "has_primary_payer", "is_adjustment",
    # provider / auth
    "prior_auth_present", "has_op_surgeon",
    # diagnosis
    "primary_icd9_num", "num_diagnoses", "num_procedures",
    "has_procedure", "has_hcpcs", "drg_num", "utlztn_days",
    # bene demographics
    "bene_age", "bene_sex", "bene_race", "bene_esrd",
    "bene_hi_mons", "bene_hmo_mons", "bene_is_deceased",
    # chronic conditions
    *CHRONIC_FEATURE_NAMES,
    "chronic_count",
]

ID_COLS = ["claim_id", "bene_id", "claim_type", "sample_num"]


def build_features(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw
        .pipe(_temporal)
        .pipe(_amounts)
        .pipe(_provider)
        .pipe(_diagnosis)
        .pipe(_bene_demographics)
        .pipe(_chronic_conditions)
        .pipe(_claim_type_flag)
    )


# ---------------------------------------------------------------------------
# Denial label construction
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def build_denial_label(df: pl.DataFrame, meps_rates: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    """
    Compute a synthetic denial probability from claim features and sample
    a binary label.  The intercept is calibrated to the MEPS overall base
    denial rate for the mixed-age, mixed-insurance population.
    """
    # MEPS base rate (weighted average across all demographic segments)
    base_rate = meps_rates["denial_rate"].mean()
    # Convert base rate to log-odds intercept: logit(p) = log(p/(1-p))
    intercept = float(np.log(base_rate / (1.0 - base_rate)))

    # Pull numpy arrays (fill nulls so arithmetic is clean)
    prior_auth_missing  = (1 - df["prior_auth_present"].fill_null(0)).to_numpy().astype(np.float32)
    zero_or_neg_payment = (df["clm_pmt_amt"].fill_null(0) <= 0).to_numpy().astype(np.float32)
    vcode_primary       = (df["primary_icd9_num"].fill_null(0) >= 900).to_numpy().astype(np.float32)
    zero_los_ip         = (
        (df["is_inpatient"].fill_null(0).to_numpy() == 1) &
        (df["los_days"].fill_null(1).to_numpy() <= 0)
    ).astype(np.float32)
    high_cost           = (df["clm_pmt_amt"].fill_null(0) > 10_000).to_numpy().astype(np.float32)
    chronic_burden      = df["chronic_count"].fill_null(0).to_numpy().astype(np.float32) / 10.0
    coverage_stability  = df["bene_hi_mons"].fill_null(0).to_numpy().astype(np.float32) / 12.0
    has_hmo             = (df["bene_hmo_mons"].fill_null(0) > 0).to_numpy().astype(np.float32)

    log_odds = (
        intercept
        + 0.80 * prior_auth_missing
        + 1.20 * zero_or_neg_payment
        + 0.30 * vcode_primary
        + 0.50 * zero_los_ip
        + 0.20 * high_cost
        + 0.15 * chronic_burden
        - 0.10 * coverage_stability
        - 0.20 * has_hmo
    )

    denial_prob  = _sigmoid(log_odds).astype(np.float32)
    rng          = np.random.default_rng(seed)
    denial_label = rng.binomial(1, denial_prob).astype(np.int8)

    return df.with_columns([
        pl.Series("denial_prob",  denial_prob),
        pl.Series("denial_label", denial_label),
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg     = _cfg()
    proc    = ROOT / cfg["paths"]["data_processed"]
    raw_in  = proc / "claims_raw.parquet"
    rates_in = proc / "meps_denial_rates.parquet"
    out     = proc / "claims_features.parquet"

    print(f"Loading {raw_in.name} ...", end=" ", flush=True)
    raw = pl.read_parquet(raw_in)
    print(f"{len(raw):,} rows")

    print("Loading MEPS denial rates ...", end=" ", flush=True)
    meps_rates = pl.read_parquet(rates_in)
    base_rate  = float(meps_rates["denial_rate"].mean())
    print(f"base rate = {base_rate:.1%}")

    print("Engineering features ...")
    features = build_features(raw)

    print("Generating denial labels ...")
    features = build_denial_label(features, meps_rates)

    # Select final columns — keep only what exists
    all_cols   = ID_COLS + FEATURE_COLS + ["denial_prob", "denial_label"]
    final_cols = [c for c in all_cols if c in features.columns]
    output     = features.select(final_cols)

    output.write_parquet(out, compression="snappy")

    # Summary
    denial_rate = output["denial_label"].mean()
    print(f"\nWrote {len(output):,} rows x {len(output.columns)} cols to {out.name}")
    print(f"Overall denial rate : {denial_rate:.1%}  (target ~{base_rate:.1%})")
    print(f"Denied claims       : {output['denial_label'].sum():,}")
    print(f"Approved claims     : {(output['denial_label'] == 0).sum():,}")
    print()

    # Key feature breakdown
    pa = output.filter(pl.col("prior_auth_present") == 0)
    print(f"prior_auth_present=0 count : {len(pa):,}  "
          f"denial rate: {pa['denial_label'].mean():.1%}")
    print(f"prior_auth_present=1 count : {len(output) - len(pa):,}  "
          f"denial rate: {output.filter(pl.col('prior_auth_present')==1)['denial_label'].mean():.1%}")
    print()
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")


if __name__ == "__main__":
    main()
