"""
Reusable encoding functions for SynPUF claim features.

Polars Expr variants are used in the training pipeline.
Scalar variants are used by the FastAPI layer to encode single incoming claims.
"""

import polars as pl

# ---------------------------------------------------------------------------
# ICD-9 numeric prefix
# Numeric codes 001-999 → first-3-digit integer
# V-codes (V01-V99)    → sentinel 900
# E-codes (E800-E999)  → sentinel 950
# null / unparseable   → null
# ---------------------------------------------------------------------------

def icd9_prefix_expr(col: str) -> pl.Expr:
    """Map an ICD-9 code column to its 3-digit numeric prefix (Polars Expr)."""
    return (
        pl.when(pl.col(col).is_null())
        .then(None)
        .when(pl.col(col).str.starts_with("V"))
        .then(pl.lit(900, dtype=pl.Int16))
        .when(pl.col(col).str.starts_with("E"))
        .then(pl.lit(950, dtype=pl.Int16))
        .otherwise(
            pl.col(col).str.slice(0, 3).cast(pl.Int16, strict=False)
        )
    )


def icd9_prefix_scalar(code: str | None) -> int | None:
    """Map a single ICD-9 code string to its numeric prefix (for API use)."""
    if not code:
        return None
    code = code.strip().upper()
    if code.startswith("V"):
        return 900
    if code.startswith("E"):
        return 950
    try:
        return int(code[:3])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SynPUF chronic-condition flag encoding
# SynPUF encodes: 1 = Yes (has condition), 2 = No — convert to 0/1 int8
# ---------------------------------------------------------------------------

def chronic_flag_expr(col: str) -> pl.Expr:
    """Convert SynPUF 1/2 chronic-condition coding to 0/1 Int8."""
    return (pl.col(col) == 1).cast(pl.Int8)


# Mapping from SynPUF SP_* column names to clean feature names
CHRONIC_COL_MAP: dict[str, str] = {
    "SP_ALZHDMTA": "has_alzheimer",
    "SP_CHF":      "has_chf",
    "SP_CHRNKIDN": "has_ckd",
    "SP_CNCR":     "has_cancer",
    "SP_COPD":     "has_copd",
    "SP_DEPRESSN": "has_depression",
    "SP_DIABETES": "has_diabetes",
    "SP_ISCHMCHT": "has_ihd",
    "SP_OSTEOPRS": "has_osteoporosis",
    "SP_RA_OA":    "has_ra_oa",
    "SP_STRKETIA": "has_stroke",
}

CHRONIC_FEATURE_NAMES: list[str] = list(CHRONIC_COL_MAP.values())
