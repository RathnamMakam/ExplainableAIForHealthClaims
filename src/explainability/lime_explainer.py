"""
Phase 7a -- LIME local explainability for the XGBoost claims model.

LimeTabularExplainer is fitted once on a background sample drawn from
claims_features.parquet.  Each call to explain_claim() returns ranked
feature coefficients suitable for member-facing denial letters.

Usage (standalone):
  python src/explainability/lime_explainer.py --config config.yaml
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from lime import lime_tabular

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.registry import load_xgb

# Plain-English labels for each feature, used in denial letters
_FEATURE_LABELS: dict[str, str] = {
    "prior_auth_present":    "Prior authorization on file",
    "clm_pmt_amt":           "Claim payment amount",
    "primary_payer_amt":     "Primary payer amount",
    "has_primary_payer":     "Primary payer present",
    "is_adjustment":         "Adjustment claim indicator",
    "is_inpatient":          "Inpatient admission",
    "los_days":              "Length of stay (days)",
    "claim_year":            "Claim year",
    "claim_month":           "Claim month",
    "primary_icd9_num":      "Primary diagnosis code (numeric)",
    "num_diagnoses":         "Number of diagnosis codes",
    "num_procedures":        "Number of procedure codes",
    "has_procedure":         "Procedure code present",
    "has_hcpcs":             "HCPCS supply/service code present",
    "has_op_surgeon":        "Operating surgeon NPI present",
    "drg_num":               "DRG (Diagnosis Related Group)",
    "utlztn_days":           "Utilization days",
    "bene_age":              "Beneficiary age",
    "bene_sex":              "Beneficiary sex",
    "bene_race":             "Beneficiary race",
    "bene_esrd":             "End-Stage Renal Disease indicator",
    "bene_hi_mons":          "Hospital Insurance enrollment months",
    "bene_hmo_mons":         "HMO enrollment months",
    "bene_is_deceased":      "Deceased beneficiary flag",
    "has_alzheimer":         "Alzheimer's disease / dementia",
    "has_chf":               "Congestive heart failure",
    "has_ckd":               "Chronic kidney disease",
    "has_cancer":            "Cancer",
    "has_copd":              "Chronic obstructive pulmonary disease",
    "has_depression":        "Depression",
    "has_diabetes":          "Diabetes",
    "has_ihd":               "Ischemic heart disease",
    "has_osteoporosis":      "Osteoporosis",
    "has_ra_oa":             "Rheumatoid / osteo arthritis",
    "has_stroke":            "Stroke / TIA",
    "chronic_count":         "Total chronic condition count",
}

# Categorical feature indices (0/1 flags) for LIME — everything not continuous
_CATEGORICAL_NAMES: dict[int, list[str]] = {}  # populated in __init__


class LIMEExplainer:
    def __init__(
        self,
        model: Any,
        feature_cols: list[str],
        X_background: pd.DataFrame,
        cfg: dict,
        random_state: int = 42,
    ) -> None:
        self.model        = model
        self.feature_cols = feature_cols
        self.num_features = cfg["lime"].get("num_features", 10)
        self.num_samples  = cfg["lime"].get("num_samples", 1000)
        # Column medians used to impute NaN before LIME (LIME's sklearn internals reject NaN)
        self._col_medians: dict[str, float] = X_background.median().to_dict()

        # Identify binary/categorical columns by unique-value count in background
        cat_indices = [
            i for i, col in enumerate(feature_cols)
            if X_background[col].dropna().nunique() <= 4
        ]

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data        = X_background.values.astype(float),
            feature_names        = feature_cols,
            class_names          = ["approved", "denied"],
            categorical_features = cat_indices,
            discretize_continuous= False,   # avoids truncnorm zero-scale error on constant cols
            mode                 = "classification",
            random_state         = random_state,
        )

    # ------------------------------------------------------------------
    # Core explain
    # ------------------------------------------------------------------

    def explain_claim(self, X_row: pd.DataFrame) -> dict:
        """
        Run LIME on a single claim row.

        Returns:
          prediction_prob  -- model probability of denial (class 1)
          coefficients     -- {feature: coeff} for all LIME features (positive = toward denial)
          top_reasons      -- list of {feature, label, coeff, direction} sorted by |coeff|
          intercept        -- LIME local intercept
          local_pred       -- LIME local linear prediction at this point
        """
        if len(X_row) != 1:
            raise ValueError("X_row must be a single-row DataFrame")

        # LIME's sklearn distance calculation rejects NaN — impute with column medians
        X_filled = X_row.copy()
        for col, med in self._col_medians.items():
            if col in X_filled.columns:
                X_filled[col] = X_filled[col].fillna(med)

        x_arr = X_filled.values[0].astype(float)
        prob = float(self.model.predict_proba(X_row)[0, 1])

        medians = self._col_medians
        feat_cols = self.feature_cols

        def _predict_fn(arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(arr, columns=feat_cols).astype(float)
            for col, med in medians.items():
                if col in df.columns:
                    df[col] = df[col].fillna(med)
            return self.model.predict_proba(df)

        exp = self.explainer.explain_instance(
            data_row        = x_arr,
            predict_fn      = _predict_fn,
            num_features    = self.num_features,
            num_samples     = self.num_samples,
            top_labels      = 1,
        )

        label = 1  # denied class
        raw_list = exp.as_list(label=label)  # [(feature_condition, coeff), ...]

        # Map LIME condition strings back to feature names
        coefficients: dict[str, float] = {}
        for condition, coeff in raw_list:
            # LIME formats conditions like "prior_auth_present <= 0.50"
            # Extract the leading feature name
            feat = _parse_feature_name(condition, self.feature_cols)
            coefficients[feat] = float(coeff)

        top_reasons = sorted(
            [
                {
                    "feature":   f,
                    "label":     _FEATURE_LABELS.get(f, f),
                    "coeff":     v,
                    "direction": "increases_denial" if v > 0 else "decreases_denial",
                    "condition": next((c for c, _ in raw_list if f in c), ""),
                }
                for f, v in coefficients.items()
            ],
            key=lambda x: abs(x["coeff"]),
            reverse=True,
        )

        intercept  = float(exp.intercept[label])
        local_pred = float(exp.local_pred[0] if hasattr(exp.local_pred, "__len__") else exp.local_pred)

        return {
            "prediction_prob": prob,
            "coefficients":    coefficients,
            "top_reasons":     top_reasons,
            "intercept":       intercept,
            "local_pred":      local_pred,
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_lime_bar(self, lime_result: dict, out_path: Path | None = None) -> bytes:
        """Horizontal bar chart of LIME coefficients; returns PNG bytes."""
        reasons = lime_result["top_reasons"]
        labels  = [r["label"] for r in reasons]
        coeffs  = [r["coeff"] for r in reasons]
        colors  = ["#d73027" if c > 0 else "#4575b4" for c in coeffs]

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
        y_pos   = range(len(labels))
        ax.barh(list(y_pos), coeffs[::-1] if False else coeffs, color=colors)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME coefficient (positive = toward denial)")
        ax.set_title(
            f"LIME local explanation  |  Denial prob: {lime_result['prediction_prob']:.1%}"
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, dpi=120, format="png", bbox_inches="tight")
        plt.close("all")
        png_bytes = buf.getvalue()

        if out_path is not None:
            out_path.write_bytes(png_bytes)

        return png_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_feature_name(condition: str, feature_cols: list[str]) -> str:
    """
    LIME condition strings look like:
      "prior_auth_present <= 0.50"
      "0.50 < clm_pmt_amt <= 120.00"
    Match the longest feature name contained in the condition string.
    """
    matches = [f for f in feature_cols if f in condition]
    if not matches:
        return condition.split()[0]
    return max(matches, key=len)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models_dir    = ROOT / cfg["paths"]["models"]
    processed_dir = ROOT / cfg["paths"]["data_processed"]

    print("Loading XGBoost model ...")
    model, feature_cols = load_xgb(models_dir)

    feat_path = processed_dir / "claims_features.parquet"
    label_col = "denial_label"

    print(f"Loading features from {feat_path.name} ...", end=" ", flush=True)
    table = pq.read_table(feat_path, columns=feature_cols + [label_col])
    df    = table.to_pandas()
    print(f"{len(df):,} rows")

    # Background sample for LIME (stratified)
    rng       = np.random.default_rng(42)
    bg_idx    = rng.choice(len(df), size=500, replace=False)
    X_bg      = df.iloc[bg_idx][feature_cols].reset_index(drop=True)

    print("Fitting LIME explainer on 500-row background sample ...")
    lime_exp = LIMEExplainer(model, feature_cols, X_bg, cfg)

    # Demo: denied claim with prior_auth_present = 0
    mask     = (df[label_col] == 1) & (df["prior_auth_present"] == 0)
    demo_idx = df[mask].index[0]
    X_row    = df.loc[[demo_idx], feature_cols]

    print(f"\n=== LIME local explanation (claim index {demo_idx}) ===")
    result = lime_exp.explain_claim(X_row)
    print(f"  Denial probability : {result['prediction_prob']:.4f}")
    print(f"  LIME local pred    : {result['local_pred']:.4f}")
    print(f"  LIME intercept     : {result['intercept']:.4f}")

    print("\n  Top reasons (sorted by |coeff|):")
    for r in result["top_reasons"]:
        sign = "+" if r["coeff"] > 0 else ""
        print(f"    {sign}{r['coeff']:+.4f}  {r['label']:<40s}  ({r['condition']})")

    lime_path = models_dir / "lime_bar_demo.png"
    lime_exp.plot_lime_bar(result, out_path=lime_path)
    print(f"\n  LIME bar chart -> {lime_path}")

    print("\nPhase 7a complete.")


if __name__ == "__main__":
    main()
