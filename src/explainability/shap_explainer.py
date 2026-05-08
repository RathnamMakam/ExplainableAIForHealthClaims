"""
Phase 6 -- SHAP global and local explainability for the XGBoost claims model.

Global:
  - Beeswarm summary plot (top-N features by mean |SHAP|)
  - Bar summary plot (mean absolute SHAP per feature)
  - SHAP values saved to data/processed/shap_global.parquet

Local (per-claim):
  - Waterfall plot saved to a temp path or models/
  - Dict of {feature: shap_value} for API response
  - Counterfactual hint text for denied claims

Usage (standalone):
  python src/explainability/shap_explainer.py --config config.yaml
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
import shap
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.registry import load_xgb

# Actionable hint templates keyed by feature name
_HINTS: dict[str, str] = {
    "prior_auth_present":    "Obtain prior authorization -- missing NPI indicates auth was not on file at time of claim submission.",
    "clm_pmt_amt":           "Review the billed amount against the Medicare fee schedule; an unusually high or negative amount triggers elevated denial risk.",
    "primary_payer_amt":     "Verify primary payer coordination -- discrepancies in primary payer amount increase denial probability.",
    "has_primary_payer":     "Confirm primary payer information is complete; missing primary payer data raises denial risk.",
    "is_adjustment":         "Adjustment claims carry higher denial risk; include the original claim reference and detailed reason.",
    "num_diagnoses":         "Ensure all supporting diagnosis codes are documented; low diagnosis count may signal incomplete documentation.",
    "num_procedures":        "Verify all performed procedures are coded; undercoding procedures can reduce medical necessity support.",
    "has_procedure":         "Attach procedure documentation -- claims without procedure codes have elevated denial rates.",
    "has_hcpcs":             "Supply HCPCS codes for all supplies and services rendered.",
    "primary_icd9_num":      "Confirm the primary diagnosis code is the most specific ICD code available for the condition.",
    "drg_num":               "Verify DRG assignment is accurate for the documented diagnoses and procedures.",
    "utlztn_days":           "Document medical necessity for each utilization day; long stays without clear progression trigger review.",
    "los_days":              "Provide clinical justification for length of stay; unusually long stays relative to DRG may be denied.",
    "chronic_count":         "Ensure all active chronic conditions are captured in the claim; under-reporting comorbidities reduces risk-adjustment.",
    "bene_age":              "No corrective action -- patient age is a read-only demographic factor.",
    "is_inpatient":          "If this is an inpatient claim, confirm the inpatient admission criteria are met and documented.",
    "has_op_surgeon":        "Ensure the operating surgeon NPI is included for all surgical procedures.",
}

_DEFAULT_HINT = "Review supporting documentation for this feature and ensure the claim record is complete and accurate."


class SHAPExplainer:
    def __init__(self, model: Any, feature_cols: list[str]) -> None:
        self.model        = model
        self.feature_cols = feature_cols
        self.explainer    = shap.TreeExplainer(model)
        # base_value: log-odds baseline
        # TreeExplainer returns a scalar, 1-element array, or 2-element list depending on model type
        bv = self.explainer.expected_value
        if hasattr(bv, "__len__"):
            self.base_value: float = float(bv[1] if len(bv) > 1 else bv[0])
        else:
            self.base_value: float = float(bv)

    # ------------------------------------------------------------------
    # Global
    # ------------------------------------------------------------------

    def compute_global(
        self,
        X: pd.DataFrame,
        cfg: dict,
        out_dir: Path,
        processed_dir: Path,
    ) -> np.ndarray:
        """
        Compute SHAP values on a random sample, save summary plots and
        a parquet of raw values.  Returns the shap_values array.
        """
        bg_n     = cfg["shap"].get("background_samples", 100)
        max_disp = cfg["shap"].get("max_display", 20)
        sample_n = min(2000, len(X))

        rng     = np.random.default_rng(42)
        idx     = rng.choice(len(X), size=sample_n, replace=False)
        X_samp  = X.iloc[idx].reset_index(drop=True)

        print(f"  Computing SHAP values on {sample_n:,} samples ... ", end="", flush=True)
        shap_vals = self.explainer.shap_values(X_samp)
        # For binary XGBoost the output is a 2-D array (class 1 values)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        print("done")

        # -- beeswarm --
        fig_bee = out_dir / "shap_beeswarm.png"
        shap.summary_plot(
            shap_vals, X_samp,
            feature_names=self.feature_cols,
            max_display=max_disp,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(fig_bee, dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"  Beeswarm plot -> {fig_bee}")

        # -- bar summary --
        fig_bar = out_dir / "shap_bar_summary.png"
        shap.summary_plot(
            shap_vals, X_samp,
            feature_names=self.feature_cols,
            plot_type="bar",
            max_display=max_disp,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(fig_bar, dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"  Bar summary   -> {fig_bar}")

        # -- persist raw SHAP values --
        sv_df = pd.DataFrame(shap_vals, columns=self.feature_cols)
        sv_df.insert(0, "sample_idx", idx)
        sv_path = processed_dir / "shap_global.parquet"
        sv_df.to_parquet(sv_path, index=False)
        print(f"  SHAP values   -> {sv_path}  ({len(sv_df):,} rows x {len(self.feature_cols)} cols)")

        return shap_vals

    def global_feature_importance(self, shap_vals: np.ndarray) -> dict[str, float]:
        """Mean absolute SHAP per feature, sorted descending."""
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return dict(sorted(zip(self.feature_cols, mean_abs.tolist()), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Local
    # ------------------------------------------------------------------

    def explain_local(self, X_row: pd.DataFrame) -> dict:
        """
        Return a dict suitable for JSON / API response:
          base_value      -- log-odds baseline
          prediction_prob -- model probability for class 1
          shap_values     -- {feature: shap_value}
          top_drivers     -- list of {feature, shap_value, direction} sorted by |shap|
        """
        if len(X_row) != 1:
            raise ValueError("X_row must be a single-row DataFrame")

        sv = self.explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]
        sv_flat = sv[0]  # shape (n_features,)

        prob = float(self.model.predict_proba(X_row)[0, 1])

        shap_dict = {feat: float(val) for feat, val in zip(self.feature_cols, sv_flat)}

        top_drivers = sorted(
            [{"feature": f, "shap_value": v, "direction": "increases_denial" if v > 0 else "decreases_denial"}
             for f, v in shap_dict.items()],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        return {
            "base_value":      self.base_value,
            "prediction_prob": prob,
            "shap_values":     shap_dict,
            "top_drivers":     top_drivers,
        }

    def plot_local_waterfall(self, X_row: pd.DataFrame, out_path: Path | None = None) -> bytes:
        """
        Render a waterfall plot for a single claim.  Saves to out_path if given,
        and always returns the PNG bytes (for API streaming).
        """
        sv = self.explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]

        exp = shap.Explanation(
            values        = sv[0],
            base_values   = self.base_value,
            data          = X_row.values[0],
            feature_names = self.feature_cols,
        )

        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, dpi=120, format="png", bbox_inches="tight")
        plt.close("all")
        png_bytes = buf.getvalue()

        if out_path is not None:
            out_path.write_bytes(png_bytes)
            print(f"  Waterfall     -> {out_path}")

        return png_bytes

    # ------------------------------------------------------------------
    # Counterfactual hints
    # ------------------------------------------------------------------

    def counterfactual_hints(self, local_result: dict, top_n: int = 3) -> list[str]:
        """
        For denied claims (prob > 0.5): return actionable text for the top-N
        features that pushed the score toward denial (positive SHAP values).
        """
        prob = local_result["prediction_prob"]
        if prob <= 0.5:
            return ["This claim is predicted as APPROVED -- no corrective actions required."]

        positive_drivers = [
            d for d in local_result["top_drivers"] if d["shap_value"] > 0
        ][:top_n]

        hints = []
        for d in positive_drivers:
            feat = d["feature"]
            sv   = d["shap_value"]
            hint = _HINTS.get(feat, _DEFAULT_HINT)
            hints.append(f"[{feat}  SHAP={sv:+.4f}]  {hint}")

        return hints


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
    explainer = SHAPExplainer(model, feature_cols)
    print(f"  Base value (log-odds baseline): {explainer.base_value:.4f}")

    # ------------------------------------------------------------------ #
    # Global SHAP                                                          #
    # ------------------------------------------------------------------ #
    print("\n=== Global SHAP ===")
    feat_path = processed_dir / "claims_features.parquet"
    print(f"Loading features from {feat_path.name} ...", end=" ", flush=True)
    table = pq.read_table(feat_path, columns=feature_cols)
    X_all = table.to_pandas()
    print(f"{len(X_all):,} rows")

    shap_vals = explainer.compute_global(X_all, cfg, models_dir, processed_dir)

    print("\nTop-10 features by mean |SHAP|:")
    importance = explainer.global_feature_importance(shap_vals)
    for i, (feat, val) in enumerate(list(importance.items())[:10], 1):
        print(f"  {i:2d}. {feat:<25s}  {val:.5f}")

    # ------------------------------------------------------------------ #
    # Local SHAP -- find a denied claim with prior_auth_present = 0       #
    # ------------------------------------------------------------------ #
    print("\n=== Local SHAP (demo: denied claim, no prior auth) ===")
    label_col = "denial_label"
    table2 = pq.read_table(feat_path, columns=feature_cols + [label_col])
    df_full = table2.to_pandas()

    mask = (df_full[label_col] == 1) & (df_full["prior_auth_present"] == 0)
    demo_idx = df_full[mask].index[0]
    X_row = df_full.loc[[demo_idx], feature_cols]

    print(f"  Claim index: {demo_idx}  |  prior_auth_present: {int(X_row['prior_auth_present'].iloc[0])}")

    local = explainer.explain_local(X_row)
    print(f"  Denial probability : {local['prediction_prob']:.4f}")
    print(f"  Base value         : {local['base_value']:.4f}")

    print("\n  Top-10 SHAP drivers:")
    for d in local["top_drivers"][:10]:
        bar = "#" * int(abs(d["shap_value"]) * 50)
        sign = "+" if d["shap_value"] > 0 else "-"
        print(f"    {sign}{abs(d['shap_value']):.4f}  {d['feature']:<25s}  {bar}")

    waterfall_path = models_dir / "shap_waterfall_demo.png"
    explainer.plot_local_waterfall(X_row, out_path=waterfall_path)

    print("\n  Counterfactual hints:")
    for hint in explainer.counterfactual_hints(local):
        print(f"    - {hint}")

    print("\nPhase 6 complete.")
    print(f"  Beeswarm   : {models_dir / 'shap_beeswarm.png'}")
    print(f"  Bar summary: {models_dir / 'shap_bar_summary.png'}")
    print(f"  Waterfall  : {waterfall_path}")
    print(f"  SHAP parquet: {processed_dir / 'shap_global.parquet'}")


if __name__ == "__main__":
    main()
