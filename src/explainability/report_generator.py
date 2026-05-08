"""
Phase 7b -- Denial letter and adjuster summary report generator.

Combines SHAP local result + LIME top_reasons into two document types:

1. Member denial letter (ACA-compliant plain language)
   - Lists specific denial reasons derived from LIME coefficients
   - Includes counterfactual guidance (from SHAP) and appeal instructions

2. Adjuster summary (internal, more technical)
   - SHAP waterfall interpretation
   - LIME coefficients
   - Model confidence and override recommendation

Usage (standalone):
  python src/explainability/report_generator.py --config config.yaml
"""

import argparse
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.registry import load_xgb
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer

# ---------------------------------------------------------------------------
# Letter templates
# ---------------------------------------------------------------------------

_MEMBER_HEADER = """\
NOTICE OF CLAIM DETERMINATION
--------------------------------------------------------------
Date:          {today}
Member ID:     {member_id}
Claim ID:      {claim_id}
Service Date:  {service_date}
Provider:      {provider}
--------------------------------------------------------------

Dear Member,

We have reviewed your claim and have made the following determination:

  CLAIM STATUS: {status}
  Denial Probability Score: {prob:.1%}

"""

_MEMBER_APPROVED = """\
Your claim has been APPROVED for processing.  Payment will be issued
in accordance with your plan benefits.
"""

_MEMBER_DENIED_INTRO = """\
Your claim has been DENIED.  The specific reasons for this determination
are listed below.  You have the right to appeal this decision.

REASONS FOR DENIAL:
"""

_MEMBER_APPEAL = """\

YOUR RIGHT TO APPEAL
--------------------------------------------------------------
You have 180 days from the date of this notice to file an appeal.
To initiate an appeal, please contact us at:

  Phone : 1-800-555-0100
  Mail  : Appeals & Grievances Department
          PO Box 12345, Anytown, USA 00000
  Online: member.healthplan.example/appeals

Please include this notice and any supporting clinical documentation
with your appeal submission.

If you believe this denial is related to a data or administrative error
(such as a missing prior authorization reference), please contact your
provider's billing office to submit a corrected claim.

This determination was assisted by an AI system.  All final decisions
are subject to review by a licensed claims adjuster.
--------------------------------------------------------------
"""

_ADJUSTER_HEADER = """\
ADJUSTER REVIEW SUMMARY (INTERNAL)
======================================================
Claim ID       : {claim_id}
Member ID      : {member_id}
Review Date    : {today}
Model          : XGBoost v1 (health_claims_xai)
------------------------------------------------------
Denial Probability : {prob:.4f}  ({prob:.1%})
Recommended Action : {action}

======================================================

"""

_ADJUSTER_SHAP = """\
SHAP EXPLANATION (top drivers, log-odds scale)
------------------------------------------------------
Base value (log-odds baseline): {base_value:+.4f}

{shap_rows}

Note: positive SHAP -> pushes toward denial; negative -> toward approval.

"""

_ADJUSTER_LIME = """\
LIME LOCAL EXPLANATION (top {n} features)
------------------------------------------------------
Local intercept : {intercept:+.4f}
Local prediction: {local_pred:+.4f}

{lime_rows}

"""

_ADJUSTER_COUNTERFACTUAL = """\
COUNTERFACTUAL GUIDANCE
------------------------------------------------------
{hints}

"""

_ADJUSTER_FOOTER = """\
======================================================
OVERRIDE INSTRUCTIONS
If you override this determination, document the clinical or
administrative reason in the audit panel before submitting.
======================================================
"""


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    def __init__(self, shap_exp: SHAPExplainer, lime_exp: LIMEExplainer) -> None:
        self.shap_exp = shap_exp
        self.lime_exp = lime_exp

    def generate(
        self,
        X_row: pd.DataFrame,
        claim_meta: dict,
    ) -> dict[str, str]:
        """
        Run SHAP + LIME on X_row and return:
          member_letter    -- ACA-compliant plain-language denial notice
          adjuster_summary -- internal technical review sheet
          shap_result      -- raw SHAP dict (for API/dashboard)
          lime_result      -- raw LIME dict (for API/dashboard)
        """
        shap_result = self.shap_exp.explain_local(X_row)
        lime_result = self.lime_exp.explain_claim(X_row)
        prob        = shap_result["prediction_prob"]
        is_denied   = prob > 0.5
        status      = "DENIED" if is_denied else "APPROVED"
        action      = "DENY - review counterfactual hints below" if is_denied else "APPROVE"

        today        = claim_meta.get("today", str(date.today()))
        member_id    = claim_meta.get("member_id", "UNKNOWN")
        claim_id     = claim_meta.get("claim_id", "UNKNOWN")
        service_date = claim_meta.get("service_date", "UNKNOWN")
        provider     = claim_meta.get("provider", "UNKNOWN")

        # ── Member letter ────────────────────────────────────────────────────
        letter = _MEMBER_HEADER.format(
            today=today, member_id=member_id, claim_id=claim_id,
            service_date=service_date, provider=provider,
            status=status, prob=prob,
        )

        if not is_denied:
            letter += _MEMBER_APPROVED
        else:
            letter += _MEMBER_DENIED_INTRO
            denial_reasons = [r for r in lime_result["top_reasons"][:5] if r["coeff"] > 0]
            for i, reason in enumerate(denial_reasons, 1):
                letter += f"  {i}. {reason['label']}\n"
                letter += _denial_plain_text(reason["feature"], reason["coeff"])

            hints = self.shap_exp.counterfactual_hints(shap_result, top_n=3)
            if hints and "APPROVED" not in hints[0]:
                letter += "\nACTIONS THAT MAY RESOLVE THIS DENIAL:\n"
                for h in hints:
                    # Strip the [feature SHAP=...] prefix for member-facing text
                    clean = h.split("]", 1)[-1].strip() if "]" in h else h
                    letter += textwrap.fill(f"  - {clean}", width=70, subsequent_indent="    ") + "\n"

        letter += _MEMBER_APPEAL

        # ── Adjuster summary ─────────────────────────────────────────────────
        shap_rows = _format_shap_table(shap_result["top_drivers"][:15])
        lime_rows = _format_lime_table(lime_result["top_reasons"])
        hints_text = "\n".join(
            f"  {h}" for h in self.shap_exp.counterfactual_hints(shap_result)
        )

        adjuster = _ADJUSTER_HEADER.format(
            claim_id=claim_id, member_id=member_id, today=today,
            prob=prob, action=action,
        )
        adjuster += _ADJUSTER_SHAP.format(
            base_value=shap_result["base_value"], shap_rows=shap_rows,
        )
        adjuster += _ADJUSTER_LIME.format(
            n=len(lime_result["top_reasons"]),
            intercept=lime_result["intercept"],
            local_pred=lime_result["local_pred"],
            lime_rows=lime_rows,
        )
        adjuster += _ADJUSTER_COUNTERFACTUAL.format(hints=hints_text)
        adjuster += _ADJUSTER_FOOTER

        return {
            "member_letter":    letter,
            "adjuster_summary": adjuster,
            "shap_result":      shap_result,
            "lime_result":      lime_result,
        }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _denial_plain_text(feature: str, coeff: float) -> str:
    texts = {
        "prior_auth_present":
            "     Our records do not show a prior authorization reference for this service.\n"
            "     Please contact your provider to submit a corrected claim with the\n"
            "     authorization number if one was obtained.\n",
        "clm_pmt_amt":
            "     The billed amount falls outside the expected range for this service\n"
            "     under the applicable fee schedule.\n",
        "is_adjustment":
            "     This is an adjustment claim.  Please include the original claim number\n"
            "     and a detailed reason for adjustment.\n",
        "num_diagnoses":
            "     The claim contains fewer diagnosis codes than expected for this service.\n"
            "     Ensure all active conditions are documented.\n",
        "has_procedure":
            "     No procedure codes were found on this claim.  Please verify coding\n"
            "     and resubmit if procedures were performed.\n",
        "drg_num":
            "     The DRG assignment could not be validated against the documented\n"
            "     diagnoses and procedures.  Please review and resubmit.\n",
    }
    return texts.get(feature, "     Please review this item with your provider's billing office.\n") + "\n"


def _format_shap_table(drivers: list[dict]) -> str:
    lines = []
    for d in drivers:
        bar   = "#" * int(min(abs(d["shap_value"]) * 40, 30))
        sign  = "+" if d["shap_value"] > 0 else "-"
        lines.append(
            f"  {sign}{abs(d['shap_value']):.4f}  {d['feature']:<25s}  {bar}"
        )
    return "\n".join(lines)


def _format_lime_table(reasons: list[dict]) -> str:
    lines = []
    for r in reasons:
        sign = "+" if r["coeff"] > 0 else ""
        lines.append(
            f"  {sign}{r['coeff']:+.4f}  {r['label']:<40s}  ({r['condition']})"
        )
    return "\n".join(lines)


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

    rng    = np.random.default_rng(42)
    bg_idx = rng.choice(len(df), size=500, replace=False)
    X_bg   = df.iloc[bg_idx][feature_cols].reset_index(drop=True)

    print("Initialising SHAP explainer ...")
    shap_exp = SHAPExplainer(model, feature_cols)

    print("Fitting LIME explainer ...")
    lime_exp = LIMEExplainer(model, feature_cols, X_bg, cfg)

    rpt = ReportGenerator(shap_exp, lime_exp)

    # Demo: denied claim, prior_auth_present = 0
    mask     = (df[label_col] == 1) & (df["prior_auth_present"] == 0)
    demo_idx = df[mask].index[0]
    X_row    = df.loc[[demo_idx], feature_cols]

    claim_meta = {
        "today":        str(date.today()),
        "member_id":    f"MBR-{demo_idx:07d}",
        "claim_id":     f"CLM-{demo_idx:010d}",
        "service_date": "2008-01-01",
        "provider":     "General Hospital (NPI: N/A)",
    }

    print(f"\nGenerating reports for claim index {demo_idx} ...")
    reports = rpt.generate(X_row, claim_meta)

    # Save reports
    letter_path   = models_dir / "denial_letter_demo.txt"
    adjuster_path = models_dir / "adjuster_summary_demo.txt"
    letter_path.write_text(reports["member_letter"],    encoding="utf-8")
    adjuster_path.write_text(reports["adjuster_summary"], encoding="utf-8")

    print(f"\n  Member denial letter   -> {letter_path}")
    print(f"  Adjuster summary       -> {adjuster_path}")

    print("\n--- MEMBER DENIAL LETTER (preview) ---")
    print(reports["member_letter"][:1200])
    print("... [truncated] ...")

    print("\n--- ADJUSTER SUMMARY (preview) ---")
    print(reports["adjuster_summary"][:1200])
    print("... [truncated] ...")

    print("\nPhase 7 complete.")


if __name__ == "__main__":
    main()
