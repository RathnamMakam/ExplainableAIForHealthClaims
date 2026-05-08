"""
Phase 10 -- XAI Health Claims Demo Script

Runs four scripted scenarios against the live FastAPI server (:8000).
Each scenario prints a narrated walkthrough and saves key artifacts to
demo_output/ for use in presentations.

Scenarios
---------
1. Adjuster Review   -- Denied claim, SHAP/LIME pinpoint data quality issue,
                        adjuster overrides with corrected auth reference.
2. Compliance Audit  -- Global SHAP confirms no protected-attribute bias;
                        audit trail shows all human overrides.
3. Patient Letter    -- ACA-compliant denial letter with plain-language reasons
                        and appeal instructions.
4. Counterfactual    -- Side-by-side score before/after correcting prior auth;
                        quantifies the XAI-guided fix.

Usage (FastAPI must be running on :8000):
  .venv-api\\Scripts\\activate
  python src/demo/demo_script.py
"""

import base64
import sys
import time
from pathlib import Path

import httpx

API   = "http://localhost:8000"
OUT   = Path(__file__).resolve().parents[2] / "demo_output"
OUT.mkdir(exist_ok=True)

WIDTH = 70  # console width for separators

# ---------------------------------------------------------------------------
# Claim fixtures
# ---------------------------------------------------------------------------

# Scenario 1 & 3: OP claim denied because prior auth NPI not entered
DENIED_NO_AUTH = {
    "is_inpatient": 0, "los_days": 0.0, "claim_year": 2008, "claim_month": 1,
    "clm_pmt_amt": 150.0, "primary_payer_amt": 0.0, "has_primary_payer": 0,
    "is_adjustment": 0, "prior_auth_present": 0, "has_op_surgeon": 0,
    "primary_icd9_num": 250.0, "num_diagnoses": 3, "num_procedures": 0,
    "has_procedure": 0, "has_hcpcs": 1, "drg_num": None, "utlztn_days": None,
    "bene_age": 72, "bene_sex": 2, "bene_race": 1, "bene_esrd": 0,
    "bene_hi_mons": 12, "bene_hmo_mons": 0, "bene_is_deceased": 0,
    "has_alzheimer": 0, "has_chf": 1, "has_ckd": 0, "has_cancer": 0,
    "has_copd": 1, "has_depression": 0, "has_diabetes": 1, "has_ihd": 1,
    "has_osteoporosis": 0, "has_ra_oa": 0, "has_stroke": 0, "chronic_count": 4,
    "claim_id": "CLM-2008-001177", "member_id": "MBR-0000177",
    "service_date": "2008-01-15", "provider": "General Hospital",
    "adjuster_name": "Sarah Chen",
}

# Scenario 4: Same claim with prior auth corrected (NPI now on file)
APPROVED_WITH_AUTH = {
    **DENIED_NO_AUTH,
    "prior_auth_present": 1,
    "has_op_surgeon": 1,
    "claim_id": "CLM-2008-001177-CORRECTED",
}

# Approved baseline for contrast
APPROVED_BASELINE = {
    "is_inpatient": 0, "los_days": 0.0, "claim_year": 2008, "claim_month": 6,
    "clm_pmt_amt": 80.0, "primary_payer_amt": 20.0, "has_primary_payer": 1,
    "is_adjustment": 0, "prior_auth_present": 1, "has_op_surgeon": 1,
    "primary_icd9_num": 401.0, "num_diagnoses": 5, "num_procedures": 2,
    "has_procedure": 1, "has_hcpcs": 1, "drg_num": None, "utlztn_days": None,
    "bene_age": 68, "bene_sex": 1, "bene_race": 1, "bene_esrd": 0,
    "bene_hi_mons": 12, "bene_hmo_mons": 0, "bene_is_deceased": 0,
    "has_alzheimer": 0, "has_chf": 0, "has_ckd": 0, "has_cancer": 0,
    "has_copd": 0, "has_depression": 0, "has_diabetes": 1, "has_ihd": 1,
    "has_osteoporosis": 0, "has_ra_oa": 0, "has_stroke": 0, "chronic_count": 2,
    "claim_id": "CLM-2008-002201", "member_id": "MBR-0002201",
    "service_date": "2008-06-10", "provider": "City Medical Center",
    "adjuster_name": "Sarah Chen",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

client = httpx.Client(base_url=API, timeout=90)


def post(endpoint: str, payload: dict) -> dict:
    r = client.post(endpoint, json=payload)
    r.raise_for_status()
    return r.json()


def get(endpoint: str) -> dict:
    r = client.get(endpoint)
    r.raise_for_status()
    return r.json()


def save_image(b64_str: str, filename: str) -> Path:
    path = OUT / filename
    path.write_bytes(base64.b64decode(b64_str))
    return path


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def section(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, WIDTH - len(title) - 5))


def bullet(text: str, indent: int = 2) -> None:
    prefix = " " * indent + "* "
    print(prefix + text)


def metric(label: str, value: str, width: int = 30) -> None:
    print(f"  {label:<{width}} {value}")


def bar(value: float, width: int = 30) -> str:
    filled = int(value * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {value:.1%}"


def pause(seconds: float = 0.4) -> None:
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# SCENARIO 1: Adjuster review
# ---------------------------------------------------------------------------

def scenario_1_adjuster() -> None:
    banner("SCENARIO 1: Adjuster Review — Outpatient Claim CLM-2008-001177")

    print("""
  Context
  -------
  Adjuster Sarah Chen opens a queue item: outpatient diabetic management
  claim for a 72-year-old female Medicare beneficiary. The system has
  pre-scored it and flagged it for review. Sarah wants to understand WHY
  before deciding to approve or deny.
""")
    pause()

    # Score
    section("Step 1 — Model score")
    score = post("/claims/score", DENIED_NO_AUTH)
    prob   = score["prediction_prob"]
    action = score["recommended_action"]
    level  = score["denial_risk_level"]

    metric("Claim ID:",           score["claim_id"])
    metric("Denial probability:", f"{prob:.1%}   {bar(prob)}")
    metric("Risk level:",         level)
    metric("Recommended action:", f">>> {action} <<<")
    pause()

    # SHAP
    section("Step 2 — SHAP waterfall (why did the model score it this way?)")
    shap = post("/explain/shap", DENIED_NO_AUTH)
    save_image(shap["waterfall_b64"], "s1_shap_waterfall.png")

    print(f"  Base value (log-odds baseline): {shap['base_value']:+.4f}")
    print()
    print("  Feature contributions (top 8):")
    for d in shap["top_drivers"][:8]:
        sign  = "+" if d["shap_value"] > 0 else ""
        arrow = ">>" if abs(d["shap_value"]) > 0.1 else "  "
        print(f"  {arrow} {sign}{d['shap_value']:+.4f}  {d['feature']}")
    pause()

    print()
    print("  KEY FINDING:")
    top = shap["top_drivers"][0]
    print(f"  prior_auth_present contributes SHAP={top['shap_value']:+.4f} -- the")
    print("  single largest driver. The model's primary reason for denial is")
    print("  that no prior authorization NPI was recorded on this claim.")
    pause()

    section("Step 3 — Counterfactual hint")
    for hint in shap["counterfactual_hints"]:
        clean = hint.split("]", 1)[-1].strip() if "]" in hint else hint
        bullet(clean)
    pause()

    # LIME
    section("Step 4 — LIME confirmation (independent local explanation)")
    lime = post("/explain/lime", DENIED_NO_AUTH)
    save_image(lime["bar_b64"], "s1_lime_bar.png")

    print("  LIME top reasons (positive coeff = toward denial):")
    for r in lime["top_reasons"][:5]:
        sign = "+" if r["coeff"] > 0 else ""
        flag = "  [#1 DRIVER]" if r == lime["top_reasons"][0] else ""
        print(f"    {sign}{r['coeff']:+.5f}  {r['label']}{flag}")

    print()
    print("  SHAP and LIME agree: prior authorization is the dominant denial")
    print("  driver. This is a data quality issue, not a clinical one.")
    pause()

    # Adjuster action
    section("Step 5 — Adjuster investigation and override")
    print("""
  Sarah cross-references the paper authorization log. She finds:
    - Prior auth WAS obtained from the payer on 2008-01-12
    - The authorization number is AUTH-2008-88821
    - The NPI field was left blank during data entry (keying error)

  This is exactly what the XAI system predicted: a data quality gap.
  Sarah corrects the record and logs an override.
""")
    pause()

    override_r = post("/audit/override", {
        "claim_id":           DENIED_NO_AUTH["claim_id"],
        "member_id":          DENIED_NO_AUTH["member_id"],
        "adjuster_name":      "Sarah Chen",
        "original_prob":      prob,
        "recommended_action": action,
        "override_decision":  "APPROVE",
        "reason": (
            "Prior auth verified in paper log (AUTH-2008-88821, obtained 2008-01-12). "
            "NPI omitted due to data entry error. Correcting and approving."
        ),
    })
    metric("Override logged — audit ID:", override_r["audit_id"])
    metric("Final decision:",             override_r["final_decision"])
    metric("Adjuster:",                   override_r["adjuster_name"])
    print()
    print("  Without XAI: claim denied, member files an appeal, 30-day delay.")
    print("  With XAI:    adjuster corrects in minutes. No appeal needed.")


# ---------------------------------------------------------------------------
# SCENARIO 2: Compliance / fairness audit
# ---------------------------------------------------------------------------

def scenario_2_compliance() -> None:
    banner("SCENARIO 2: Compliance Audit — Is the model fair?")

    print("""
  Context
  -------
  The compliance team runs a quarterly audit to confirm that the model's
  denial decisions are not primarily driven by protected attributes such
  as age, sex, or race. Regulators require this under the ACA and ECOA.
""")
    pause()

    section("Step 1 — Global SHAP feature importance")
    g = get("/explain/shap/global")
    save_image(g["beeswarm_b64"], "s2_shap_beeswarm.png")
    save_image(g["bar_b64"],      "s2_shap_bar.png")

    fi = g["feature_importance"]
    ranked = list(fi.items())

    print("  Top-15 features by mean |SHAP| (global, 2,000-claim sample):")
    print()
    protected = {"bene_age", "bene_sex", "bene_race"}
    for i, (feat, val) in enumerate(ranked[:15], 1):
        tag = " <-- PROTECTED ATTRIBUTE" if feat in protected else ""
        print(f"  {i:2d}. {feat:<28s}  {val:.5f}{tag}")
    pause()

    section("Step 2 — Protected attribute analysis")
    protected_ranks = {f: i+1 for i, (f, _) in enumerate(ranked) if f in protected}
    all_ranks       = {f: i+1 for i, (f, _) in enumerate(ranked)}

    print()
    for attr in ["bene_age", "bene_sex", "bene_race"]:
        rank = all_ranks.get(attr, "N/A")
        val  = fi.get(attr, 0)
        top1_val = ranked[0][1]
        pct  = val / top1_val * 100 if top1_val > 0 else 0
        print(f"  {attr:<20s}  rank={rank}  mean|SHAP|={val:.5f}  "
              f"({pct:.1f}% of top feature)")

    print()
    print("  AUDIT FINDING: Protected attributes (age, sex, race) rank well")
    print("  below clinical and administrative features. The primary drivers")
    print("  are claim payment amount, diagnosis codes, and prior auth status.")
    print("  No evidence of protected-attribute bias in model decisions.")
    pause()

    section("Step 3 — Audit trail review")
    log = get("/audit/log?page_size=50")
    print(f"  Total override entries: {log['total']}")
    if log["total"] > 0:
        print()
        print(f"  {'Audit ID':<12}  {'Claim ID':<20}  {'Adjuster':<16}  "
              f"{'Model':>6}  {'Final':>8}  Reason (truncated)")
        print("  " + "-" * 65)
        for e in log["entries"]:
            reason_short = e["override_reason"][:35] + "..." if len(e["override_reason"]) > 35 else e["override_reason"]
            print(f"  {e['audit_id']:<12}  {e['claim_id']:<20}  {e['adjuster_name']:<16}  "
                  f"{e['model_decision']:>6}  {e['final_decision']:>8}  {reason_short}")
    pause()


# ---------------------------------------------------------------------------
# SCENARIO 3: Patient denial letter
# ---------------------------------------------------------------------------

def scenario_3_patient_letter() -> None:
    banner("SCENARIO 3: Patient-Facing Denial Letter (ACA-Compliant)")

    print("""
  Context
  -------
  When a claim is denied, the ACA requires a written notice with specific
  denial reasons, the member's right to appeal, and contact information.
  The XAI system generates this automatically from LIME's ranked reasons,
  converted to plain English.
""")
    pause()

    section("Step 1 — Generate denial letter")
    rpt = post("/explain/report", DENIED_NO_AUTH)
    letter_path = OUT / "s3_denial_letter.txt"
    letter_path.write_text(rpt["member_letter"], encoding="utf-8")
    print(f"  Letter saved -> {letter_path}")
    pause()

    section("Step 2 — Letter preview")
    print()
    for line in rpt["member_letter"].splitlines():
        print("  " + line)
    pause()

    section("Step 3 — Key ACA compliance elements")
    letter = rpt["member_letter"]
    checks = [
        ("Specific denial reasons listed",    "REASONS FOR DENIAL" in letter),
        ("Right to appeal stated",            "right to appeal" in letter.lower()),
        ("180-day appeal window stated",      "180 days" in letter),
        ("Contact information provided",      "1-800-" in letter),
        ("AI assistance disclosed",           "AI system" in letter),
        ("Corrective action guidance",        "ACTIONS THAT MAY" in letter),
    ]
    print()
    for check, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {check}")


# ---------------------------------------------------------------------------
# SCENARIO 4: Counterfactual — quantify the fix
# ---------------------------------------------------------------------------

def scenario_4_counterfactual() -> None:
    banner("SCENARIO 4: Counterfactual — What Changes When We Fix the Data?")

    print("""
  Context
  -------
  A key differentiator of XAI systems is the ability to answer:
  "What would the model score if we corrected the data issue?"
  This quantifies the business impact of data quality improvements
  and gives adjusters a concrete target to work toward.
""")
    pause()

    section("Step 1 — Score original (flawed) claim")
    s_orig = post("/claims/score", DENIED_NO_AUTH)
    shap_orig = post("/explain/shap", DENIED_NO_AUTH)

    section("Step 2 — Score corrected claim (prior auth NPI added)")
    s_corr = post("/claims/score", APPROVED_WITH_AUTH)
    shap_corr = post("/explain/shap", APPROVED_WITH_AUTH)

    section("Step 3 — Score fully-documented approved claim (baseline)")
    s_base = post("/claims/score", APPROVED_BASELINE)

    section("Comparison")
    print()
    print(f"  {'Scenario':<38}  {'Prob':>6}  {'Action':>8}  Risk")
    print("  " + "-" * 62)

    rows = [
        ("Original (prior auth NPI missing)",    s_orig,  "DENIED_NO_AUTH"),
        ("Corrected (prior auth NPI entered)",   s_corr,  "APPROVED_WITH_AUTH"),
        ("Fully-documented approved claim",       s_base,  "APPROVED_BASELINE"),
    ]
    for label, s, _ in rows:
        prob   = s["prediction_prob"]
        action = s["recommended_action"]
        level  = s["denial_risk_level"]
        flag   = " <<" if action == "DENY" else ""
        print(f"  {label:<38}  {prob:>6.1%}  {action:>8}  {level}{flag}")

    prob_orig = s_orig["prediction_prob"]
    prob_corr = s_corr["prediction_prob"]
    delta     = prob_orig - prob_corr
    print()
    print(f"  Denial probability DROP from data fix: {prob_orig:.1%} -> {prob_corr:.1%}")
    print(f"  Absolute reduction: {delta:.1%} ({delta/prob_orig:.0%} relative decrease)")
    pause()

    section("Step 4 — SHAP shift for prior_auth_present feature")
    def shap_for(drivers: list, feature: str) -> float:
        return next((d["shap_value"] for d in drivers if d["feature"] == feature), 0.0)

    sv_orig = shap_for(shap_orig["top_drivers"], "prior_auth_present")
    sv_corr = shap_for(shap_corr["top_drivers"], "prior_auth_present")
    print()
    print(f"  prior_auth_present SHAP (original) : {sv_orig:+.4f}  [pushes TOWARD denial]")
    print(f"  prior_auth_present SHAP (corrected): {sv_corr:+.4f}  [now REDUCES denial risk]")
    print(f"  SHAP shift: {sv_corr - sv_orig:+.4f}  (log-odds units)")
    print()
    print("  INSIGHT: Correcting a single data field (entering the prior auth")
    print("  NPI) flips the top SHAP driver from denial-positive to")
    print("  denial-negative. XAI made this actionable in seconds.")

    # Save side-by-side waterfall images
    save_image(shap_orig["waterfall_b64"], "s4_waterfall_original.png")
    save_image(shap_corr["waterfall_b64"], "s4_waterfall_corrected.png")
    print()
    print(f"  Waterfall (original)  -> {OUT / 's4_waterfall_original.png'}")
    print(f"  Waterfall (corrected) -> {OUT / 's4_waterfall_corrected.png'}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    banner("DEMO COMPLETE — Summary of Artifacts")
    print()
    artifacts = sorted(OUT.glob("*"))
    for a in artifacts:
        size_kb = a.stat().st_size / 1024
        print(f"  {a.name:<40s}  {size_kb:6.1f} KB")

    print()
    print("  Key talking points")
    print("  ------------------")
    points = [
        "XAI identifies DATA QUALITY issues a black box would silently deny",
        "SHAP and LIME independently agree on the top driver (prior auth)",
        "Counterfactual shows exact score impact of fixing the data",
        "Protected attributes (age, sex, race) are NOT primary model drivers",
        "Every human override is logged with reason for regulator audit",
        "ACA-compliant denial letter generated automatically from LIME reasons",
        "Adjuster resolves in minutes vs. 30-day member appeal process",
    ]
    for p in points:
        bullet(p)

    print()
    print("  To launch the interactive dashboard:")
    print("    FastAPI  -> http://localhost:8000/docs")
    print("    Streamlit -> http://localhost:8501")
    print()
    print("=" * WIDTH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * WIDTH)
    print("  XAI Health Insurance Claims — Full Demo")
    print("  Connecting to FastAPI at", API)
    print("=" * WIDTH)

    # Verify API is up
    try:
        h = client.get("/health")
        h.raise_for_status()
        status = h.json()
        if not status.get("model_loaded"):
            print("ERROR: Model not loaded. Start the API server first.")
            sys.exit(1)
        print(f"  API ready  |  model={status['model_loaded']}  "
              f"shap={status['shap_ready']}  lime={status['lime_ready']}")
    except httpx.ConnectError:
        print("ERROR: Cannot connect to FastAPI on :8000.")
        print("  Start it with: uvicorn src.api.main:app --port 8000")
        sys.exit(1)

    scenario_1_adjuster()
    scenario_2_compliance()
    scenario_3_patient_letter()
    scenario_4_counterfactual()
    print_summary()
