"""
Streamlit adjuster dashboard — calls FastAPI on :8000 via HTTPX.

Run (FastAPI must be running first):
  .venv-api\\Scripts\\activate
  streamlit run src/dashboard/app.py
"""

import base64
from datetime import date

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Demo claim presets
# ---------------------------------------------------------------------------

PRESETS = {
    "Denied: Missing Prior Auth (Key Demo)": {
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
    },
    "Approved: Well-Documented Claim": {
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
    },
    "Inpatient: High-Cost Claim": {
        "is_inpatient": 1, "los_days": 7.0, "claim_year": 2009, "claim_month": 3,
        "clm_pmt_amt": 12000.0, "primary_payer_amt": 0.0, "has_primary_payer": 0,
        "is_adjustment": 0, "prior_auth_present": 0, "has_op_surgeon": 1,
        "primary_icd9_num": 410.0, "num_diagnoses": 8, "num_procedures": 3,
        "has_procedure": 1, "has_hcpcs": 0, "drg_num": 280.0, "utlztn_days": 7.0,
        "bene_age": 80, "bene_sex": 2, "bene_race": 1, "bene_esrd": 0,
        "bene_hi_mons": 12, "bene_hmo_mons": 0, "bene_is_deceased": 0,
        "has_alzheimer": 0, "has_chf": 1, "has_ckd": 1, "has_cancer": 0,
        "has_copd": 0, "has_depression": 1, "has_diabetes": 1, "has_ihd": 1,
        "has_osteoporosis": 1, "has_ra_oa": 0, "has_stroke": 0, "chronic_count": 6,
    },
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{endpoint}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("Cannot reach FastAPI on :8000. Start the server first.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None


def _get(endpoint: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{endpoint}", timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("Cannot reach FastAPI on :8000.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None


def _b64_image(b64_str: str):
    return base64.b64decode(b64_str)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _risk_badge(action: str, prob: float) -> str:
    color = "#d73027" if action == "DENY" else "#1a9850"
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:4px;font-weight:bold">{action}</span>'


def _prob_bar(prob: float):
    color = "#d73027" if prob > 0.5 else "#1a9850"
    pct   = int(prob * 100)
    st.markdown(
        f"""
        <div style="background:#eee;border-radius:6px;height:22px;width:100%">
          <div style="background:{color};width:{pct}%;height:22px;border-radius:6px;
                      display:flex;align-items:center;padding-left:8px;color:white;font-size:13px">
            {pct}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "XAI Health Claims",
    page_icon  = "🏥",
    layout     = "wide",
)

st.title("XAI Health Claims — Adjuster Dashboard")
st.caption("Explainable AI for health insurance claim review | Powered by XGBoost + SHAP + LIME")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    adjuster_name = st.text_input("Adjuster name", value="Demo Adjuster")

    st.header("Claim Preset")
    preset_name = st.selectbox("Select demo scenario", list(PRESETS.keys()))
    preset      = PRESETS[preset_name]

    st.header("Claim Metadata")
    claim_id     = st.text_input("Claim ID",    value="CLM-0000000177")
    member_id    = st.text_input("Member ID",   value="MBR-0000177")
    service_date = st.text_input("Service date", value=str(date.today()))
    provider     = st.text_input("Provider",    value="General Hospital")

    st.divider()
    if st.button("View Global SHAP"):
        st.session_state["show_global"] = True

# ── Global SHAP modal ────────────────────────────────────────────────────────
if st.session_state.get("show_global"):
    with st.expander("Global SHAP Feature Importance", expanded=True):
        data = _get("/explain/shap/global")
        if data:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Beeswarm")
                st.image(_b64_image(data["beeswarm_b64"]), use_container_width=True)
            with col2:
                st.subheader("Bar Summary")
                st.image(_b64_image(data["bar_b64"]), use_container_width=True)
            st.subheader("Top-10 features by mean |SHAP|")
            items = list(data["feature_importance"].items())[:10]
            for rank, (feat, val) in enumerate(items, 1):
                st.text(f"  {rank:2d}. {feat:<28s} {val:.5f}")
        if st.button("Close"):
            st.session_state["show_global"] = False
            st.rerun()

# ── Claim Input Form ─────────────────────────────────────────────────────────
with st.expander("Claim Feature Inputs", expanded=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Claim Info")
        is_inpatient    = st.selectbox("Claim type",         [0, 1], index=preset["is_inpatient"],
                                        format_func=lambda x: "Outpatient" if x == 0 else "Inpatient")
        los_days        = st.number_input("Length of stay (days)", value=float(preset["los_days"]), min_value=0.0)
        claim_year_val  = st.number_input("Claim year",  value=preset["claim_year"],  min_value=2000, max_value=2025)
        claim_month_val = st.number_input("Claim month", value=preset["claim_month"], min_value=1,   max_value=12)
        clm_pmt_amt     = st.number_input("Claim payment amount ($)", value=float(preset["clm_pmt_amt"]))
        primary_payer_amt = st.number_input("Primary payer amount ($)", value=float(preset["primary_payer_amt"]), min_value=0.0)
        has_primary_payer = st.selectbox("Primary payer present", [0, 1], index=preset["has_primary_payer"],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
        is_adjustment   = st.selectbox("Adjustment claim",   [0, 1], index=preset["is_adjustment"],
                                        format_func=lambda x: "No" if x == 0 else "Yes")

    with c2:
        st.subheader("Auth & Coding")
        prior_auth      = st.selectbox("Prior auth on file",  [0, 1], index=preset["prior_auth_present"],
                                        format_func=lambda x: "Missing" if x == 0 else "Present")
        has_op_surgeon  = st.selectbox("Op surgeon NPI",     [0, 1], index=preset["has_op_surgeon"],
                                        format_func=lambda x: "Missing" if x == 0 else "Present")
        icd9_raw        = preset.get("primary_icd9_num")
        primary_icd9    = st.number_input("Primary ICD-9 (numeric)", value=float(icd9_raw) if icd9_raw else 0.0)
        num_diagnoses   = st.number_input("Num diagnosis codes", value=preset["num_diagnoses"], min_value=0)
        num_procedures  = st.number_input("Num procedure codes", value=preset["num_procedures"], min_value=0)
        has_procedure   = st.selectbox("Procedure code present", [0, 1], index=preset["has_procedure"],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
        has_hcpcs       = st.selectbox("HCPCS code present", [0, 1], index=preset["has_hcpcs"],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
        drg_raw         = preset.get("drg_num")
        drg_num_val     = st.number_input("DRG (leave 0 for N/A)", value=float(drg_raw) if drg_raw else 0.0, min_value=0.0)
        utlz_raw        = preset.get("utlztn_days")
        utlztn_days_val = st.number_input("Utilization days (0 = N/A)", value=float(utlz_raw) if utlz_raw else 0.0, min_value=0.0)

    with c3:
        st.subheader("Beneficiary")
        bene_age      = st.number_input("Age",    value=preset["bene_age"], min_value=0, max_value=120)
        bene_sex      = st.selectbox("Sex",      [0, 1, 2], index=preset["bene_sex"],
                                      format_func=lambda x: {0:"Unknown",1:"Male",2:"Female"}[x])
        bene_race     = st.number_input("Race code", value=preset["bene_race"], min_value=0)
        bene_esrd     = st.selectbox("ESRD",     [0, 1], index=preset["bene_esrd"],
                                      format_func=lambda x: "No" if x == 0 else "Yes")
        bene_hi_mons  = st.number_input("HI enrollment months", value=preset["bene_hi_mons"], min_value=0, max_value=12)
        bene_hmo_mons = st.number_input("HMO months",           value=preset["bene_hmo_mons"], min_value=0, max_value=12)
        bene_deceased = st.selectbox("Deceased", [0, 1], index=preset["bene_is_deceased"],
                                      format_func=lambda x: "No" if x == 0 else "Yes")

        st.subheader("Chronic Conditions")
        conds = {
            "has_alzheimer": "Alzheimer's", "has_chf": "CHF",
            "has_ckd": "CKD", "has_cancer": "Cancer", "has_copd": "COPD",
            "has_depression": "Depression", "has_diabetes": "Diabetes",
            "has_ihd": "IHD", "has_osteoporosis": "Osteoporosis",
            "has_ra_oa": "RA/OA", "has_stroke": "Stroke/TIA",
        }
        cond_vals = {}
        for key, label in conds.items():
            cond_vals[key] = 1 if st.checkbox(label, value=bool(preset[key])) else 0
        chronic_count = sum(cond_vals.values())
        st.caption(f"Chronic count: {chronic_count}")

# ── Payload assembly ─────────────────────────────────────────────────────────
payload = {
    "is_inpatient":      is_inpatient,
    "los_days":          los_days,
    "claim_year":        int(claim_year_val),
    "claim_month":       int(claim_month_val),
    "clm_pmt_amt":       clm_pmt_amt,
    "primary_payer_amt": primary_payer_amt,
    "has_primary_payer": has_primary_payer,
    "is_adjustment":     is_adjustment,
    "prior_auth_present": prior_auth,
    "has_op_surgeon":    has_op_surgeon,
    "primary_icd9_num":  primary_icd9 if primary_icd9 > 0 else None,
    "num_diagnoses":     int(num_diagnoses),
    "num_procedures":    int(num_procedures),
    "has_procedure":     has_procedure,
    "has_hcpcs":         has_hcpcs,
    "drg_num":           drg_num_val if drg_num_val > 0 else None,
    "utlztn_days":       utlztn_days_val if utlztn_days_val > 0 else None,
    "bene_age":          int(bene_age),
    "bene_sex":          bene_sex,
    "bene_race":         int(bene_race),
    "bene_esrd":         bene_esrd,
    "bene_hi_mons":      int(bene_hi_mons),
    "bene_hmo_mons":     int(bene_hmo_mons),
    "bene_is_deceased":  bene_deceased,
    **cond_vals,
    "chronic_count":     chronic_count,
    "claim_id":          claim_id,
    "member_id":         member_id,
    "service_date":      service_date,
    "provider":          provider,
    "adjuster_name":     adjuster_name,
}

# ── Score button ─────────────────────────────────────────────────────────────
st.divider()
col_btn, col_pad = st.columns([2, 8])
with col_btn:
    score_clicked = st.button("Score Claim", type="primary", use_container_width=True)

if score_clicked:
    with st.spinner("Scoring and computing explanations..."):
        score_resp = _post("/claims/score", payload)
        shap_resp  = _post("/explain/shap",  payload)
        lime_resp  = _post("/explain/lime",  payload)
        rpt_resp   = _post("/explain/report", payload)

    if score_resp:
        st.session_state.update({
            "score": score_resp,
            "shap":  shap_resp,
            "lime":  lime_resp,
            "report": rpt_resp,
            "payload": payload,
        })

# ── Results ───────────────────────────────────────────────────────────────────
if "score" in st.session_state:
    score_data = st.session_state["score"]
    shap_data  = st.session_state.get("shap")
    lime_data  = st.session_state.get("lime")
    rpt_data   = st.session_state.get("report")
    pay        = st.session_state.get("payload", payload)

    prob   = score_data["prediction_prob"]
    action = score_data["recommended_action"]
    level  = score_data["denial_risk_level"]

    st.divider()
    st.subheader("Claim Decision")
    m1, m2, m3 = st.columns(3)
    m1.metric("Denial Probability", f"{prob:.1%}")
    m2.metric("Risk Level", level)
    m3.metric("Recommended Action", action)
    _prob_bar(prob)
    st.markdown(_risk_badge(action, prob), unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["SHAP Explanation", "LIME Explanation", "Denial Letter", "Adjuster Summary"])

    with tab1:
        if shap_data:
            st.subheader("SHAP Waterfall — Why was this claim scored this way?")
            st.image(_b64_image(shap_data["waterfall_b64"]), use_container_width=True)
            st.subheader("Top SHAP Drivers")
            for d in shap_data["top_drivers"][:10]:
                sign  = "+" if d["shap_value"] > 0 else ""
                color = "red" if d["shap_value"] > 0 else "green"
                st.markdown(
                    f"**{d['feature']}**: :{color}[{sign}{d['shap_value']:.4f}] — {d['direction'].replace('_',' ')}"
                )
            if shap_data.get("counterfactual_hints"):
                st.subheader("Counterfactual Guidance")
                for h in shap_data["counterfactual_hints"]:
                    clean = h.split("]", 1)[-1].strip() if "]" in h else h
                    st.info(clean)

    with tab2:
        if lime_data:
            st.subheader("LIME Local Explanation — Feature contributions (linear approximation)")
            st.image(_b64_image(lime_data["bar_b64"]), use_container_width=True)
            st.subheader("Top Reasons")
            for r in lime_data["top_reasons"]:
                sign  = "+" if r["coeff"] > 0 else ""
                color = "red" if r["coeff"] > 0 else "green"
                st.markdown(
                    f"**{r['label']}** :{color}[{sign}{r['coeff']:.4f}] — `{r['condition']}`"
                )

    with tab3:
        if rpt_data:
            st.subheader("Member Denial Letter")
            st.text(rpt_data["member_letter"])
            st.download_button(
                "Download Letter (.txt)",
                data=rpt_data["member_letter"].encode(),
                file_name=f"denial_letter_{pay.get('claim_id','CLM')}.txt",
                mime="text/plain",
            )

    with tab4:
        if rpt_data:
            st.subheader("Adjuster Summary")
            st.text(rpt_data["adjuster_summary"])
            st.download_button(
                "Download Summary (.txt)",
                data=rpt_data["adjuster_summary"].encode(),
                file_name=f"adjuster_summary_{pay.get('claim_id','CLM')}.txt",
                mime="text/plain",
            )

    # ── Override panel ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Adjuster Override")
    with st.form("override_form"):
        override_decision = st.radio(
            "Override decision",
            ["APPROVE", "DENY"],
            index=0 if action == "DENY" else 1,
            horizontal=True,
        )
        override_reason = st.text_area(
            "Override reason (required)",
            placeholder="Clinical review confirmed medical necessity. Patient has documented chronic CHF...",
        )
        submitted = st.form_submit_button("Submit Override")

    if submitted:
        if not override_reason or len(override_reason) < 5:
            st.warning("Please enter an override reason (at least 5 characters).")
        else:
            ov_payload = {
                "claim_id":           pay.get("claim_id", "CLM-UNKNOWN"),
                "member_id":          pay.get("member_id", "MBR-UNKNOWN"),
                "adjuster_name":      adjuster_name,
                "original_prob":      prob,
                "recommended_action": action,
                "override_decision":  override_decision,
                "reason":             override_reason,
            }
            result = _post("/audit/override", ov_payload)
            if result:
                st.success(
                    f"Override logged (audit ID: {result['audit_id']}) — "
                    f"Final decision: **{override_decision}**"
                )

    # ── Audit log ─────────────────────────────────────────────────────────────
    with st.expander("Audit Log"):
        log = _get("/audit/log?page_size=50")
        if log and log["total"] > 0:
            st.caption(f"Total entries: {log['total']}")
            import pandas as pd
            df_log = pd.DataFrame(log["entries"])
            st.dataframe(df_log, use_container_width=True)
        else:
            st.caption("No audit entries yet.")
