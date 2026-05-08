"""
Streamlit adjuster dashboard — calls FastAPI on :8000 via HTTPX.

Run (FastAPI must be running first):
  .venv-api\\Scripts\\activate
  streamlit run src/dashboard/app.py
"""

import base64
from datetime import date

import httpx
import pandas as pd
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
# Page config + compact CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="XAI Health Claims",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Tighter top padding */
.block-container { padding-top: 0.6rem !important; padding-bottom: 0.5rem !important; }

/* Shrink number/select input vertical rhythm */
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"]  { margin-bottom: -10px !important; }

/* Smaller label text */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"]  label,
div[data-testid="stCheckbox"]   label { font-size: 0.78rem !important; }

/* Section mini-headers */
.sec-hdr {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #888;
    margin: 6px 0 2px 0;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 2px;
}

/* Score badge */
.badge-deny    { background:#d73027; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; }
.badge-approve { background:#1a9850; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; }

/* Tighten tab bar */
button[data-baseweb="tab"] { padding: 6px 14px !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{endpoint}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("Cannot reach FastAPI on :8000 — start the server first.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:120]}")
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
        st.error(f"API {e.response.status_code}: {e.response.text[:120]}")
        return None


def _img(b64: str) -> bytes:
    return base64.b64decode(b64)


def _sec(label: str) -> None:
    st.markdown(f'<div class="sec-hdr">{label}</div>', unsafe_allow_html=True)


def _prob_bar(prob: float) -> None:
    color = "#d73027" if prob > 0.5 else "#1a9850"
    pct   = int(prob * 100)
    st.markdown(
        f'<div style="background:#eee;border-radius:5px;height:18px;width:100%;margin:4px 0">'
        f'<div style="background:{color};width:{pct}%;height:18px;border-radius:5px;'
        f'display:flex;align-items:center;padding-left:7px;color:#fff;font-size:11px">'
        f'{pct}%</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### XAI Health Claims")
    adjuster_name = st.text_input("Adjuster", value="Demo Adjuster", label_visibility="visible")
    st.divider()
    st.markdown("**Demo Scenario**")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), label_visibility="collapsed")
    preset = PRESETS[preset_name]
    st.divider()
    st.markdown("**Claim Metadata**")
    claim_id     = st.text_input("Claim ID",     value="CLM-0000000177")
    member_id    = st.text_input("Member ID",    value="MBR-0000177")
    service_date = st.text_input("Service date", value=str(date.today()))
    provider     = st.text_input("Provider",     value="General Hospital")


# ---------------------------------------------------------------------------
# Main header + top-level tabs
# ---------------------------------------------------------------------------

st.markdown(
    "#### 🏥 XAI Health Claims — Adjuster Dashboard "
    '<span style="font-size:0.8rem;color:#888">| XGBoost + SHAP + LIME</span>',
    unsafe_allow_html=True,
)

tab_review, tab_global, tab_audit = st.tabs(["Claim Review", "Global SHAP Analysis", "Audit Log"])


# ===========================================================================
# TAB 1 — CLAIM REVIEW
# ===========================================================================

with tab_review:
    form_col, result_col = st.columns([4, 6], gap="medium")

    # ── Left: compact claim input form ──────────────────────────────────────
    with form_col:

        # ── Claim Info ──────────────────────────────────────────────────────
        _sec("Claim Info")
        fa, fb, fc = st.columns(3)
        is_inpatient    = fa.selectbox("Type", [0, 1], index=preset["is_inpatient"],
                                       format_func=lambda x: "OP" if x == 0 else "IP",
                                       label_visibility="visible")
        claim_year_val  = fb.number_input("Year",  value=preset["claim_year"],  min_value=2000, max_value=2025)
        claim_month_val = fc.number_input("Month", value=preset["claim_month"], min_value=1,    max_value=12)

        ga, gb = st.columns(2)
        clm_pmt_amt       = ga.number_input("Payment ($)",      value=float(preset["clm_pmt_amt"]))
        primary_payer_amt = gb.number_input("Primary payer ($)", value=float(preset["primary_payer_amt"]), min_value=0.0)

        ha, hb = st.columns(2)
        has_primary_payer = ha.selectbox("Primary payer",  [0, 1], index=preset["has_primary_payer"],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
        is_adjustment     = hb.selectbox("Adjustment",     [0, 1], index=preset["is_adjustment"],
                                          format_func=lambda x: "No" if x == 0 else "Yes")

        # ── Auth & Coding ───────────────────────────────────────────────────
        _sec("Auth & Coding")
        ia, ib = st.columns(2)
        prior_auth     = ia.selectbox("Prior auth",   [0, 1], index=preset["prior_auth_present"],
                                       format_func=lambda x: "Missing" if x == 0 else "Present")
        has_op_surgeon = ib.selectbox("Op surgeon NPI", [0, 1], index=preset["has_op_surgeon"],
                                       format_func=lambda x: "Missing" if x == 0 else "Present")

        ja, jb, jc = st.columns(3)
        icd9_raw     = preset.get("primary_icd9_num")
        primary_icd9 = ja.number_input("ICD-9",   value=float(icd9_raw) if icd9_raw else 0.0)
        num_diagnoses   = jb.number_input("Dx codes", value=preset["num_diagnoses"],  min_value=0)
        num_procedures  = jc.number_input("Proc codes", value=preset["num_procedures"], min_value=0)

        ka, kb, kc, kd = st.columns(4)
        has_procedure   = ka.selectbox("Proc?",  [0, 1], index=preset["has_procedure"],
                                        format_func=lambda x: "N" if x == 0 else "Y")
        has_hcpcs       = kb.selectbox("HCPCS?", [0, 1], index=preset["has_hcpcs"],
                                        format_func=lambda x: "N" if x == 0 else "Y")
        drg_raw         = preset.get("drg_num")
        drg_num_val     = kc.number_input("DRG", value=float(drg_raw) if drg_raw else 0.0, min_value=0.0)
        utlz_raw        = preset.get("utlztn_days")
        utlztn_days_val = kd.number_input("Util days", value=float(utlz_raw) if utlz_raw else 0.0, min_value=0.0)

        los_days = st.number_input("Length of stay (days)", value=float(preset["los_days"]), min_value=0.0)

        # ── Beneficiary ─────────────────────────────────────────────────────
        _sec("Beneficiary")
        la, lb, lc, ld = st.columns(4)
        bene_age  = la.number_input("Age",  value=preset["bene_age"],  min_value=0, max_value=120)
        bene_sex  = lb.selectbox("Sex", [0, 1, 2], index=preset["bene_sex"],
                                  format_func=lambda x: {0:"Unk",1:"M",2:"F"}[x])
        bene_race = lc.number_input("Race", value=preset["bene_race"], min_value=0)
        bene_esrd = ld.selectbox("ESRD", [0, 1], index=preset["bene_esrd"],
                                  format_func=lambda x: "N" if x == 0 else "Y")

        ma, mb, mc = st.columns(3)
        bene_hi_mons  = ma.number_input("HI mons",  value=preset["bene_hi_mons"],  min_value=0, max_value=12)
        bene_hmo_mons = mb.number_input("HMO mons", value=preset["bene_hmo_mons"], min_value=0, max_value=12)
        bene_deceased = mc.selectbox("Deceased", [0, 1], index=preset["bene_is_deceased"],
                                      format_func=lambda x: "N" if x == 0 else "Y")

        # ── Chronic conditions ───────────────────────────────────────────────
        _sec("Chronic Conditions")
        CONDS = [
            ("has_alzheimer", "Alzheimer"), ("has_chf", "CHF"),    ("has_ckd", "CKD"),
            ("has_cancer",    "Cancer"),    ("has_copd", "COPD"),   ("has_depression", "Depression"),
            ("has_diabetes",  "Diabetes"),  ("has_ihd", "IHD"),     ("has_osteoporosis", "Osteo"),
            ("has_ra_oa",     "RA/OA"),     ("has_stroke", "Stroke"),
        ]
        cond_vals: dict[str, int] = {}
        cols_cc = st.columns(4)
        for i, (key, label) in enumerate(CONDS):
            cond_vals[key] = 1 if cols_cc[i % 4].checkbox(label, value=bool(preset[key]), key=key) else 0
        chronic_count = sum(cond_vals.values())
        st.caption(f"Chronic count: **{chronic_count}**")

        # ── Score button ─────────────────────────────────────────────────────
        st.markdown("")
        score_clicked = st.button("Score Claim", type="primary", use_container_width=True)

    # ── Payload assembly ─────────────────────────────────────────────────────
    payload = {
        "is_inpatient":       is_inpatient,
        "los_days":           los_days,
        "claim_year":         int(claim_year_val),
        "claim_month":        int(claim_month_val),
        "clm_pmt_amt":        clm_pmt_amt,
        "primary_payer_amt":  primary_payer_amt,
        "has_primary_payer":  has_primary_payer,
        "is_adjustment":      is_adjustment,
        "prior_auth_present": prior_auth,
        "has_op_surgeon":     has_op_surgeon,
        "primary_icd9_num":   primary_icd9 if primary_icd9 > 0 else None,
        "num_diagnoses":      int(num_diagnoses),
        "num_procedures":     int(num_procedures),
        "has_procedure":      has_procedure,
        "has_hcpcs":          has_hcpcs,
        "drg_num":            drg_num_val if drg_num_val > 0 else None,
        "utlztn_days":        utlztn_days_val if utlztn_days_val > 0 else None,
        "bene_age":           int(bene_age),
        "bene_sex":           bene_sex,
        "bene_race":          int(bene_race),
        "bene_esrd":          bene_esrd,
        "bene_hi_mons":       int(bene_hi_mons),
        "bene_hmo_mons":      int(bene_hmo_mons),
        "bene_is_deceased":   bene_deceased,
        **cond_vals,
        "chronic_count":      chronic_count,
        "claim_id":           claim_id,
        "member_id":          member_id,
        "service_date":       service_date,
        "provider":           provider,
        "adjuster_name":      adjuster_name,
    }

    if score_clicked:
        with st.spinner("Scoring…"):
            sr = _post("/claims/score",   payload)
            sh = _post("/explain/shap",   payload)
            li = _post("/explain/lime",   payload)
            rp = _post("/explain/report", payload)
        if sr:
            st.session_state.update({"score": sr, "shap": sh, "lime": li,
                                      "report": rp, "payload": payload})

    # ── Right: results ────────────────────────────────────────────────────────
    with result_col:
        if "score" not in st.session_state:
            st.markdown(
                '<div style="margin-top:120px;text-align:center;color:#aaa;font-size:1rem">'
                'Select a demo scenario and click <b>Score Claim</b></div>',
                unsafe_allow_html=True,
            )
        else:
            sd   = st.session_state["score"]
            shd  = st.session_state.get("shap")
            lmd  = st.session_state.get("lime")
            rpd  = st.session_state.get("report")
            pay  = st.session_state.get("payload", payload)

            prob   = sd["prediction_prob"]
            action = sd["recommended_action"]
            level  = sd["denial_risk_level"]
            badge  = "badge-deny" if action == "DENY" else "badge-approve"

            # Score card
            sc1, sc2, sc3, sc4 = st.columns([2, 2, 2, 3])
            sc1.metric("Denial Prob", f"{prob:.1%}")
            sc2.metric("Risk", level)
            sc3.metric("Action", action)
            with sc4:
                st.markdown(f'<div style="margin-top:28px"><span class="{badge}">{action}</span></div>',
                            unsafe_allow_html=True)
            _prob_bar(prob)

            st.markdown("")

            # Result sub-tabs
            rt1, rt2, rt3, rt4, rt5 = st.tabs(["SHAP", "LIME", "Denial Letter", "Adj Summary", "Override"])

            with rt1:
                if shd:
                    r_left, r_right = st.columns([5, 3])
                    with r_left:
                        st.image(_img(shd["waterfall_b64"]), use_container_width=True)
                    with r_right:
                        st.markdown("**Top drivers**")
                        for d in shd["top_drivers"][:8]:
                            sign  = "+" if d["shap_value"] > 0 else ""
                            color = "red" if d["shap_value"] > 0 else "green"
                            st.markdown(
                                f":{color}[{sign}{d['shap_value']:.3f}] **{d['feature']}**"
                            )
                        if shd.get("counterfactual_hints"):
                            st.markdown("**Counterfactual**")
                            for h in shd["counterfactual_hints"][:2]:
                                clean = h.split("]", 1)[-1].strip() if "]" in h else h
                                st.info(clean, icon="💡")

            with rt2:
                if lmd:
                    r_left, r_right = st.columns([5, 3])
                    with r_left:
                        st.image(_img(lmd["bar_b64"]), use_container_width=True)
                    with r_right:
                        st.markdown("**Top reasons**")
                        for r in lmd["top_reasons"][:6]:
                            sign  = "+" if r["coeff"] > 0 else ""
                            color = "red" if r["coeff"] > 0 else "green"
                            st.markdown(
                                f":{color}[{sign}{r['coeff']:.3f}] **{r['label']}**"
                            )

            with rt3:
                if rpd:
                    st.text_area("Member Denial Letter", value=rpd["member_letter"],
                                 height=380, label_visibility="collapsed")
                    st.download_button("Download (.txt)",
                                       data=rpd["member_letter"].encode(),
                                       file_name=f"denial_letter_{pay.get('claim_id','CLM')}.txt",
                                       mime="text/plain")

            with rt4:
                if rpd:
                    st.text_area("Adjuster Summary", value=rpd["adjuster_summary"],
                                 height=380, label_visibility="collapsed")
                    st.download_button("Download (.txt)",
                                       data=rpd["adjuster_summary"].encode(),
                                       file_name=f"adjuster_summary_{pay.get('claim_id','CLM')}.txt",
                                       mime="text/plain")

            with rt5:
                with st.form("override_form"):
                    ov_decision = st.radio("Decision", ["APPROVE", "DENY"],
                                           index=0 if action == "DENY" else 1,
                                           horizontal=True)
                    ov_reason   = st.text_area("Reason (required)", height=100,
                                               placeholder="Clinical review confirmed…")
                    submitted   = st.form_submit_button("Submit Override", type="primary")

                if submitted:
                    if not ov_reason or len(ov_reason) < 5:
                        st.warning("Enter a reason (min 5 characters).")
                    else:
                        res = _post("/audit/override", {
                            "claim_id":           pay.get("claim_id", "CLM-UNKNOWN"),
                            "member_id":          pay.get("member_id", "MBR-UNKNOWN"),
                            "adjuster_name":      adjuster_name,
                            "original_prob":      prob,
                            "recommended_action": action,
                            "override_decision":  ov_decision,
                            "reason":             ov_reason,
                        })
                        if res:
                            st.success(
                                f"Logged (audit ID: **{res['audit_id']}**) — "
                                f"Final: **{ov_decision}**"
                            )


# ===========================================================================
# TAB 2 — GLOBAL SHAP
# ===========================================================================

with tab_global:
    if st.button("Load Global SHAP", type="primary"):
        st.session_state["global_shap"] = _get("/explain/shap/global")

    if "global_shap" in st.session_state:
        gd = st.session_state["global_shap"]
        if gd:
            img_col, tbl_col = st.columns([6, 3])
            with img_col:
                gt1, gt2 = st.tabs(["Beeswarm", "Bar Summary"])
                with gt1:
                    st.image(_img(gd["beeswarm_b64"]), use_container_width=True)
                with gt2:
                    st.image(_img(gd["bar_b64"]), use_container_width=True)
            with tbl_col:
                st.markdown("**Top-20 features by mean |SHAP|**")
                fi_items = list(gd["feature_importance"].items())[:20]
                fi_df = pd.DataFrame(fi_items, columns=["Feature", "Mean |SHAP|"])
                fi_df.index = fi_df.index + 1
                fi_df["Mean |SHAP|"] = fi_df["Mean |SHAP|"].map(lambda x: f"{x:.5f}")
                protected = {"bene_age", "bene_sex", "bene_race"}
                fi_df["Protected"] = fi_df["Feature"].apply(lambda x: "Yes" if x in protected else "")
                st.dataframe(fi_df, use_container_width=True, height=560)
    else:
        st.markdown(
            '<div style="margin-top:80px;text-align:center;color:#aaa">'
            'Click <b>Load Global SHAP</b> to fetch the feature importance summary.</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# TAB 3 — AUDIT LOG
# ===========================================================================

with tab_audit:
    al1, al2 = st.columns([2, 8])
    with al1:
        refresh = st.button("Refresh Log", type="primary")
    if refresh or "audit_log" not in st.session_state:
        st.session_state["audit_log"] = _get("/audit/log?page_size=100")

    log = st.session_state.get("audit_log")
    if log and log["total"] > 0:
        st.caption(f"Total entries: **{log['total']}**")
        df_log = pd.DataFrame(log["entries"])
        # Show concise columns first, put long reason last
        cols_order = ["audit_id", "timestamp", "claim_id", "member_id", "adjuster_name",
                      "model_prob", "model_decision", "final_decision", "override_reason"]
        df_log = df_log[[c for c in cols_order if c in df_log.columns]]
        st.dataframe(df_log, use_container_width=True, height=500)
    else:
        st.markdown(
            '<div style="margin-top:80px;text-align:center;color:#aaa">'
            'No audit entries yet. Submit an override from the Claim Review tab.</div>',
            unsafe_allow_html=True,
        )
