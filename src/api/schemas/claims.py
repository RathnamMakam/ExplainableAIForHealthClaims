"""Pydantic request/response schemas for the XAI Health Claims API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ClaimInput(BaseModel):
    # Claim-level features
    is_inpatient:       int   = Field(0,    ge=0, le=1)
    los_days:           float = Field(0.0,  ge=0)
    claim_year:         int   = Field(2008)
    claim_month:        int   = Field(1, ge=1, le=12)
    clm_pmt_amt:        float = Field(0.0)
    primary_payer_amt:  float = Field(0.0)
    has_primary_payer:  int   = Field(0, ge=0, le=1)
    is_adjustment:      int   = Field(0, ge=0, le=1)
    prior_auth_present: int   = Field(0, ge=0, le=1)
    has_op_surgeon:     int   = Field(0, ge=0, le=1)
    # Diagnosis / procedure
    primary_icd9_num:   Optional[float] = Field(None)
    num_diagnoses:      int   = Field(1, ge=0)
    num_procedures:     int   = Field(0, ge=0)
    has_procedure:      int   = Field(0, ge=0, le=1)
    has_hcpcs:          int   = Field(0, ge=0, le=1)
    drg_num:            Optional[float] = Field(None)
    utlztn_days:        Optional[float] = Field(None)
    # Beneficiary demographics
    bene_age:           int   = Field(70, ge=0, le=120)
    bene_sex:           int   = Field(0)
    bene_race:          int   = Field(0)
    bene_esrd:          int   = Field(0, ge=0, le=1)
    bene_hi_mons:       int   = Field(12, ge=0, le=12)
    bene_hmo_mons:      int   = Field(0,  ge=0, le=12)
    bene_is_deceased:   int   = Field(0, ge=0, le=1)
    # Chronic conditions
    has_alzheimer:      int   = Field(0, ge=0, le=1)
    has_chf:            int   = Field(0, ge=0, le=1)
    has_ckd:            int   = Field(0, ge=0, le=1)
    has_cancer:         int   = Field(0, ge=0, le=1)
    has_copd:           int   = Field(0, ge=0, le=1)
    has_depression:     int   = Field(0, ge=0, le=1)
    has_diabetes:       int   = Field(0, ge=0, le=1)
    has_ihd:            int   = Field(0, ge=0, le=1)
    has_osteoporosis:   int   = Field(0, ge=0, le=1)
    has_ra_oa:          int   = Field(0, ge=0, le=1)
    has_stroke:         int   = Field(0, ge=0, le=1)
    chronic_count:      int   = Field(0, ge=0)
    # Optional metadata (passed through to reports / audit)
    claim_id:           str   = Field("CLM-UNKNOWN")
    member_id:          str   = Field("MBR-UNKNOWN")
    service_date:       str   = Field("")
    provider:           str   = Field("")
    adjuster_name:      str   = Field("")


class OverrideRequest(BaseModel):
    claim_id:           str
    member_id:          str   = ""
    adjuster_name:      str   = ""
    original_prob:      float
    recommended_action: str
    override_decision:  str   = Field(..., pattern="^(APPROVE|DENY)$")
    reason:             str   = Field(..., min_length=5)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class TopDriver(BaseModel):
    feature:   str
    shap_value: float
    direction: str


class TopReason(BaseModel):
    feature:   str
    label:     str
    coeff:     float
    direction: str
    condition: str


class ScoreResponse(BaseModel):
    claim_id:           str
    prediction_prob:    float
    recommended_action: str
    denial_risk_level:  str


class SHAPLocalResponse(BaseModel):
    claim_id:           str
    base_value:         float
    prediction_prob:    float
    shap_values:        dict[str, float]
    top_drivers:        list[TopDriver]
    waterfall_b64:      str
    counterfactual_hints: list[str]


class LIMELocalResponse(BaseModel):
    claim_id:        str
    prediction_prob: float
    top_reasons:     list[TopReason]
    intercept:       float
    local_pred:      float
    bar_b64:         str


class GlobalSHAPResponse(BaseModel):
    feature_importance: dict[str, float]
    beeswarm_b64:       str
    bar_b64:            str


class ReportResponse(BaseModel):
    claim_id:          str
    member_letter:     str
    adjuster_summary:  str
    prediction_prob:   float
    recommended_action: str


class AuditEntry(BaseModel):
    audit_id:           str
    claim_id:           str
    member_id:          str
    adjuster_name:      str
    model_decision:     str
    model_prob:         float
    final_decision:     str
    override_reason:    str
    timestamp:          str


class AuditLogResponse(BaseModel):
    entries:   list[AuditEntry]
    total:     int
    page:      int
    page_size: int
