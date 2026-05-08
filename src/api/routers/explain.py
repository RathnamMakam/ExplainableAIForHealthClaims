"""
POST /explain/shap
POST /explain/lime
GET  /explain/shap/global
POST /explain/report
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas.claims import (
    ClaimInput,
    SHAPLocalResponse,
    LIMELocalResponse,
    GlobalSHAPResponse,
    ReportResponse,
    TopDriver,
    TopReason,
)
from src.api.services import model_service as svc

router = APIRouter(prefix="/explain", tags=["explain"])


@router.post("/shap", response_model=SHAPLocalResponse)
def explain_shap(claim: ClaimInput) -> SHAPLocalResponse:
    X_row  = svc.claim_to_df(claim.model_dump())
    result = svc.explain_shap(X_row)
    return SHAPLocalResponse(
        claim_id              = claim.claim_id,
        base_value            = result["base_value"],
        prediction_prob       = result["prediction_prob"],
        shap_values           = result["shap_values"],
        top_drivers           = [TopDriver(**d) for d in result["top_drivers"]],
        waterfall_b64         = result["waterfall_b64"],
        counterfactual_hints  = result["hints"],
    )


@router.post("/lime", response_model=LIMELocalResponse)
def explain_lime(claim: ClaimInput) -> LIMELocalResponse:
    X_row  = svc.claim_to_df(claim.model_dump())
    result = svc.explain_lime(X_row)
    return LIMELocalResponse(
        claim_id        = claim.claim_id,
        prediction_prob = result["prediction_prob"],
        top_reasons     = [TopReason(**r) for r in result["top_reasons"]],
        intercept       = result["intercept"],
        local_pred      = result["local_pred"],
        bar_b64         = result["bar_b64"],
    )


@router.get("/shap/global", response_model=GlobalSHAPResponse)
def global_shap() -> GlobalSHAPResponse:
    cache = svc.get_state().global_cache
    if cache is None:
        raise HTTPException(status_code=503, detail="Global SHAP cache not available.")
    return GlobalSHAPResponse(**cache)


@router.post("/report", response_model=ReportResponse)
def generate_report(claim: ClaimInput) -> ReportResponse:
    X_row  = svc.claim_to_df(claim.model_dump())
    score  = svc.score_claim(X_row)
    meta   = {
        "claim_id":    claim.claim_id,
        "member_id":   claim.member_id,
        "service_date": claim.service_date,
        "provider":    claim.provider,
    }
    reports = svc.generate_report(X_row, meta)
    return ReportResponse(
        claim_id           = claim.claim_id,
        member_letter      = reports["member_letter"],
        adjuster_summary   = reports["adjuster_summary"],
        prediction_prob    = score["prediction_prob"],
        recommended_action = score["recommended_action"],
    )
