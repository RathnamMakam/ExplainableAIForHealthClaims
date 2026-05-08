"""
GET  /audit/log
POST /audit/override
"""

from fastapi import APIRouter, Query
from src.api.schemas.claims import AuditEntry, AuditLogResponse, OverrideRequest
from src.api.services import model_service as svc

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/log", response_model=AuditLogResponse)
def get_log(
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> AuditLogResponse:
    data = svc.get_audit_log(page=page, page_size=page_size)
    return AuditLogResponse(
        entries   = [AuditEntry(**e) for e in data["entries"]],
        total     = data["total"],
        page      = data["page"],
        page_size = data["page_size"],
    )


@router.post("/override", response_model=AuditEntry)
def override(req: OverrideRequest) -> AuditEntry:
    entry = svc.add_audit_entry(
        claim_id           = req.claim_id,
        member_id          = req.member_id,
        adjuster_name      = req.adjuster_name,
        model_prob         = req.original_prob,
        recommended_action = req.recommended_action,
        override_decision  = req.override_decision,
        override_reason    = req.reason,
    )
    return AuditEntry(**entry)
