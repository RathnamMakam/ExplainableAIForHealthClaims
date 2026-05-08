"""POST /claims/score"""

from fastapi import APIRouter
from src.api.schemas.claims import ClaimInput, ScoreResponse
from src.api.services import model_service as svc

router = APIRouter(prefix="/claims", tags=["claims"])


@router.post("/score", response_model=ScoreResponse)
def score(claim: ClaimInput) -> ScoreResponse:
    X_row  = svc.claim_to_df(claim.model_dump())
    result = svc.score_claim(X_row)
    return ScoreResponse(claim_id=claim.claim_id, **result)
