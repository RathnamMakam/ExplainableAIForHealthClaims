"""
FastAPI application — XAI Health Claims API.

Startup loads XGBoost model, SHAP explainer, and LIME explainer once.
All heavy computation runs in the default thread pool via run_in_executor
when needed; SHAP/LIME are CPU-bound and fast enough synchronously for demo.

Run:
  .venv-api\\Scripts\\activate
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import claims, explain, audit
from src.api.services.model_service import init_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_service()
    yield


app = FastAPI(
    title       = "XAI Health Claims API",
    description = "Explainable AI for health insurance claims — SHAP + LIME explanations.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(claims.router)
app.include_router(explain.router)
app.include_router(audit.router)


@app.get("/health", tags=["system"])
def health():
    from src.api.services.model_service import get_state
    st = get_state()
    return {
        "status":       "ok",
        "model_loaded": st.model is not None,
        "shap_ready":   st.shap_exp is not None,
        "lime_ready":   st.lime_exp is not None,
        "global_cache": st.global_cache is not None,
        "audit_entries": len(st.audit_log),
    }
