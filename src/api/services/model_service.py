"""
Singleton model service — loaded once at API startup via FastAPI lifespan.

Holds:
  model         -- XGBClassifier loaded from models/
  feature_cols  -- ordered list of feature names
  shap_exp      -- SHAPExplainer instance
  lime_exp      -- LIMEExplainer instance (fitted on 500-row background)
  global_shap   -- cached GlobalSHAPResponse (built once at startup)
  audit_log     -- in-memory list[dict] for adjuster decisions
"""

from __future__ import annotations

import base64
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.model.registry import load_xgb
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

class _State:
    model:         Any | None = None
    feature_cols:  list[str] | None = None
    shap_exp:      SHAPExplainer | None = None
    lime_exp:      LIMEExplainer | None = None
    report_gen:    ReportGenerator | None = None
    global_cache:  dict | None = None   # pre-built GlobalSHAPResponse payload
    audit_log:     list[dict] = []
    cfg:           dict = {}

_state = _State()


def init_service() -> None:
    """Call once during FastAPI lifespan startup."""
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        _state.cfg = yaml.safe_load(f)

    models_dir    = ROOT / _state.cfg["paths"]["models"]
    processed_dir = ROOT / _state.cfg["paths"]["data_processed"]

    print("[model_service] Loading XGBoost model ...", flush=True)
    _state.model, _state.feature_cols = load_xgb(models_dir)

    print("[model_service] Initialising SHAP explainer ...", flush=True)
    _state.shap_exp = SHAPExplainer(_state.model, _state.feature_cols)

    print("[model_service] Loading LIME background sample ...", flush=True)
    feat_path = processed_dir / "claims_features.parquet"
    table     = pq.read_table(feat_path, columns=_state.feature_cols)
    X_all     = table.to_pandas()
    rng       = np.random.default_rng(42)
    bg_idx    = rng.choice(len(X_all), size=500, replace=False)
    X_bg      = X_all.iloc[bg_idx].reset_index(drop=True)

    print("[model_service] Fitting LIME explainer ...", flush=True)
    _state.lime_exp = LIMEExplainer(
        _state.model, _state.feature_cols, X_bg, _state.cfg
    )

    _state.report_gen = ReportGenerator(_state.shap_exp, _state.lime_exp)

    print("[model_service] Building global SHAP cache ...", flush=True)
    _build_global_cache(models_dir)

    print("[model_service] Ready.", flush=True)


def get_state() -> _State:
    return _state


# ---------------------------------------------------------------------------
# Helpers used by routers
# ---------------------------------------------------------------------------

def claim_to_df(claim_dict: dict) -> pd.DataFrame:
    """Convert a ClaimInput dict to a single-row DataFrame with correct column order."""
    row = {col: claim_dict.get(col, np.nan) for col in _state.feature_cols}
    df  = pd.DataFrame([row])
    # None becomes object dtype; force all columns to float so XGBoost accepts them
    return df.astype(float)


def score_claim(X_row: pd.DataFrame) -> dict:
    prob = float(_state.model.predict_proba(X_row)[0, 1])
    action = "DENY" if prob > 0.5 else "APPROVE"
    level  = (
        "HIGH"   if prob >= 0.7 else
        "MEDIUM" if prob >= 0.4 else
        "LOW"
    )
    return {"prediction_prob": prob, "recommended_action": action, "denial_risk_level": level}


def explain_shap(X_row: pd.DataFrame) -> dict:
    result   = _state.shap_exp.explain_local(X_row)
    png_bytes = _state.shap_exp.plot_local_waterfall(X_row)
    hints    = _state.shap_exp.counterfactual_hints(result)
    return {**result, "waterfall_b64": base64.b64encode(png_bytes).decode(), "hints": hints}


def explain_lime(X_row: pd.DataFrame) -> dict:
    result    = _state.lime_exp.explain_claim(X_row)
    png_bytes = _state.lime_exp.plot_lime_bar(result)
    return {**result, "bar_b64": base64.b64encode(png_bytes).decode()}


def generate_report(X_row: pd.DataFrame, meta: dict) -> dict:
    return _state.report_gen.generate(X_row, meta)


def add_audit_entry(
    claim_id: str,
    member_id: str,
    adjuster_name: str,
    model_prob: float,
    recommended_action: str,
    override_decision: str,
    override_reason: str,
) -> dict:
    entry = {
        "audit_id":          str(uuid.uuid4())[:8],
        "claim_id":          claim_id,
        "member_id":         member_id,
        "adjuster_name":     adjuster_name,
        "model_decision":    recommended_action,
        "model_prob":        round(model_prob, 4),
        "final_decision":    override_decision,
        "override_reason":   override_reason,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }
    _state.audit_log.append(entry)
    return entry


def get_audit_log(page: int = 1, page_size: int = 20) -> dict:
    total   = len(_state.audit_log)
    start   = (page - 1) * page_size
    entries = _state.audit_log[start: start + page_size]
    return {"entries": entries, "total": total, "page": page, "page_size": page_size}


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _build_global_cache(models_dir: Path) -> None:
    bee_path = models_dir / "shap_beeswarm.png"
    bar_path = models_dir / "shap_bar_summary.png"

    if not bee_path.exists() or not bar_path.exists():
        print("[model_service] Global SHAP plots not found — skipping cache.", flush=True)
        _state.global_cache = None
        return

    shap_parquet = ROOT / _state.cfg["paths"]["data_processed"] / "shap_global.parquet"
    if shap_parquet.exists():
        sv_df = pd.read_parquet(shap_parquet)
        mean_abs = sv_df.drop(columns=["sample_idx"], errors="ignore").abs().mean()
        importance = mean_abs.sort_values(ascending=False).to_dict()
    else:
        importance = {}

    _state.global_cache = {
        "feature_importance": importance,
        "beeswarm_b64":       base64.b64encode(bee_path.read_bytes()).decode(),
        "bar_b64":            base64.b64encode(bar_path.read_bytes()).decode(),
    }
