"""
Model persistence — save and load XGBoost / LightGBM artifacts.

Each saved model directory contains:
  xgboost_claims.ubj   — XGBoost binary model
  lgbm_claims.txt      — LightGBM text model (if trained)
  feature_cols.json    — ordered list of feature column names
"""

import json
from pathlib import Path
from typing import Any


def save_xgb(model: Any, feature_cols: list[str], models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgboost_claims.ubj"
    model.save_model(model_path)
    (models_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    return model_path


def load_xgb(models_dir: Path) -> tuple[Any, list[str]]:
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(models_dir / "xgboost_claims.ubj")
    feature_cols = json.loads((models_dir / "feature_cols.json").read_text())
    return m, feature_cols


def save_lgbm(model: Any, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "lgbm_claims.txt"
    model.booster_.save_model(str(model_path))
    return model_path


def load_lgbm(models_dir: Path) -> Any:
    import lightgbm as lgb
    return lgb.Booster(model_file=str(models_dir / "lgbm_claims.txt"))
