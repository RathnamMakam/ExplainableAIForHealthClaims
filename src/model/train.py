"""
Phase 5 — Model training + MLflow experiment tracking.

Trains XGBoost (primary, GPU) and LightGBM (challenger, GPU) on
claims_features.parquet.  Each model gets its own MLflow run under the
experiment "health_claims_xai".  Best model (XGBoost) is saved to models/.

Usage:
  python src/model/train.py --config config.yaml
"""

import argparse
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.evaluate import compute_metrics, plot_feature_importance
from src.model.registry import save_lgbm, save_xgb

EXPERIMENT_NAME = "health_claims_xai"

# Feature columns — must match src/features/claim_features.py FEATURE_COLS
FEATURE_COLS = [
    "is_inpatient", "los_days", "claim_year", "claim_month",
    "clm_pmt_amt", "primary_payer_amt", "has_primary_payer", "is_adjustment",
    "prior_auth_present", "has_op_surgeon",
    "primary_icd9_num", "num_diagnoses", "num_procedures",
    "has_procedure", "has_hcpcs", "drg_num", "utlztn_days",
    "bene_age", "bene_sex", "bene_race", "bene_esrd",
    "bene_hi_mons", "bene_hmo_mons", "bene_is_deceased",
    "has_alzheimer", "has_chf", "has_ckd", "has_cancer", "has_copd",
    "has_depression", "has_diabetes", "has_ihd", "has_osteoporosis",
    "has_ra_oa", "has_stroke",
    "chronic_count",
]
LABEL_COL = "denial_label"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    path = ROOT / cfg["paths"]["data_processed"] / "claims_features.parquet"
    print(f"Loading {path.name} ...", end=" ", flush=True)

    # Read only the columns we need — faster than loading all 42 cols
    table = pq.read_table(path, columns=FEATURE_COLS + [LABEL_COL])
    df    = table.to_pandas()
    print(f"{len(df):,} rows")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    cfg:     dict,
    models_dir: Path,
) -> dict:
    p = cfg["model"]["params"]
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"\n  scale_pos_weight = {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        device             = p.get("device", "cuda"),
        n_estimators       = p.get("n_estimators", 500),
        max_depth          = p.get("max_depth", 6),
        learning_rate      = p.get("learning_rate", 0.05),
        subsample          = p.get("subsample", 0.8),
        colsample_bytree   = p.get("colsample_bytree", 0.8),
        eval_metric        = p.get("eval_metric", "auc"),
        early_stopping_rounds = p.get("early_stopping_rounds", 20),
        scale_pos_weight   = scale_pos_weight,
        random_state       = cfg["model"].get("random_state", 42),
        verbosity          = 1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"\n  Best iteration: {best_iter}")

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]
    train_metrics = {f"train_{k}": v for k, v in compute_metrics(y_train.to_numpy(), y_prob_train).items()}
    test_metrics  = {f"test_{k}":  v for k, v in compute_metrics(y_test.to_numpy(),  y_prob_test).items()}

    model_path = save_xgb(model, FEATURE_COLS, models_dir)
    print(f"  Saved -> {model_path}")

    return {
        "model":        model,
        "y_prob_test":  y_prob_test,
        "best_iter":    best_iter,
        "scale_pos_weight": scale_pos_weight,
        **train_metrics,
        **test_metrics,
    }


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    cfg:     dict,
    models_dir: Path,
) -> dict:
    p = cfg["lightgbm"]["params"]

    model = lgb.LGBMClassifier(
        device          = p.get("device", "gpu"),
        n_estimators    = p.get("n_estimators", 500),
        max_depth       = p.get("max_depth", 6),
        learning_rate   = p.get("learning_rate", 0.05),
        subsample       = p.get("subsample", 0.8),
        colsample_bytree= p.get("colsample_bytree", 0.8),
        class_weight    = "balanced",
        random_state    = cfg["model"].get("random_state", 42),
        verbosity       = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(50)],
    )

    best_iter = model.best_iteration_
    print(f"\n  Best iteration: {best_iter}")

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]
    train_metrics = {f"train_{k}": v for k, v in compute_metrics(y_train.to_numpy(), y_prob_train).items()}
    test_metrics  = {f"test_{k}":  v for k, v in compute_metrics(y_test.to_numpy(),  y_prob_test).items()}

    model_path = save_lgbm(model, models_dir)
    print(f"  Saved -> {model_path}")

    return {
        "model":       model,
        "y_prob_test": y_prob_test,
        "best_iter":   best_iter,
        **train_metrics,
        **test_metrics,
    }


# ---------------------------------------------------------------------------
# MLflow run wrapper
# ---------------------------------------------------------------------------

def _log_run(
    run_name:    str,
    model_type:  str,
    params:      dict,
    results:     dict,
    model,
    models_dir:  Path,
) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", model_type)
        mlflow.log_params(params)

        # Log all scalar metrics
        scalar_metrics = {k: v for k, v in results.items()
                          if isinstance(v, (int, float)) and k not in ("model", "y_prob_test")}
        mlflow.log_metrics(scalar_metrics)

        # Feature importance plot
        with tempfile.TemporaryDirectory() as tmp:
            fig_path = Path(tmp) / "feature_importance.png"
            if model_type == "xgboost":
                importance = model.feature_importances_
            else:
                importance = model.feature_importances_
            plot_feature_importance(
                importance, FEATURE_COLS,
                title=f"{model_type} — Feature Importance (gain)",
                out_path=fig_path,
            )
            mlflow.log_artifact(str(fig_path), artifact_path="plots")

        # Log model artifact path (the .ubj / .txt is already saved)
        mlflow.log_artifact(
            str(models_dir / ("xgboost_claims.ubj" if model_type == "xgboost" else "lgbm_claims.txt")),
            artifact_path="model",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models_dir = ROOT / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri((ROOT / cfg["paths"]["mlruns"]).as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y = load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = cfg["model"].get("test_size", 0.2),
        random_state = cfg["model"].get("random_state", 42),
        stratify     = y,
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Denial rate — train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\n=== XGBoost (primary, GPU) ===")
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test, cfg, models_dir)
    _log_run(
        run_name   = "xgboost_v1",
        model_type = "xgboost",
        params     = {**cfg["model"]["params"],
                      "scale_pos_weight": round(xgb_results["scale_pos_weight"], 2),
                      "train_rows": len(X_train), "test_rows": len(X_test)},
        results    = xgb_results,
        model      = xgb_results["model"],
        models_dir = models_dir,
    )

    print("\nXGBoost test metrics:")
    for k in ("test_auc_roc", "test_avg_precision", "test_f1", "test_precision", "test_recall"):
        print(f"  {k}: {xgb_results[k]:.4f}")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print("\n=== LightGBM (challenger, GPU) ===")
    lgb_results = train_lightgbm(X_train, y_train, X_test, y_test, cfg, models_dir)
    _log_run(
        run_name   = "lightgbm_v1",
        model_type = "lightgbm",
        params     = {**cfg["lightgbm"]["params"],
                      "class_weight": "balanced",
                      "train_rows": len(X_train), "test_rows": len(X_test)},
        results    = lgb_results,
        model      = lgb_results["model"],
        models_dir = models_dir,
    )

    print("\nLightGBM test metrics:")
    for k in ("test_auc_roc", "test_avg_precision", "test_f1", "test_precision", "test_recall"):
        print(f"  {k}: {lgb_results[k]:.4f}")

    # ── Comparison ───────────────────────────────────────────────────────────
    print("\n=== Model Comparison (test AUC-ROC) ===")
    print(f"  XGBoost  : {xgb_results['test_auc_roc']:.4f}  "
          f"(best iter {xgb_results['best_iter']})")
    print(f"  LightGBM : {lgb_results['test_auc_roc']:.4f}  "
          f"(best iter {lgb_results['best_iter']})")
    print(f"\nPrimary model (XGBoost) saved to: {models_dir}/xgboost_claims.ubj")
    print(f"MLflow UI: mlflow ui --port 5000 --backend-store-uri {ROOT / cfg['paths']['mlruns']}")


if __name__ == "__main__":
    main()
