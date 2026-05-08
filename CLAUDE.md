# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: XAI Health Insurance Claims Demo

An Explainable AI demo showcasing SHAP and LIME for health insurance claims processing, running fully local on Windows 11 (64GB RAM, 12GB VRAM, NVIDIA RTX 3060).

**Target audiences:** Claims adjusters, compliance/audit teams, medical directors, patients (denial letters).

---

## Infrastructure Decisions

### Environment
- **Three isolated Python 3.11 venvs** — do not mix dependencies across them:
  - `.venv-data` — ingestion and feature engineering (Polars, PyArrow, Pandas)
  - `.venv-model` — training and XAI (XGBoost GPU, LightGBM GPU, SHAP, LIME, MLflow)
  - `.venv-api` — serving layer (FastAPI + Uvicorn, Streamlit, HTTPX)
- CUDA Toolkit 12.4 must be installed system-wide before creating `.venv-model`
- **Do not use Conda** — project uses Python venv + pip only

### GPU Usage
- XGBoost: `device="cuda"` param
- LightGBM: `device="gpu"` param (requires system CUDA — verify at setup)
- SHAP TreeExplainer runs on CPU — this is expected and correct
- VRAM budget: ~2–3GB of 12GB used during training

### Storage
- Raw data lives in `data/raw/` — never overwrite, treat as immutable
- All processed outputs written as **Parquet** to `data/processed/`
- MLflow artifacts go to `mlruns/` (local tracking server, no remote)

---

## Data Sources

| Dataset | Access | Purpose |
|---|---|---|
| CMS DE-SynPUF | cms.gov — free, no DUA | Claim-level records: ICD-10, HCPCS, billed/paid amounts, provider NPI |
| MEPS | meps.ahrq.gov — free | Payment denial flags — ground-truth outcome variable |
| CMS Provider Utilization | data.cms.gov — free | Fee schedule benchmarks, provider specialty |

Use 2–3 SynPUF sample files for development; all 20 for full demo. The SynPUF + MEPS combination provides claim-level records with a ground-truth denial/approval label without any access barriers.

---

## Project Structure

```
ExplainableAIForHealthClaims/
├── data/
│   ├── raw/synpuf/       # Immutable downloaded CSVs
│   ├── raw/meps/
│   ├── processed/        # Parquet outputs of ingestion pipeline
│   └── reference/        # ICD-10 / HCPCS lookup tables
├── notebooks/            # Numbered exploration notebooks (01–05)
├── src/
│   ├── ingestion/        # synpuf_loader.py, meps_loader.py
│   ├── features/         # claim_features.py, encoders.py
│   ├── model/            # train.py, evaluate.py, registry.py
│   ├── explainability/   # shap_explainer.py, lime_explainer.py, report_generator.py
│   ├── api/              # FastAPI app (main.py, routers/, schemas/, services/)
│   └── dashboard/        # app.py — Streamlit, calls FastAPI via HTTPX
├── models/               # Saved XGBoost model artifact (.ubj)
├── mlruns/               # MLflow local tracking (auto-generated)
├── requirements-data.txt
├── requirements-model.txt
├── requirements-api.txt
└── config.yaml           # Paths, model params, GPU flags
```

---

## Key Commands

### Environment setup
```powershell
python -m venv .venv-data
python -m venv .venv-model
python -m venv .venv-api

.venv-data\Scripts\activate; pip install -r requirements-data.txt
.venv-model\Scripts\activate; pip install -r requirements-model.txt
.venv-api\Scripts\activate; pip install -r requirements-api.txt
```

### GPU validation (run after .venv-model setup)
```powershell
.venv-model\Scripts\activate
python -c "import xgboost as xgb; m = xgb.XGBClassifier(device='cuda', n_estimators=10); print('XGBoost GPU OK')"
```

### Data ingestion
```powershell
.venv-data\Scripts\activate
python src/ingestion/synpuf_loader.py --samples 1 2 3
python src/ingestion/meps_loader.py
```

### Model training
```powershell
.venv-model\Scripts\activate
python src/model/train.py --config config.yaml
```

### FastAPI server
```powershell
.venv-api\Scripts\activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Streamlit dashboard (FastAPI must be running on :8000 first)
```powershell
.venv-api\Scripts\activate
streamlit run src/dashboard/app.py
# Opens at localhost:8501
```

### MLflow UI
```powershell
.venv-model\Scripts\activate
mlflow ui --port 5000
# Opens at localhost:5000
```

---

## Architecture: How the Layers Connect

```
SynPUF CSV + MEPS CSV
       │ (ingestion pipeline — .venv-data)
       ▼
  Parquet files (data/processed/)
       │ (feature engineering + training — .venv-model)
       ▼
  XGBoost model (models/)  ←── MLflow tracks all experiments
       │
       ├── SHAP TreeExplainer (global + per-claim)
       └── LIME perturbation explainer (per-claim)
               │
               ▼
         FastAPI :8000  (.venv-api)
          POST /claims/score
          POST /explain/shap
          POST /explain/lime
          GET  /explain/shap/global
          GET  /audit/log
          POST /audit/override
               │ (HTTPX calls)
               ▼
       Streamlit :8501  (.venv-api)
         Adjuster dashboard — claim input, SHAP/LIME plots, override panel
```

Streamlit **never** imports model or SHAP/LIME code directly — it always calls FastAPI over HTTP. This keeps the API independently testable and the serving venv lightweight.

---

## XAI Design

- **SHAP global** (summary/beeswarm): for compliance teams — audits feature importance and confirms age/gender are not primary drivers
- **SHAP local** (waterfall/force plot): per-claim adjuster explanation — shows baseline score and each feature's directional contribution
- **LIME local**: ranked reason list in plain language — used for member-facing denial letters (ACA requires denial reasons)
- **Counterfactual framing**: derived from SHAP values — "if prior auth were on file, score would increase by X" — actionable feedback for providers

Key demo scenario to engineer: a claim the model denies whose top SHAP driver is a data quality issue (e.g., prior auth submitted but not matched). This shows XAI catches model errors a black box would silently perpetuate.

---

## FastAPI Endpoints

```
POST  /claims/score           Submit claim → approval score + confidence
POST  /explain/shap           Claim → SHAP values + feature contributions
POST  /explain/lime           Claim → LIME coefficients + reason list
GET   /explain/shap/global    Global SHAP summary (feature importance)
GET   /audit/log              Paginated adjuster decision audit trail
POST  /audit/override         Log adjuster override with reason
GET   /health                 Liveness check
```

---

## Build Sequence (current status: planning complete, Phase 1 not yet started)

```
Phase 1   System setup — CUDA 12.4, Python 3.11, three venvs + pip installs
Phase 2   GPU validation — XGBoost + LightGBM smoke test on RTX 3060
Phase 3   Data ingestion pipeline — SynPUF + MEPS loaders → Parquet
Phase 4   Feature engineering + label construction
Phase 5   Model training + MLflow experiment tracking
Phase 6   SHAP global + local explainability
Phase 7   LIME explanation + denial letter generator
Phase 8   FastAPI layer — endpoints, Pydantic schemas, services
Phase 9   Streamlit dashboard — HTTPX integration with FastAPI
Phase 10  Demo scenario scripting — adjuster, fraud detection, audit trail
```
