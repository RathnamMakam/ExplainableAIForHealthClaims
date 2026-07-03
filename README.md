# XAI Health Insurance Claims

An end-to-end **Explainable AI** demo for health insurance claims processing, running fully local on Windows 11 with GPU acceleration. The system scores claims for denial risk, explains every decision using SHAP and LIME, generates ACA-compliant denial letters, and logs every adjuster override for regulator audit.

---

## Business Use Case

Health insurance claim denials affect millions of patients and cost payers and providers billions in administrative overhead. Traditional ML-based scoring systems act as black boxes — they score a claim but cannot tell an adjuster *why*, leaving humans unable to catch model errors, data quality issues, or unfair patterns.

This system demonstrates how **explainability closes that gap**:

| Without XAI | With XAI |
|---|---|
| Model says "DENY" — adjuster guesses why | SHAP waterfall shows the exact feature driving denial |
| Data entry errors silently cause denials | Top SHAP driver flags the missing prior auth NPI instantly |
| Member files a 30-day appeal | Adjuster corrects the record in minutes, no appeal needed |
| Compliance team can't audit for bias | Global SHAP confirms age/sex/race are not primary drivers |
| Denial letter lists vague reasons | LIME-derived plain-language reasons, ACA-compliant |

**Key demo scenario:** A claim is denied because the prior authorization NPI was never entered (keying error). The authorization was obtained. SHAP identifies `prior_auth_present` as the dominant driver (+0.556 log-odds) — 11× larger than any other feature. Correcting the data field drops the denial probability by 15 percentage points and flips the feature contribution from positive to negative. The adjuster resolves it in minutes.

---

## Target Audiences

- **Claims adjusters** — per-claim SHAP waterfall and LIME reason list on the adjuster dashboard
- **Compliance / audit teams** — global SHAP feature importance confirms no protected-attribute bias; full override audit trail
- **Medical directors** — counterfactual scores quantify impact of data quality fixes
- **Members / patients** — ACA-compliant denial letter with plain-language reasons and appeal instructions

---

## Architecture

```
CMS DE-SynPUF (claim records)  +  MEPS (denial rate calibration)
              |
              | ingestion pipeline (.venv-data)
              v
      data/processed/claims_features.parquet   (2.6M rows, 42 cols)
              |
              | training + XAI  (.venv-model)
              v
      XGBoost model (models/)  <--- MLflow experiment tracking
              |
              +--- SHAP TreeExplainer  (global beeswarm + per-claim waterfall)
              +--- LIME TabularExplainer  (per-claim ranked reasons)
              +--- ReportGenerator  (denial letter + adjuster summary)
              |
              | serving layer  (.venv-api)
              v
      FastAPI :8000
        POST /claims/score          -> denial probability + risk level
        POST /explain/shap          -> SHAP values + waterfall PNG
        POST /explain/lime          -> LIME coefficients + bar chart PNG
        GET  /explain/shap/global   -> global feature importance + plots
        POST /explain/report        -> member letter + adjuster summary
        GET  /audit/log             -> paginated override audit trail
        POST /audit/override        -> log adjuster decision
        GET  /health                -> liveness + readiness
              |
              | HTTPX calls
              v
      Streamlit :8501  (adjuster dashboard)
        - 3 demo claim presets
        - Full 36-feature input form
        - SHAP / LIME / Letter / Adjuster Summary tabs
        - Override panel with audit trail viewer
```

Streamlit never imports model or XAI code directly — all computation runs inside FastAPI.

---

## Project Structure

```
ExplainableAIForHealthClaims/
├── config.yaml                    # Paths, model params, SHAP/LIME settings
├── requirements-data.txt          # .venv-data dependencies
├── requirements-model.txt         # .venv-model dependencies
├── requirements-api.txt           # .venv-api dependencies
│
├── data/
│   ├── raw/synpuf/                # Immutable SynPUF CSV/ZIP downloads
│   ├── raw/meps/                  # MEPS Full Year Consolidated files
│   └── processed/                 # Parquet outputs
│       ├── claims_raw.parquet         (2,575,504 x 45)
│       ├── claims_features.parquet    (2,575,504 x 42, 36 features + label)
│       ├── meps_denial_rates.parquet  (12 rows, literature fallback)
│       └── shap_global.parquet        (2,000 x 36 SHAP values)
│
├── models/
│   ├── xgboost_claims.ubj         # Trained XGBoost model (256 KB)
│   ├── lgbm_claims.txt            # Trained LightGBM model (102 KB)
│   ├── feature_cols.json          # Ordered feature name list
│   ├── shap_beeswarm.png          # Global SHAP beeswarm plot
│   ├── shap_bar_summary.png       # Global SHAP bar chart
│   └── shap_waterfall_demo.png    # Per-claim waterfall (demo claim)
│
├── mlruns/                        # MLflow local experiment tracking
│
├── src/
│   ├── ingestion/
│   │   ├── synpuf_loader.py       # SynPUF ZIP/CSV -> claims_raw.parquet
│   │   └── meps_loader.py         # MEPS -> meps_denial_rates.parquet
│   ├── features/
│   │   ├── encoders.py            # ICD-9 prefix encoder, chronic flag map
│   │   └── claim_features.py      # Full Polars pipeline -> claims_features.parquet
│   ├── model/
│   │   ├── train.py               # XGBoost + LightGBM GPU training + MLflow
│   │   ├── evaluate.py            # Metrics (AUC, F1, PR) + feature importance plot
│   │   └── registry.py            # save/load .ubj and .txt model artifacts
│   ├── explainability/
│   │   ├── shap_explainer.py      # Global beeswarm/bar + local waterfall + counterfactuals
│   │   ├── lime_explainer.py      # LIME TabularExplainer + bar chart
│   │   └── report_generator.py    # Member denial letter + adjuster summary
│   ├── api/
│   │   ├── main.py                # FastAPI app, lifespan startup, CORS
│   │   ├── schemas/claims.py      # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── claims.py          # /claims/score
│   │   │   ├── explain.py         # /explain/shap, /lime, /shap/global, /report
│   │   │   └── audit.py           # /audit/log, /audit/override
│   │   └── services/
│   │       └── model_service.py   # Singleton model/explainer loader, helpers
│   ├── dashboard/
│   │   └── app.py                 # Streamlit adjuster dashboard
│   └── demo/
│       └── demo_script.py         # Four narrated demo scenarios (CLI)
│
├── demo_output/                   # Artifacts generated by demo_script.py
└── logs/                          # API and dashboard stdout/stderr logs
```

---

## Data Sources

| Dataset | Access | Purpose |
|---|---|---|
| [CMS DE-SynPUF](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files) | Free, no DUA | Claim records: ICD-9, HCPCS, billed/paid amounts, provider NPI |
| [MEPS Full Year Consolidated](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp) | Free, no DUA | Denial rate calibration (base rate ~14.9%) |

SynPUF samples 1–3 are used for development (2.6M claim rows). All 20 samples can be used for production scale.

The denial label is synthetic (Bernoulli sample calibrated to the MEPS base rate of 14.9%) because SynPUF contains no ground-truth denial outcome. The log-odds model weights `prior_auth_present` heavily, which is intentional — it makes the XAI demo story clear and auditable.

---

## Model Performance

| Model | Test AUC-ROC | Best Iteration | Hardware |
|---|---|---|---|
| XGBoost (primary) | 0.5750 | 35 | RTX 3060 CUDA |
| LightGBM (challenger) | 0.5750 | 28 | RTX 3060 OpenCL |

The modest AUC is expected — the synthetic label has inherent Bernoulli noise concentrated near the 15% base rate. The XAI story is carried by the feature ordering (SHAP ranks `prior_auth_present` correctly), not raw accuracy.

SHAP global top features: `clm_pmt_amt`, `primary_icd9_num`, `bene_hmo_mons`, `prior_auth_present`, `chronic_count`.
Protected attributes: `bene_age` ranks #8 (1.4% of top feature), `bene_sex` #32 (0.1%), `bene_race` #27 (0.2%).

---

## System Requirements

- Windows 11 (64-bit)
- Python 3.11 or 3.12
- NVIDIA GPU with CUDA 12.4 (for training only; inference is CPU)
- 16 GB RAM minimum, 64 GB recommended for full 2.6M-row dataset
- ~4 GB disk for models, data, and venvs

---

## Setup Guide

### 1. Prerequisites

Install CUDA Toolkit 12.4 from [developer.nvidia.com](https://developer.nvidia.com/cuda-12-4-0-download-archive) before creating `.venv-model`.

### 2. Clone and enter the project

```powershell
cd C:\Data\FrontlineProjects\ExplainableAIForHealthClaims
```

### 3. Create the three virtual environments each for data, model and api

```powershell
python -m venv .venv-data
python -m venv .venv-model
python -m venv .venv-api
```

### 4. Install dependencies

```powershell
.venv-data\Scripts\activate
pip install -r requirements-data.txt

.venv-model\Scripts\activate
pip install -r requirements-model.txt

.venv-api\Scripts\activate
pip install -r requirements-api.txt
```

### 5. Validate GPU (optional but recommended)

```powershell
.venv-model\Scripts\activate
python -c "import xgboost as xgb; m = xgb.XGBClassifier(device='cuda', n_estimators=10); print('XGBoost GPU OK')"
```

---

## Data Preparation

### Download SynPUF files

Download [DE-SynPUF Sample 1–3](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/de-synpuf-file-information-and-documentation) and place in `data/raw/synpuf/`. The loader accepts both ZIP and extracted CSV files.

Expected filenames (case-insensitive):
```
DE1_0_2008_Beneficiary_Summary_File_Sample*.csv
DE1_0_2008_to_2010_Inpatient_Claims_Sample*.csv
DE1_0_2008_to_2010_Outpatient_Claims_Sample*.csv
```

### Run the ingestion pipeline

```powershell
.venv-data\Scripts\activate
python src/ingestion/synpuf_loader.py --samples 1 2 3
python src/ingestion/meps_loader.py
```

Outputs: `data/processed/claims_raw.parquet`, `data/processed/meps_denial_rates.parquet`

### Run feature engineering

```powershell
.venv-data\Scripts\activate
python src/features/claim_features.py --config config.yaml
```

Output: `data/processed/claims_features.parquet` (2,575,504 rows × 42 columns)

---

## Model Training

```powershell
.venv-model\Scripts\activate
python src/model/train.py --config config.yaml
```

Trains XGBoost (GPU) and LightGBM (GPU) with early stopping. Saves artifacts to `models/` and logs both runs to MLflow.

**View MLflow experiment results:**
```powershell
.venv-model\Scripts\activate
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## SHAP and LIME Explainability

Run SHAP global + local (standalone, saves plots to `models/`):

```powershell
.venv-model\Scripts\activate
python src/explainability/shap_explainer.py --config config.yaml
```

Run LIME local + denial letter (standalone):

```powershell
.venv-model\Scripts\activate
python src/explainability/report_generator.py --config config.yaml
```

---

## Running the API and Dashboard

### Terminal 1 — FastAPI (must start first)

```powershell
.venv-api\Scripts\activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The server loads the XGBoost model, SHAP explainer, LIME explainer, and global SHAP cache on startup (~30 seconds). When ready you will see:

```
[model_service] Ready.
INFO:     Application startup complete.
```

**Interactive API docs:** http://localhost:8000/docs

### Terminal 2 — Streamlit dashboard

```powershell
.venv-api\Scripts\activate
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

### Dashboard walkthrough

1. Select a **demo scenario** from the sidebar (e.g., "Denied: Missing Prior Auth")
2. Review pre-populated claim features — or edit any field manually
3. Click **Score Claim**
4. Review the four result tabs:
   - **SHAP Explanation** — waterfall plot showing each feature's log-odds contribution, plus counterfactual hints
   - **LIME Explanation** — independent bar chart of local linear feature weights
   - **Denial Letter** — ACA-compliant member notice (downloadable)
   - **Adjuster Summary** — internal technical report (downloadable)
5. Use the **Override Panel** to approve or deny with a documented reason
6. Expand **Audit Log** to see all logged decisions

---

## Demo Script (Four Scenarios)

Runs all four scripted scenarios against the live API and saves presentation artifacts to `demo_output/`:

```powershell
# FastAPI must be running on :8000 first
.venv-api\Scripts\activate
python src/demo/demo_script.py
```

| Scenario | What it shows |
|---|---|
| 1. Adjuster Review | SHAP + LIME identify missing prior auth as top denial driver; adjuster overrides after verifying paper record |
| 2. Compliance Audit | Global SHAP confirms protected attributes (age, sex, race) are not primary drivers; audit trail reviewed |
| 3. Patient Letter | ACA-compliant denial notice with plain-language LIME reasons; all 6 compliance checks pass |
| 4. Counterfactual | Score drops 15 pp (24% relative) and SHAP flips sign when prior auth NPI is corrected |

Artifacts saved: SHAP waterfall × 3, LIME bar, beeswarm, global bar chart, denial letter text.

---

## Configuration

All tunable parameters live in `config.yaml`:

```yaml
model:
  params:
    device: cuda          # cuda for GPU training, cpu for CPU
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    early_stopping_rounds: 20

shap:
  max_display: 20         # features shown in summary plots
  background_samples: 100

lime:
  num_features: 10        # top features in local explanation
  num_samples: 1000       # perturbation samples per claim

api:
  port: 8000

dashboard:
  api_base_url: http://localhost:8000
  port: 8501
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/claims/score` | Score a claim → denial probability, risk level, recommended action |
| `POST` | `/explain/shap` | SHAP local explanation → waterfall PNG (base64) + top drivers |
| `POST` | `/explain/lime` | LIME local explanation → bar chart PNG (base64) + ranked reasons |
| `GET` | `/explain/shap/global` | Global SHAP summary → beeswarm + bar PNG + feature importance dict |
| `POST` | `/explain/report` | Combined report → member denial letter + adjuster summary text |
| `GET` | `/audit/log` | Paginated override audit trail |
| `POST` | `/audit/override` | Log an adjuster decision with reason |
| `GET` | `/health` | Liveness + readiness check |

All POST endpoints accept a `ClaimInput` JSON body with 36 feature fields. See `/docs` for the full schema and try-it-out interface.

---

## Key Design Decisions

**Three isolated virtual environments** — data processing (Polars), model training (XGBoost/SHAP/LIME), and serving (FastAPI/Streamlit) have separate dependency sets. This prevents version conflicts, keeps the serving layer lightweight, and makes each layer independently testable.

**Synthetic denial label** — SynPUF has no ground-truth denial outcome, so the label is constructed via a log-odds model calibrated to the MEPS denial base rate (~14.9%). `prior_auth_present` is weighted heavily by design: it makes the XAI story auditable and the counterfactual compelling.

**SHAP + LIME agreement as validation** — both methods run independently on every claim. When they agree on the top driver, it is a strong signal that the explanation is robust rather than an artifact of one method's assumptions.

**No SHAP GPU** — SHAP `TreeExplainer` runs on CPU. This is expected and correct; TreeExplainer uses the exact tree path algorithm which doesn't benefit from GPU parallelism.

**LIME NaN imputation** — columns that are null for outpatient claims (`drg_num`, `utlztn_days`) are imputed with the column median from the background sample before LIME's sklearn distance calculations, which reject NaN.

**Streamlit calls FastAPI only** — the dashboard never imports model or XAI code. All computation is in FastAPI, keeping the serving venv lightweight and the API independently testable.
