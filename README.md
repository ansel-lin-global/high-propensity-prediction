![status](https://img.shields.io/badge/status-portfolio-blue)
![mlops](https://img.shields.io/badge/MLOps-Vertex%20AI%20Pipelines-informational)
![scope](https://img.shields.io/badge/scope-Drift→Retrain→Deploy-success)

# 🎯 High Propensity Purchase Prediction System

An end-to-end MLOps system to identify high-intent users for e-commerce campaigns — built with Vertex AI Pipelines, automated drift detection, retraining, and integrated CI/CD deployment.

🔗 **Full project write-up**: [Medium Article](https://medium.com/@ansel-lin/from-model-to-deployment-building-an-automated-high-propensity-purchase-prediction-system-2aed17de9412)

> **📌 Portfolio Note**  
> This repo is a sanitized version of a production system. All GCP project IDs, table names, and credentials are parameterized — no sensitive information is committed. Some components are excluded for confidentiality, but the pipeline orchestration and architecture are fully representative of the production system.

---

## 🧠 Problem Statement

Traditional EDM segmentation often targets users based on simple heuristics or trending products. This leads to low conversion rates and wasted campaign budget.

This project solves that by:
- Predicting high-intent users using rich behavioral signals (GA4)
- Automating model lifecycle: training, prediction, drift detection, and retraining
- Enabling targeted EDM segmentation at global scale

---

## 📈 Business Impact

- Identifies **50+ high-intent users daily** across 40+ e-commerce sites
- Powers **real EDM campaigns**, improving targeting recall
- Serves as **an internal reference project** for predictive modeling & GenAI

---

## ⚙️ System Architecture

The system is built as a fully modular pipeline with scheduled jobs on Vertex AI. It automates prediction, monitoring, drift detection, and retraining — all data-driven, not time-driven.

```mermaid
flowchart LR
    A[GA4 Event Data<br>in BigQuery] --> B[Rolling Feature Generation<br>14d window → 3d label]
    B --> C[Model Training<br>LightGBM + SMOTE + Undersampling]
    C --> D[Daily Prediction<br>Vertex AI Pipelines]
    D --> E[Drift Detection<br>Data & Concept Drift]
    E --> F[Trigger Retraining<br>GitHub Actions]
    F --> G[Export Top-K Users<br>to GCS for EDM]
```

---

## 🗂 Repo Structure

```
high-propensity-prediction/
├── README.md
├── requirements.txt                  # All Python dependencies
├── LICENSE
│
├── components/                       # Reusable Vertex AI Pipeline components
│   ├── README.md                     #   ↳ Component documentation & unpublished list
│   ├── train.py                      #   ↳ Time-series splitting, LightGBM training
│   ├── predict.py                    #   ↳ Load best model, score daily, write to BQ
│   ├── drift.py                      #   ↳ PSI-based data drift, recall-based concept drift
│   └── retrain.py                    #   ↳ Drift check → feature engineering → pipeline trigger
│
├── pipelines/                        # Vertex AI Pipeline definitions (DAGs)
│   ├── README.md                     #   ↳ Pipeline documentation
│   ├── training_pipeline.py          #   ↳ Sliding window training with multi-model comparison
│   ├── predict_pipeline.py           #   ↳ Daily scoring using latest model
│   ├── drift_pipeline.py             #   ↳ Data drift + concept drift monitoring
│   └── retrain_pipeline.py           #   ↳ Conditional retraining on drift
│
├── scripts/                          # CLI tools for CI/CD
│   ├── compile_and_package.py        #   ↳ Compile pipelines to JSON specs
│   └── submit_pipeline_job.py        #   ↳ Submit pipeline jobs to Vertex AI
│
├── configs/                          # Configuration templates
│   └── example_pipeline_params.yaml  #   ↳ Example params for all pipelines
│
├── docs/
│   └── decisions.md                  # Key design decisions & rationale
│
├── .github/workflows/
│   └── deploy-vertex-ai.yml          # CI/CD: compile → submit on Vertex AI
│
├── .env.example                      # Required GitHub Secrets reference
└── .gitignore
```

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Cloud** | GCP Vertex AI Pipelines, BigQuery, GCS |
| **ML Frameworks** | LightGBM, CatBoost, XGBoost, SMOTE, Scikit-learn |
| **Pipeline SDK** | Kubeflow Pipelines (KFP v2) |
| **CI/CD** | GitHub Actions + Workload Identity Federation |
| **Monitoring** | Custom drift detection (PSI, recall degradation) |

---

## 🧪 Key Features & Techniques

- **Modular Pipeline Design**: Separated components for feature engineering, training, scoring, and drift detection
- **Rolling Window Labeling**: 14-day behavior window → 3-day intent prediction, built for temporal generalization
- **Multi-Model Training**: LightGBM, XGBoost, CatBoost trained and evaluated in parallel via `dsl.ParallelFor`
- **Data-Driven Retraining**: Drift-aware mechanism based on PSI and recall monitoring, not fixed schedule
- **Business-Centric Metrics**: Optimized for Recall@TopK and campaign ROI, not just model loss
- **Zero-Credential Codebase**: All sensitive values are parameterized or pulled from GitHub Secrets

---

## 🔁 Auto-Retraining Architecture

This project implements **automatic retraining** triggered by **data or concept drift**.

1. **Daily Drift Check**: A scheduled pipeline evaluates feature drift (PSI) and concept drift (recall degradation)
2. **Drift Detected**: If `drift_severity = "STRONG"`, the pipeline triggers model retraining automatically
3. **CI/CD Integration**: GitHub Actions handles code updates and pipeline template deployment; daily operations run inside Vertex AI Pipelines

> This architecture ensures retraining is **data-driven**, not time-driven, making the system more robust and cost-effective.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- GCP project with Vertex AI, BigQuery, and GCS enabled
- Authenticated via `gcloud auth application-default login` or Workload Identity Federation

### Setup
```bash
# Clone the repo
git clone https://github.com/ansel-lin-global/high-propensity-prediction.git
cd high-propensity-prediction

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your configuration
cp configs/example_pipeline_params.yaml configs/my_params.yaml
# Edit configs/my_params.yaml with your GCP project details
```

### Compile & Submit a Pipeline
```bash
# Compile a specific pipeline
python scripts/compile_and_package.py --only training --out-dir artifacts

# Submit to Vertex AI
python scripts/submit_pipeline_job.py \
  --project <YOUR_PROJECT> \
  --region us-central1 \
  --staging-bucket gs://<YOUR_BUCKET> \
  --service-account <YOUR_SA>@<PROJECT>.iam.gserviceaccount.com \
  --pipeline-spec artifacts/training-*.json \
  --param bq_project=<YOUR_PROJECT> \
  --param date_col=date \
  --param gap=3 \
  --param prediction_window=1
```

### CI/CD (GitHub Actions)
1. Set up the required secrets (see [`.env.example`](.env.example))
2. Go to **Actions → Deploy Vertex AI Pipeline → Run workflow**
3. Select the pipeline to deploy

---

## 🔭 Roadmap

✅ Data Drift Detection — implemented and running in production  
✅ Concept Drift Detection — monitors feature-target relationship shifts  
✅ Auto-Retraining — powered by Vertex AI Pipelines & GitHub Actions  

🔄 Multi-Model Comparison & A/B Testing — experiment design in progress  
🔄 Uplift Modeling Extension — prototyping with synthetic campaign data  

**📄 For detailed design decisions, see [docs/decisions.md](docs/decisions.md)**  

---

## 👋 About Me

I'm [Ansel](https://www.linkedin.com/in/ansel-lin/), a Product-Focused Data Scientist building predictive systems, GenAI applications, and MLOps pipelines that drive real impact across global e-commerce products.

📩 Reach out via [Medium](https://medium.com/@ansel-lin) or [GitHub](https://github.com/ansel-lin-global) to collaborate.
