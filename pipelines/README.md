# Pipelines Overview

This folder contains template pipelines built with [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines), designed to demonstrate a real-world MLOps workflow for high-propensity prediction.

All pipelines are adapted from a production project and sanitized for public display.  
They retain the structural and orchestration logic but exclude sensitive queries, table names, or business-specific details.

---

## ✅ Published Pipelines

### `training_pipeline.py`
Orchestrates **end-to-end model training and evaluation** across multiple time-series windows.

- **Workflow Highlights**:
  - Fetch raw data from BigQuery
  - Split into sliding windows (with gap and prediction period)
  - Train multiple model candidates (e.g., LightGBM, XGBoost, CatBoost)
  - Evaluate performance and select the best model
  - Save evaluation results to BigQuery for review

- **Key Features**:
  - Modular component structure for reusability
  - Parallel window training using `dsl.ParallelFor`
  - Compatible with imbalanced datasets

---

### `prediction_pipeline.py`
Generates **daily or on-demand predictions** using the best available model.

- **Workflow Highlights**:
  - Load the most recent best model from GCS
  - Fetch and preprocess the latest input data
  - Generate prediction scores for all target entities
  - Save predictions to BigQuery for downstream use (e.g., marketing automation)

- **Key Features**:
  - Designed for scheduled or ad-hoc runs
  - Minimal latency by reusing trained models
  - Output ready for business integration (e.g., EDM targeting)

---

### `drift_pipeline.py`
Checks for **data drift** and **concept drift** to monitor model stability over time.

- **Workflow Highlights**:
  - Compare statistical distributions of recent vs. training data
  - Detect significant shifts in feature importance or prediction patterns
  - Log drift metrics and recommendations to BigQuery

- **Key Features**:
  - Supports both **data drift** (input distribution changes) and **concept drift** (target relationship changes)
  - Modular drift detection components for flexible thresholds
  - Output feeds into `retrain_pipeline.py` for automated retraining triggers

---

### `retrain_pipeline.py`
Triggers **model retraining** when drift is detected.

- **Workflow Highlights**:
  - Check drift logs and determine if retraining is necessary
  - Run feature engineering SQL to refresh the training dataset
  - Trigger the `training_pipeline.py` with updated data

- **Key Features**:
  - Conditional execution using `dsl.If`
  - End-to-end automated retraining with minimal manual intervention
  - Maintains model relevance without overfitting to short-term noise

---

## 📦 Not Published (Original Project)
The original project contains additional orchestration pipelines for:
- Multi-market batch predictions
- A/B testing integration with model scores
- Real-time scoring via Vertex AI endpoints

These are excluded here for brevity and confidentiality.

---

## 💡 Notes
- All pipelines follow the same design philosophy:
  - Modular, reusable components (`@kfp.dsl.component`)
  - Explicit parameterization for flexibility across markets and datasets
  - Secure handling of data sources and output locations
- This public version uses placeholders for:
  - BigQuery dataset and table names
  - GCS bucket URIs
  - Service account references

For reusable building blocks, see [`components`](components).
