# Components Overview

This folder contains **reusable Vertex AI Pipeline components** from a production-grade high-propensity prediction system.

Only a **sanitized subset** is published to demonstrate structure, typed I/O, and modular design.  
All sensitive queries, table names, and business logic have been removed or parameterized.

---

## ✅ Published Components

### `train.py`

Contains core training components used in the sliding-window cross-validation pipeline.

| Component | Purpose |
|-----------|---------|
| `split_data_by_time_series` | Splits time-series data into rolling train/test windows with a configurable gap and prediction period. Outputs a pickled list of `(train_idx, test_idx)` per window. |
| `count_total_windows` | Reads the split file and returns a list of window indices for iteration with `dsl.ParallelFor`. |
| `train_lgb_model` | Trains a LightGBM classifier with `scale_pos_weight` for imbalanced data, early stopping, and category-aware handling. |

---

### `predict.py`

| Component | Purpose |
|-----------|---------|
| `predict_with_best_model` | Loads the latest model and scaler from GCS, scores fresh data from BigQuery, and writes top-k predictions. Supports LightGBM, XGBoost, and CatBoost. |

---

### `drift.py`

| Component | Purpose |
|-----------|---------|
| `detect_data_drift_psi` | Computes PSI (Population Stability Index) for a single feature to detect input distribution shifts. |
| `detect_concept_drift_recall_drop` | Detects concept drift by comparing today's recall@k with the historical average from BigQuery. |

---

### `retrain.py`

| Component | Purpose |
|-----------|---------|
| `check_drift_decision` | Queries drift log tables in BigQuery and returns `'RETRAIN'` or `'SKIP'` based on drift severity. |
| `run_feature_engineering_sql` | Downloads a SQL template from GCS, injects date-range placeholders, and executes as a BigQuery job. |
| `trigger_training_pipeline` | Launches the full training pipeline on Vertex AI using a compiled pipeline template URI. |

---

## 📦 Not Published (Production Components)

These components are part of the original production system but excluded for brevity and confidentiality.  
They follow the same conventions: `@kfp.dsl.component`, typed I/O, and independent dependencies.

| Component | Purpose |
|-----------|---------|
| `fetch_raw_data` | Fetches data from BigQuery via parameterized SQL |
| `inspect_schema` / `store_schema_features` | Detects and persists numeric/categorical feature lists to GCS |
| `resample_data` | Undersampling + optional SMOTE for class balancing |
| `preprocess_data` | Applies scaling and prepares categorical variables |
| `extract_window_data` | Extracts X/y splits per window from the full dataset |
| `train_xgb_model` | Trains XGBoost model with categorical support |
| `train_catboost_model` | Trains CatBoost model with Pool API and class weights |
| `evaluate_model_to_file` | Calculates PR-AUC, precision@k, lift; saves results to GCS |
| `merge_and_write_to_bq` | Merges evaluation files and loads them to BigQuery |
| `summarize_eval_from_bq` | Aggregates evaluation metrics across models |
| `export_best_model` | Selects best model and copies it to GCS |
| `calculate_baseline_statistics` | Builds reference distributions from training features |
| `log_drift_metrics_to_bigquery` | Logs drift results for dashboards and audit purposes |

---

## 💡 Design Conventions

- All components use `@kfp.dsl.component` for portability and reusability
- I/O types use `Input[Dataset]` and `Output[Dataset]` for seamless GCS/Vertex AI integration
- Dependencies are declared within each component via `packages_to_install`
- No hardcoded project IDs, table names, or bucket paths — everything is parameterized

For pipeline orchestration examples, see:
- [`pipelines/training_pipeline.py`](../pipelines/training_pipeline.py)
- [`pipelines/predict_pipeline.py`](../pipelines/predict_pipeline.py)
- [`pipelines/drift_pipeline.py`](../pipelines/drift_pipeline.py)
- [`pipelines/retrain_pipeline.py`](../pipelines/retrain_pipeline.py)
