# Components Overview

This folder contains **reusable components** from a real-world [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines) project for high-propensity prediction.  

Only a **sanitized subset** of components is published here to demonstrate structure, typed I/O, and modular design.  
All sensitive queries, table names, and business logic have been removed or replaced with placeholders.

---

## ✅ Published Components

### `time_series_split.py`
Splits time-series data into multiple train/test windows with a configurable gap and prediction period.

- **Highlights**:
  - Accepts a Parquet dataset with a date column
  - Outputs a pickled list of `(train_idx, test_idx)` per window
  - Enables sliding-window model training without data leakage

---

### `count_total_windows.py`
Reads the split file and returns a list of window indices for iteration.

- **Highlights**:
  - Works seamlessly with `dsl.ParallelFor` to train on multiple windows in parallel
  - Outputs a simple list of integers representing window indices

---

### `model_training.py`
Trains a LightGBM classifier with proper handling of imbalanced data and categorical features.

- **Highlights**:
  - Uses `scale_pos_weight` for imbalance handling
  - Early stopping to prevent overfitting
  - Category-aware training using LightGBM native support

---

### `prediction_scoring.py`
Generates predictions using the most recent exported model and writes the top-k predictions to BigQuery.

- **Highlights**:
  - Automatically detects the latest model and feature schema from GCS
  - Supports LightGBM, XGBoost, and CatBoost
  - Outputs ready for business integration (e.g., EDM targeting)

---

### `drift_detection.py`
Checks for **data drift** and **concept drift** by comparing recent inference data with training baselines.

- **Highlights**:
  - Uses statistical metrics such as PSI and KS tests
  - Detects shifts in feature distribution and score patterns
  - Logs results to BigQuery for monitoring and triggering retraining

---

### `run_feature_engineering_sql.py`
Executes parameterized SQL to generate engineered features in BigQuery.

- **Highlights**:
  - Dynamically injects date ranges into the SQL
  - Materializes a new table with refreshed features
  - Fully compatible with automated retraining workflows

---

## 📦 Not Published (Original Project)

| Component                       | Purpose                                                           |
|--------------------------------|-------------------------------------------------------------------|
| `resample_data`                | Undersampling + optional SMOTE for class balancing                |
| `preprocess_data`              | Applies scaling and prepares categorical variables                |
| `extract_window_data`          | Extracts X/y splits per window                                    |
| `train_xgb_model`              | Trains XGBoost model with categorical support                     |
| `train_catboost_model`         | Trains CatBoost model with Pool API and class weights             |
| `evaluate_model_to_file`       | Calculates PR-AUC, precision@k, lift; saves results to GCS         |
| `merge_and_write_to_bq`        | Merges evaluation files and loads them to BigQuery                |
| `summarize_eval_from_bq`       | Aggregates evaluation metrics across models                       |
| `export_best_model`            | Selects best model and copies it to GCS                           |
| `calculate_baseline_statistics`| Builds reference distributions from training features             |
| `log_drift_metrics_to_bigquery`| Logs drift results for dashboards and audit purposes              |

---

## 💡 Notes
- All components use `@kfp.dsl.component` for portability and reusability.
- I/O types use `Input[Dataset]` and `Output[Dataset]` for seamless GCS/Vertex AI integration.
- Dependencies are declared within each component via `packages_to_install`.
- For pipeline orchestration examples, see:
  - [`pipelines/training_pipeline.py`](../training_pipeline.py)
  - [`pipelines/daily_predict_pipeline.py`](../daily_predict_pipeline.py)
  - [`pipelines/drift_detection_pipeline.py`](../drift_detection_pipeline.py)
  - [`pipelines/retraining_pipeline.py`](../retraining_pipeline.py)
