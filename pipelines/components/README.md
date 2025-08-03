# Components Overview

This folder contains selected reusable components from a real-world Vertex AI Pipelines project.  
The components enable time-series based model training, evaluation, and selection using a sliding window framework.

Only a subset of components is published here to demonstrate the structure and logic.  
The rest are described below with usage and function summaries.

---

## ✅ Published Components

### `split_data_by_time_series`
Splits time-series data into multiple train/test windows with a gap and prediction period.

- **Inputs**: Parquet dataset with a date column
- **Outputs**: Pickled list of (train_idx, test_idx) per window
- **Use Case**: Enables cross-window model training without leakage

---

### `count_total_windows`
Reads split file and returns a list of window indices for looping.

- **Inputs**: Split file from `split_data_by_time_series`
- **Outputs**: List of integers (window indices)
- **Use Case**: Enables `dsl.ParallelFor` to loop over windows

---

### `train_lgb_model`
Trains a LightGBM classifier with proper handling of imbalanced data and categorical features.

- **Inputs**: Scaled train/test data, category column list
- **Outputs**: Saved model in `.pkl` format
- **Highlight**: Uses `scale_pos_weight` for imbalance, early stopping, category-aware training

---

## 📦 Other Components (Not Published)

| Component                | Purpose                                                |
|--------------------------|--------------------------------------------------------|
| `resample_data`          | Undersampling + optional SMOTE to balance label classes |
| `preprocess_data`        | Applies standard scaling and prepares category types    |
| `extract_window_data`    | Extracts X/y splits per window                          |
| `train_xgb_model`        | XGBoost training with categorical support               |
| `train_catboost_model`   | CatBoost model with Pool API and class weight           |
| `evaluate_model_to_file` | Calculates PR-AUC, precision@k, lift, saves to GCS      |
| `merge_and_write_to_bq`  | Merges evaluation files and loads to BigQuery           |
| `summarize_eval_from_bq` | Aggregates evaluation metrics across models             |
| `export_best_model`      | Selects best model based on a metric and copies to GCS  |

---

## 💡 Notes

- All components are written in a portable, reusable manner using `@kfp.dsl.component`.
- Inputs/Outputs use `Input[Dataset]` and `Output[Dataset]` for compatibility with GCS/Vertex Pipelines.
- All Python dependencies are defined in `packages_to_install`.

For full pipeline orchestration, see [`pipelines/training_pipeline.py`](../pipelines/training_pipeline.py)
