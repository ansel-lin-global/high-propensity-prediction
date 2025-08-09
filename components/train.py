"""
This module defines reusable Vertex AI Pipeline components
for end-to-end ML training and evaluation on sliding windows.
All confidential data sources, paths, and logic have been sanitized
for open-source showcasing.
"""

from kfp.dsl import component, Dataset, Input, Output
from typing import List

# === Component 1: split_data_by_time_series ===
@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def split_data_by_time_series(
    input_dataset: Input[Dataset],
    date_col: str,
    gap: int,
    prediction_window: int,
    output_splits_path: Output[Dataset],
):
    """
    Splits time series data into rolling train/test windows with a defined gap.
    """
    import pandas as pd
    import pickle

    df = pd.read_parquet(input_dataset.path)
    df[date_col] = pd.to_datetime(df[date_col])

    unique_dates = sorted(df[date_col].unique())
    splits = []

    for i in range(len(unique_dates) - gap - prediction_window + 1):
        train_end = unique_dates[i]
        test_range = unique_dates[i + gap: i + gap + prediction_window]
        train_idx = df[df[date_col] <= train_end].index
        test_idx = df[df[date_col].isin(test_range)].index
        splits.append((train_idx.tolist(), test_idx.tolist()))

    with open(output_splits_path.path, 'wb') as f:
        pickle.dump(splits, f)


# === Component 2: count_total_windows ===
@component(
    base_image="python:3.10",
    packages_to_install=["pandas"]
)
def count_total_windows(
    splits_path: Input[Dataset]
) -> List[int]:
    """
    Returns the number of split windows to enable iteration.
    """
    import pickle
    with open(splits_path.path, 'rb') as f:
        splits = pickle.load(f)
    return list(range(len(splits)))


# === Component 3: train_lgb_model ===
@component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'scikit-learn', 'lightgbm', 'joblib', 'pyarrow']
)
def train_lgb_model(
    X_train_path: Input[Dataset],
    y_train_path: Input[Dataset],
    X_valid_path: Input[Dataset],
    y_valid_path: Input[Dataset],
    cat: Input[Dataset],
    model_output_path: Output[Dataset]
):
    """
    Train a LightGBM classifier using class weighting and category-aware handling.
    """
    import pandas as pd
    import joblib
    import json
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    X_train = pd.read_parquet(X_train_path.path)
    y_train = pd.read_parquet(y_train_path.path).squeeze()
    X_valid = pd.read_parquet(X_valid_path.path)
    y_valid = pd.read_parquet(y_valid_path.path).squeeze()

    with open(cat.path, 'r') as f:
        cat_cols = json.load(f)

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='pr-auc',
        categorical_feature=cat_cols,
        callbacks=[early_stopping(30), log_evaluation(0)]
    )

    joblib.dump(model, model_output_path.path)


# === Not included: other components ===
# The following are part of the original project but excluded for open-source brevity:
# - extract_window_data
# - resample_data (undersampling/SMOTE)
# - preprocess_data (scaling)
# - train_xgb_model, train_catboost_model
# - evaluate_model_to_file
# - merge_and_write_to_bq
# - summarize_eval_from_bq
# - export_best_model

# These follow similar component structure: typed I/O, modular logic, reusable design.
