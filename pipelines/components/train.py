"""
This module defines reusable Vertex AI Pipeline components
for end-to-end ML training and evaluation on sliding windows.
All confidential data sources, paths, and logic have been sanitized
for open-source showcasing.
"""

from kfp.dsl import component, Dataset, Input, Output
from typing import List

# Example: split_data_by_time_series
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


# Example: count_total_windows
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


# Other components such as:
# - extract_window_data
# - resample_data (undersampling/SMOTE)
# - preprocess_data (scaling)
# - train_lgb_model, train_xgb_model, train_catboost_model
# - evaluate_model_to_file
# - merge_and_write_to_bq
# - summarize_eval_from_bq
# - export_best_model
# would follow similar refactoring:
# - Clear docstrings
# - Sanitized inputs/outputs
# - Removed company-specific paths
# - Focused on composability and transparency

# For brevity, full definitions are modularized in this folder
# to showcase practical, production-ready ML components.

# See full pipeline logic in pipelines/training_pipeline.py
# and GitHub README for system diagram and explanation.
