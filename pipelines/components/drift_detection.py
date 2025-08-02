"""
Component: Drift Detection

📌 Purpose:
Evaluate whether the current prediction window has significant data or concept drift compared to the training window.

🧾 Input:
- BigQuery tables: user_features_{train_date}, user_features_{predict_date}, labels_{predict_date}
- GCS: prediction scores from top_K_users_{predict_date}.csv

📤 Output:
- Drift flag (STRONG / WEAK / NONE) written to BigQuery or returned to pipeline
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from google.cloud import bigquery, storage

def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index (PSI) for continuous features"""
    def scale_range(input_array):
        return (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array) + 1e-5)

    expected, actual = scale_range(expected), scale_range(actual)
    breakpoints = np.linspace(0, 1, buckets + 1)

    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum((expected_bins - actual_bins) * np.log((expected_bins + 1e-4) / (actual_bins + 1e-4)))
    return psi

def detect_drift(train_date: str, predict_date: str, project_id: str, bucket_name: str, feature_cols: list):
    bq = bigquery.Client(project=project_id)

    # 1. Load training and prediction features
    df_train = bq.query(f"SELECT * FROM `project.dataset.user_features_{train_date}`").to_dataframe()
    df_predict = bq.query(f"SELECT * FROM `project.dataset.user_features_{predict_date}`").to_dataframe()

    # 2. Data Drift using PSI
    psi_scores = []
    for col in feature_cols:
        psi = calculate_psi(df_train[col].values, df_predict[col].values)
        psi_scores.append(psi)

    avg_psi = np.mean(psi_scores)

    # 3. Concept Drift using Score-Label correlation
    storage_client = storage.Client(project=project_id)
    prediction_blob = storage_client.bucket(bucket_name).blob(f"top_users/top_50_users_{predict_date}.csv")
    prediction_blob.download_to_filename("top_users.csv")
    df_score = pd.read_csv("top_users.csv")

    df_label = bq.query(f"SELECT user_pseudo_id, label FROM `project.dataset.labels_{predict_date}`").to_dataframe()
    df_joined = df_score.merge(df_label, on="user_pseudo_id", how="inner")

    correlation = df_joined['score'].corr(df_joined['label'])

    # 4. Heuristic decision rules
    drift_flag = "NONE"
    if avg_psi > 0.2 or (correlation is not None and correlation < 0.1):
        drift_flag = "STRONG"
    elif avg_psi > 0.1 or (correlation is not None and correlation < 0.3):
        drift_flag = "WEAK"

    print(f"[✓] Drift Check — PSI: {avg_psi:.3f}, Corr: {correlation:.3f}, Flag: {drift_flag}")
    return drift_flag
