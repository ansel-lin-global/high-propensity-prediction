"""
This module defines reusable Vertex AI Pipeline components
for detecting both data drift and concept drift in deployed ML systems.

All sensitive logic, metrics, and configurations have been sanitized
for open-source demonstration.

The included components are designed to run regularly (e.g., daily) to monitor
the stability of model inputs and outputs and trigger retraining logic if needed.
"""

from kfp.dsl import component, Dataset, Input, Output


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scipy"]
)
def detect_data_drift_psi(
    baseline_data: Input[Dataset],
    current_data: Input[Dataset],
    feature: str,
    output_drift_score: Output[Dataset]
):
    """
    Computes PSI (Population Stability Index) for a single feature.
    """
    import pandas as pd
    import numpy as np

    def calculate_psi(expected, actual, buckets=10):
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1e-6
        actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1e-6
        expected_percents = expected_counts / expected_counts.sum()
        actual_percents = actual_counts / actual_counts.sum()
        psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
        return psi

    df_base = pd.read_parquet(baseline_data.path)
    df_curr = pd.read_parquet(current_data.path)

    psi_value = calculate_psi(df_base[feature], df_curr[feature])
    pd.DataFrame({"feature": [feature], "psi_score": [psi_value]}).to_parquet(output_drift_score.path, index=False)


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "google-cloud-bigquery"]
)
def detect_concept_drift_recall_drop(
    project: str,
    predict_table: str,
    current_date: str,
    output_concept_drift: Output[Dataset]
):
    """
    Detects concept drift by comparing today's recall@k with previous average.
    """
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project=project)

    current_query = f"""
        SELECT recall_at_k
        FROM `{predict_table}`
        WHERE predict_date = '{current_date}'
    """

    baseline_query = f"""
        SELECT AVG(recall_at_k) AS baseline_recall
        FROM `{predict_table}`
        WHERE predict_date < '{current_date}'
        AND recall_at_k IS NOT NULL
        LIMIT 30
    """

    current = client.query(current_query).to_dataframe()
    baseline = client.query(baseline_query).to_dataframe()

    current_val = current["recall_at_k"].values[0] if not current.empty else None
    baseline_val = baseline["baseline_recall"].values[0] if not baseline.empty else None
    degradation = baseline_val - current_val if current_val is not None else None

    pd.DataFrame({
        "baseline_recall": [baseline_val],
        "current_recall": [current_val],
        "recall_degradation": [degradation]
    }).to_parquet(output_concept_drift.path, index=False)

# === Not included: other components ===
# The following drift detection components were part of the original pipeline
# but are excluded here for brevity and confidentiality:
#
# - calculate_baseline_statistics:
#     Generates baseline distributions for key features from training data,
#     including histograms, value counts, and statistical summaries.
#
# - detect_data_drift:
#     Compares recent inference data to the training baseline using metrics
#     such as Population Stability Index (PSI), KS test, or Wasserstein distance.
#
# - detect_concept_drift:
#     Monitors prediction distributions, prediction confidence shifts,
#     or actual vs. predicted label discrepancies to capture post-deployment drift.
#
# - log_drift_metrics_to_bigquery:
#     Writes daily drift scores and metadata into a BigQuery log table
#     for monitoring, visualization, and retrain decisioning.
#
# All components follow the same conventions: clean input/output typing,
# scalable design, and full compatibility with Vertex AI Pipelines.
