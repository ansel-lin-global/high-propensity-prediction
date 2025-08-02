"""
Component: Check Retrain Decision

📌 Purpose:
This component checks recent entries in BigQuery data/concept drift logs and returns a retrain decision ("RETRAIN" or "SKIP").

🧾 Input:
- project: GCP project ID
- data_drift_table: BigQuery table path (e.g., project.dataset.data_drift_log)
- concept_drift_table: BigQuery table path (e.g., project.dataset.concept_drift_log)

📤 Output:
- A string: "RETRAIN" if either log indicates strong drift, else "SKIP"
"""

from kfp.dsl import component

@component(
    base_image='python:3.10',
    packages_to_install=["google-cloud-bigquery", "pandas"]
)
def check_retrain_decision(
    project: str,
    data_drift_table: str,
    concept_drift_table: str
) -> str:
    from google.cloud import bigquery
    client = bigquery.Client(project=project)

    data_sql = f"""
    SELECT drift_recommendation
    FROM `{data_drift_table}`
    WHERE drift_check_date = (
        SELECT MAX(drift_check_date) FROM `{data_drift_table}`
    )
    """

    concept_sql = f"""
    SELECT drift_recommendation
    FROM `{concept_drift_table}`
    WHERE anchor_date = (
        SELECT MAX(anchor_date) FROM `{concept_drift_table}`
    )
    """

    try:
        data_result = client.query(data_sql).to_dataframe()
        concept_result = client.query(concept_sql).to_dataframe()

        if "Strong" in data_result["drift_recommendation"].values or \
           "Strong" in concept_result["drift_recommendation"].values:
            print("Detected strong drift → RETRAIN")
            return "RETRAIN"

        print("No strong drift → SKIP")
        return "SKIP"

    except Exception as e:
        print(f"[Error] Could not retrieve or process drift data: {e}")
        return "SKIP"
