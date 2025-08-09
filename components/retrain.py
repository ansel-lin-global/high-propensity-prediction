"""
Vertex AI Component Examples for Retraining Automation

Includes:
- check_drift_decision: determine whether retraining is needed
- run_feature_engineering_sql: generate features via GCS-based SQL template
- trigger_training_pipeline: launch full training pipeline if needed

Note: Logic has been sanitized for public demonstration. Some project-specific details are omitted.
"""

from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "pandas"]
)
def check_drift_decision(project: str) -> str:
    """
    Check latest drift logs in BigQuery and return 'RETRAIN' or 'SKIP'
    """
    from google.cloud import bigquery
    client = bigquery.Client(project=project)

    data_sql = """
        SELECT DISTINCT drift_recommendation
        FROM `your_project.drift_log_table`
        WHERE drift_check_date = (SELECT MAX(drift_check_date) FROM `your_project.drift_log_table`)
    """
    concept_sql = """
        SELECT DISTINCT drift_recommendation
        FROM `your_project.concept_drift_log`
        WHERE anchor_date = (SELECT MAX(anchor_date) FROM `your_project.concept_drift_log`)
    """

    data_df = client.query(data_sql).to_dataframe()
    concept_df = client.query(concept_sql).to_dataframe()

    decision = "RETRAIN" if "Strong" in data_df.values or "Strong" in concept_df.values else "SKIP"
    print(f"[DRIFT CHECK] Decision: {decision}")
    return decision


@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "google-cloud-storage"]
)
def run_feature_engineering_sql(
    project: str,
    raw_data_table: str,
    output_table: str,
    feature_sql_gcs_uri: str
):
    """
    Download SQL from GCS, replace template variables, and run as BigQuery job.
    """
    from google.cloud import bigquery, storage
    from datetime import datetime, timedelta

    # Load SQL template
    bucket_name, blob_path = feature_sql_gcs_uri.replace("gs://", "").split("/", 1)
    gcs = storage.Client(project=project)
    sql_template = gcs.bucket(bucket_name).blob(blob_path).download_as_text()

    # Replace placeholders
    today = datetime.utcnow().date()
    sql = sql_template \
        .replace("{{RAW_DATA_TABLE}}", raw_data_table) \
        .replace("{{OUTPUT_TABLE}}", output_table) \
        .replace("{{START_DATE}}", str(today - timedelta(days=30))) \
        .replace("{{END_DATE}}", str(today - timedelta(days=1)))

    # Execute SQL
    bigquery.Client(project=project).query(sql).result()
    print(f"[FEATURE ENGINEERING] Output to {output_table}")


@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform[pipelines]"]
)
def trigger_training_pipeline(
    project: str,
    region: str,
    pipeline_uri: str,
    pipeline_params: dict,
    bucket: str,
    service_account: str,
    encryption_key: str
):
    """
    Trigger training pipeline via Vertex AI using pipeline template URI.
    """
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region, staging_bucket=bucket)

    job = aiplatform.PipelineJob(
        display_name="retrain-pipeline-job",
        template_path=pipeline_uri,
        pipeline_root=f"{bucket}/pipeline_root",
        parameter_values=pipeline_params,
        enable_caching=False,
        encryption_spec_key_name=encryption_key
    )

    job.run(service_account=service_account, sync=False)
    print("[RETRAIN PIPELINE] Triggered asynchronously.")
