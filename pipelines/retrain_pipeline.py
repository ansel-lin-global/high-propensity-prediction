"""
Vertex AI Pipeline — Conditional Retraining

What this pipeline showcases:
- Check latest drift summary to decide RETRAIN vs SKIP
- If RETRAIN: run feature engineering SQL (from GCS template) to materialize training table
- Trigger the training pipeline with a clean parameter dict (no hard-coded secrets)

Notes:
- Anonymized and parameterized for showcase; replace table names, buckets, and SQL URI.
- Demonstrates control-flow with dsl.If and clean hand-off to another pipeline.

Inputs (params):
- BigQuery: bq_project, raw_data_table, output_table
- GCS: feature_sql_gcs_uri, training_pipeline_uri, train_bucket
- Vertex: train_region, service_account, encryption_key
- Training params: fetch_raw_data_query, date_col, gap, prediction_window, top_k, output_bq_table, selection_metric, gcs_project, export_bucket
"""

from kfp import dsl
from kfp.dsl import pipeline
from components.retrain import check_drift_decision, run_feature_engineering_sql, trigger_training_pipeline

@pipeline(name="retrain-pipeline", description="Retrain model if drift is detected.")
def daily_drift_check_and_retrain(
    bq_project: str,
    raw_data_table: str,
    output_table: str,
    feature_sql_gcs_uri: str,
    training_pipeline_uri: str,
    train_region: str,
    train_bucket: str,
    service_account: str,
    encryption_key: str,
    fetch_raw_data_query: str,
    date_col: str,
    gap: int,
    prediction_window: int,
    top_k: int,
    output_bq_table: str,
    selection_metric: str,
    gcs_project: str,
    export_bucket: str
):
    decision = check_drift_decision(project=bq_project)

    with dsl.If(decision.output == "RETRAIN"):
        fe = run_feature_engineering_sql(
            project=bq_project,
            raw_data_table=raw_data_table,
            output_table=output_table,
            feature_sql_gcs_uri=feature_sql_gcs_uri
        )

        trigger_training_pipeline(
            project=bq_project,
            region=train_region,
            bucket=train_bucket,
            service_account=service_account,
            pipeline_uri=training_pipeline_uri,
            pipeline_params={
                "bq_project": bq_project,
                "fetch_raw_data_query": fetch_raw_data_query,
                "date_col": date_col,
                "gap": gap,
                "prediction_window": prediction_window,
                "k": top_k,
                "bq_table": output_bq_table,
                "selection_metric": selection_metric,
                "gcs_project": gcs_project,
                "export_bucket": export_bucket
            },
            encryption_key=encryption_key
        ).after(fe)
