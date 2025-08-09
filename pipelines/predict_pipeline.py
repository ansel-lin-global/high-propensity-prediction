"""
Vertex AI Pipeline — Daily Prediction

What this pipeline showcases:
- Load the latest exported model & scaler from GCS
- Fetch fresh scoring data from BigQuery and compute scores
- Write top-k results back to BigQuery for activation

Notes:
- Anonymized for showcase; replace project, tables, bucket, query.
- Caching is disabled to ensure daily scoring always runs.

Inputs (params):
- project, export_bucket, top_k
- daily_predict_query, prediction_output_table
"""

from kfp import dsl
from kfp.dsl import pipeline
from components.predict import predict_with_best_model

@pipeline(name="daily-predict-pipeline", description="Score daily data using the latest exported model")
def daily_predict_pipeline(
    project: str,
    export_bucket: str,
    top_k: int,
    daily_predict_query: str,
    prediction_output_table: str
):
    task = predict_with_best_model(
        project=project,
        export_bucket=export_bucket,
        top_k=top_k,
        daily_predict_query=daily_predict_query,
        prediction_output_table=prediction_output_table
    )

