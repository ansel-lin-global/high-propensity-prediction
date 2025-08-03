"""
Vertex AI Pipeline: Daily Prediction

This pipeline fetches the latest deployed model and applies it
to fresh data from BigQuery, returning the top-k highest scored users.
"""

from kfp import dsl
from kfp.dsl import pipeline
from components.predict import predict_with_best_model

@pipeline(name="daily-predict-pipeline")
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
    task.set_caching_options(enable_caching=False)
