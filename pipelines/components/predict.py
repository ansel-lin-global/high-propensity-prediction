"""
Component: predict_with_best_model

Loads the latest trained model and feature config from GCS,
applies the model to fresh input data from BigQuery, and
writes the top-k scored predictions to a destination table.

All sensitive paths and logic have been anonymized for open-source demonstration.
"""

from kfp.dsl import component
from typing import List

@component(
    base_image="python:3.10",
    packages_to_install=[
        'pandas', 'joblib', 'google-cloud-bigquery', 'google-cloud-storage',
        'google-cloud-bigquery-storage', 'scikit-learn',
        'catboost', 'lightgbm', 'xgboost', 'db-dtypes'
    ]
)
def predict_with_best_model(
    project: str,
    export_bucket: str,
    top_k: int,
    daily_predict_query: str,
    prediction_output_table: str
):
    """
    Applies the latest trained model to fresh data and writes top-k scored results to BigQuery.
    """
    import pandas as pd
    import joblib, io, json, re
    from datetime import datetime
    from google.cloud import bigquery, storage, bigquery_storage

    # Load latest model and scaler from GCS
    prefix = export_bucket.split("/", 1)[1] if "/" in export_bucket else ""
    bucket_name = export_bucket.split("/", 1)[0] if "/" in export_bucket else export_bucket
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    model_pattern = re.compile(rf"{prefix}/(\d{{4}}-\d{{2}}-\d{{2}})_(.+)_model\.pkl" if prefix else r"(\d{4}-\d{2}-\d{2})_(.+)_model\.pkl")
    latest_date = datetime.min
    latest_model_blob, latest_model_type = None, None

    for blob in blobs:
        match = model_pattern.match(blob.name)
        if match:
            date_str, model_type = match.groups()
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date > latest_date:
                latest_date, latest_model_blob, latest_model_type = date, blob, model_type

    if not latest_model_blob:
        raise RuntimeError("No valid model found.")

    scaler_blob = bucket.blob(latest_model_blob.name.replace("_model.pkl", "_scaler.pkl"))
    model = joblib.load(io.BytesIO(latest_model_blob.download_as_bytes()))
    scaler = joblib.load(io.BytesIO(scaler_blob.download_as_bytes()))

    # Load prediction input data from BigQuery
    bq_client = bigquery.Client(project=project)
    bq_storage = bigquery_storage.BigQueryReadClient()
    df = bq_client.query(daily_predict_query).to_dataframe(bqstorage_client=bq_storage)

    # Load latest feature config
    feature_pattern = re.compile(rf"{prefix}/(\d{{4}}-\d{{2}}-\d{{2}})_(cat|numeric)\.json")
    feature_date, numeric_cols, cat_cols = datetime.min, None, None

    for blob in blobs:
        match = feature_pattern.match(blob.name)
        if match:
            date = datetime.strptime(match.group(1), "%Y-%m-%d")
            if date > feature_date:
                feature_date = date
                numeric_cols = json.loads(bucket.blob(blob.name.replace("cat", "numeric")).download_as_text())
                cat_cols = json.loads(bucket.blob(blob.name.replace("numeric", "cat")).download_as_text())

    # Preprocess features
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])

    if latest_model_type == "catboost":
        for col in cat_cols:
            df_scaled[col] = df_scaled[col].fillna("missing").astype(str)
    else:
        for col in cat_cols:
            df_scaled[col] = df_scaled[col].astype("category")

    df_scaled = df_scaled[model.feature_names_]
    df["score"] = model.predict_proba(df_scaled)[:, 1]

    df_top_k = df.sort_values("score", ascending=False).head(top_k)
    df_top_k["predict_date"] = datetime.today().date()

    # Output to BigQuery
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    bq_client.load_table_from_dataframe(
        df_top_k[["user_pseudo_id", "score", "predict_date"]],
        prediction_output_table,
        job_config=job_config
    ).result()

    print(f"Top-{top_k} predictions written to {prediction_output_table}")
