"""
Component: Feature Engineering (SQL-based)

📌 Purpose:
Execute SQL transformation to materialize features for retraining.

🧾 Input:
- GCS path to feature SQL template
- Raw GA4 BigQuery table
- Output BigQuery table

📤 Output:
- Transformed table written to BigQuery
"""

from google.cloud import bigquery, storage
from datetime import datetime, timedelta

def run_feature_engineering_sql(project: str, raw_data_table: str, output_table: str, feature_sql_gcs_uri: str):
    client = bigquery.Client(project=project)
    storage_client = storage.Client(project=project)

    today = datetime.utcnow().date()
    start_date = today - timedelta(days=30)
    end_date = today - timedelta(days=1)

    # Load SQL template from GCS
    bucket, blob = feature_sql_gcs_uri.replace("gs://", "").split("/", 1)
    sql_template = storage_client.bucket(bucket).blob(blob).download_as_text()

    # Replace template placeholders
    sql = sql_template.replace("{{RAW_DATA_TABLE}}", raw_data_table)\
                      .replace("{{OUTPUT_TABLE}}", output_table)\
                      .replace("{{START_DATE}}", str(start_date))\
                      .replace("{{END_DATE}}", str(end_date))

    client.query(sql).result()
