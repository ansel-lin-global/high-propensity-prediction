from google.cloud import bigquery
import pandas as pd

def fetch_data(project_id: str, event_table: str, output_path: str):
    """
    Fetch GA4 user events from BigQuery and save to GCS or local.
    Only select page_view, add_to_cart, purchase events.
    """
    # NOTE: Real query omitted due to company NDA
    query = f"SELECT * FROM `{event_table}` WHERE event_name IN (...)"
    
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    df.to_csv(output_path, index=False)
