from kfp.dsl import component, Output, Dataset

@component(
    base_image='python:3.10',
    packages_to_install=["google-cloud-bigquery", "pyarrow", "pandas"]
)
def fetch_raw_data(
    project: str,
    event_table: str,   # e.g., "project.dataset.ga4_events"
    output_dataset: Output[Dataset]
):
    """
    Fetch GA4 user events from BigQuery.
    Select only key behavior events and save as Parquet.
    """

    from google.cloud import bigquery
    import pandas as pd

    query = f"""
    SELECT user_pseudo_id, event_name, event_date, event_timestamp
    FROM `{event_table}`
    WHERE event_name IN ('page_view', 'add_to_cart', 'purchase')
    """

    client = bigquery.Client(project=project)
    df = client.query(query).to_dataframe()
    
    # Normalize datetime fields (optional)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")

    # Save to Parquet for downstream use
    df.to_parquet(output_dataset.path, index=False)
