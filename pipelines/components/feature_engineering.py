"""
Component: Feature Engineering

Purpose:
Use BigQuery SQL to generate features from raw GA4 logs — such as session depth, product views, category diversity,
event frequency, recency ratios, etc.

Input:
- Raw event logs (already pre-filtered and split via time_series_split)
- Stored in BigQuery table: `project.dataset.raw_event_log`

Output:
- Daily updated user-feature table in BigQuery: `project.dataset.user_features_{train_date}`

Notes:
- This step is SQL-based and scheduled in BigQuery via dbt / scheduled query.
- Features are fetched downstream by Vertex AI components (model training, prediction).
"""

from google.cloud import bigquery

def run_feature_engineering(train_date: str):
    """
    Trigger scheduled BigQuery feature generation query.

    Args:
        train_date (str): Date string for generating that day's features (e.g., '2025-07-15')
    """

    client = bigquery.Client()

    sql = f"""
    -- Simplified placeholder
    CREATE OR REPLACE TABLE `project.dataset.user_features_{train_date}` AS
    SELECT
        user_pseudo_id,
        COUNTIF(event_name = 'add_to_cart') AS add_to_cart_count,
        COUNTIF(event_name = 'view_item') AS view_item_count,
        COUNTIF(event_name = 'purchase') AS purchase_count,
        COUNT(*) AS total_events,
        MAX(event_timestamp) AS last_event_time
    FROM `project.dataset.raw_event_log`
    WHERE DATE(event_timestamp) BETWEEN DATE_SUB(DATE('{train_date}'), INTERVAL 14 DAY) AND DATE('{train_date}')
    GROUP BY user_pseudo_id
    """

    client.query(sql).result()
    print(f"Feature table generated for {train_date}")
