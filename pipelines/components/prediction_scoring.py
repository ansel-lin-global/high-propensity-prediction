"""
Component: Prediction Scoring

📌 Purpose:
Use trained LightGBM model to score new daily user features, and export top-K users with highest purchase propensity.

🧾 Input:
- BigQuery table: user_features_{predict_date}
- GCS: trained model from lgbm_model_{train_date}.pkl

📤 Output:
- CSV file of top-K high-propensity users uploaded to GCS
"""

import pandas as pd
import joblib
from google.cloud import bigquery, storage

def predict_users(predict_date: str, train_date: str, project_id: str, bucket_name: str, k_top: int = 50):
    # 1. Load user features from BigQuery
    bq_client = bigquery.Client(project=project_id)
    query = f"""
    SELECT * FROM `project.dataset.user_features_{predict_date}`
    """
    df = bq_client.query(query).to_dataframe()

    user_ids = df['user_pseudo_id']
    features = df.drop(columns=['user_pseudo_id'])

    # 2. Load trained model from GCS
    storage_client = storage.Client(project=project_id)
    model_blob = storage_client.bucket(bucket_name).blob(f"model_output/lgbm_model_{train_date}.pkl")
    model_path = f"models/lgbm_model_{train_date}.pkl"
    model_blob.download_to_filename(model_path)

    model = joblib.load(model_path)

    # 3. Predict probabilities
    probs = model.predict_proba(features)[:, 1]
    df['score'] = probs

    # 4. Select top-K users
    top_k_df = df[['user_pseudo_id', 'score']].sort_values(by='score', ascending=False).head(k_top)

    # 5. Upload results to GCS
    result_path = f"top_users/top_{k_top}_users_{predict_date}.csv"
    top_k_df.to_csv(result_path, index=False)

    result_blob = storage_client.bucket(bucket_name).blob(result_path)
    result_blob.upload_from_filename(result_path)

    print(f"[✓] Prediction results for {predict_date} uploaded to GCS.")

