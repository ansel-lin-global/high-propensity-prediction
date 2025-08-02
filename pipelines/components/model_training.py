"""
Component: Model Training

📌 Purpose:
Train a LightGBM model to predict purchase propensity, using engineered features from BigQuery.
Tackle severe class imbalance using undersampling + SMOTE.

🔁 Input:
- BigQuery table: user_features_{train_date}
- Label column: purchased_within_3d (binary)

📤 Output:
- Trained LightGBM model saved to GCS
- Evaluation metrics saved to GCS as JSON
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
import json
import os

from google.cloud import bigquery, storage

def train_model(train_date: str, project_id: str, bucket_name: str):
    # 1. Load features from BigQuery
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT * FROM `project.dataset.user_features_{train_date}`
    """
    df = client.query(query).to_dataframe()

    # 2. Prepare data
    label_col = 'purchased_within_3d'
    features = df.drop(columns=[label_col, 'user_pseudo_id'])
    labels = df[label_col]

    # 3. Handle class imbalance: undersample + SMOTE
    X_train, _, y_train, _ = train_test_split(features, labels, stratify=labels, test_size=0.9, random_state=42)

    # Apply SMOTE on minority class
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 4. Train LightGBM model
    lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    lgb_model.fit(X_res, y_res)

    # 5. Evaluate
    y_pred = lgb_model.predict(X_res)
    precision = precision_score(y_res, y_pred)
    report = classification_report(y_res, y_pred, output_dict=True)

    # 6. Export model
    model_path = f'models/lgbm_model_{train_date}.pkl'
    os.makedirs('models', exist_ok=True)
    joblib.dump(lgb_model, model_path)

    # 7. Upload to GCS
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    # Model file
    blob_model = bucket.blob(f"model_output/lgbm_model_{train_date}.pkl")
    blob_model.upload_from_filename(model_path)

    # Metrics file
    metrics = {
        'train_date': train_date,
        'precision': precision,
        'report': report
    }
    metrics_path = f'models/metrics_{train_date}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    blob_metrics = bucket.blob(f"model_output/metrics_{train_date}.json")
    blob_metrics.upload_from_filename(metrics_path)

    print(f"[✓] Model + metrics saved for {train_date}")

