import pandas as pd
import lightgbm as lgb
from imblearn.combine import SMOTEENN
from google.cloud import storage
import joblib
import os

def train_model(
    train_data_path: str,
    target_column: str,
    model_output_path: str,
    gcs_output_path: str
) -> None:
    """Train LightGBM model with SMOTEENN for class imbalance and save the model to GCS."""
    
    # 1. Load data
    df = pd.read_csv(train_data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 2. Apply SMOTEENN to balance data
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    # 3. Train LightGBM model
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        random_state=42,
        n_estimators=100,
        learning_rate=0.05,
    )
    lgb_model.fit(X_resampled, y_resampled)

    # 4. Save and upload model
    joblib.dump(lgb_model, model_output_path)
    upload_to_gcs(model_output_path, gcs_output_path)

def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a local file to a GCS path."""
    client = storage.Client()
    bucket_name, blob_path = parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

def parse_gcs_path(gcs_path: str) -> tuple:
    """Parse GCS path into bucket name and blob path."""
    assert gcs_path.startswith("gs://")
    parts = gcs_path[5:].split("/", 1)
    return parts[0], parts[1]
