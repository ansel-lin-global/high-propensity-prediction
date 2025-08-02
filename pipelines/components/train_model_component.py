@component(
    packages_to_install=["pandas", "lightgbm", "imbalanced-learn", "google-cloud-storage", "joblib"],
    base_image="python:3.10"
)
def train_model_component(
    train_data_path: str,
    target_column: str,
    model_output_path: str,
    gcs_output_path: str
):
    from model_training import train_model
    train_model(train_data_path, target_column, model_output_path, gcs_output_path)
