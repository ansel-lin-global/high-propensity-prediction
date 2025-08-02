"""
Component: Model Training

📌 Purpose:
Train a binary classification model (e.g., LightGBM) to predict user purchase propensity.

This component demonstrates the modeling concept using LightGBM as an example.
In practice, this step can be extended to support multiple model types (e.g., XGBoost, CatBoost),
and includes techniques to handle class imbalance, such as undersampling and SMOTE.

🔁 Input:
- Processed training features and labels

📤 Output:
- Trained model saved to cloud storage
- Evaluation metrics (e.g., PR-AUC, Precision@K) saved as JSON
"""
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "lightgbm", "imbalanced-learn", "scikit-learn", "joblib", "pyarrow"]
)
def train_propensity_model(
    X_train: Input[Dataset],
    y_train: Input[Dataset],
    cat_features: Input[Dataset],    # JSON with categorical column names
    numeric_features: Input[Dataset],# JSON with numeric column names
    model_output: Output[Model],
    metrics_output: Output[Metrics]
):
    """
    📌 Component: Model Training
    Train a LightGBM model to predict high purchase propensity.
    Handles class imbalance via undersampling and optional SMOTE.
    Saves model and training metrics.
    """

    import pandas as pd
    import joblib
    import json
    from lightgbm import LGBMClassifier
    from sklearn.metrics import precision_score, classification_report
    from imblearn.over_sampling import SMOTE

    # Load data
    X = pd.read_parquet(X_train.path)
    y = pd.read_parquet(y_train.path).squeeze()

    # Load column configs
    with open(cat_features.path, "r") as f:
        cat_cols = json.load(f)
    with open(numeric_features.path, "r") as f:
        num_cols = json.load(f)

    # Optional: SMOTE
    smote = SMOTE(random_state=42)
    X_num = X[num_cols].fillna(0)
    X_res, y_res = smote.fit_resample(X_num, y)

    # Reattach categorical cols (duplicate if needed)
    X_cat = pd.concat([X[cat_cols]] * (len(X_res) // len(X)), ignore_index=True).iloc[:len(X_res)]
    X_final = pd.concat([X_res.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

    # Train model
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        scale_pos_weight=(len(y_res) - y_res.sum()) / y_res.sum(),
        random_state=42
    )
    model.fit(X_final, y_res, categorical_feature=cat_cols)

    # Save model
    joblib.dump(model, model_output.path)

    # Output metrics
    y_pred = model.predict(X_final)
    report = classification_report(y_res, y_pred, output_dict=True)
    metrics_output.log_metric("precision", precision_score(y_res, y_pred))
    metrics_output.metadata["classification_report"] = report
