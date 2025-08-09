"""
Vertex AI Pipelines — Drift Monitoring (Data & Concept)

What these pipelines showcase:
- Data Drift: compare incoming features vs. training baseline (e.g., PSI) and weight by feature importance
- Concept Drift: monitor recall@k, score distribution, and label rate shifts
- Full drift monitoring: run both and persist results for decisioning

Notes:
- Anonymized for showcase; replace projects, tables, and buckets with your own.
- Emphasizes modular design: each step is a component; pipelines stitch them into repeatable routines.
"""

from kfp import dsl
from kfp.dsl import pipeline

from components.utils import fetch_raw_data, write_to_bq
from components.drift import (
    load_latest_model_from_gcs,
    load_latest_json_from_gcs,
    extract_feature_importance_from_model,
    detect_feature_drift,
    evaluate_drift_and_retrain,
    calculate_anchor_date,
    compute_model_performance_drift,
    compute_prediction_score_drift,
    compute_label_distribution_drift,
    merge_and_evaluate_concept_drift
)

@pipeline(name="drift-analysis-pipeline", description="Data drift detection via feature monitoring")
def run_data_drift_analysis_pipeline(
    bq_project: str,
    train_query: str,
    predict_query: str,
    gcs_project: str,
    model_bucket: str,
    model_folder: str,
    table_id: str,
    write_mode: str,
    top_n: int = 40,
    psi_bins: int = 10,
    unseen_threshold: float = 0.01
):
    # Fetch baseline (train) and current (predict) feature data
    train_op = fetch_raw_data(project=bq_project, query=train_query)
    predict_op = fetch_raw_data(project=bq_project, query=predict_query)

    # Pull latest exported model & feature lists for alignment
    model_op = load_latest_model_from_gcs(project=gcs_project, bucket_name=model_bucket, folder_name=model_folder)
    cat_op = load_latest_json_from_gcs(project=gcs_project, bucket_name=model_bucket, folder_name=model_folder, suffix_type="cat")
    num_op = load_latest_json_from_gcs(project=gcs_project, bucket_name=model_bucket, folder_name=model_folder, suffix_type="numeric")
    importance_op = extract_feature_importance_from_model(model_file=model_op.outputs["output_model"])

    # Detect drift on top-N important features
    drift_op = detect_feature_drift(
        train_data=train_op.outputs["output_dataset"],
        predict_data=predict_op.outputs["output_dataset"],
        model_file=model_op.outputs["output_model"],
        cat_json_file=cat_op.outputs["output_json"],
        num_json_file=num_op.outputs["output_json"],
        top_n=top_n,
        psi_bins=psi_bins,
        unseen_threshold=unseen_threshold
    )

    # Weight drift by importance & recommend action
    evaluate_op = evaluate_drift_and_retrain(
        importance_data=importance_op.outputs["importance_df"],
        drift_data=drift_op.outputs["drift_result"],
        top_n=top_n
    )

    write_to_bq(
        project=bq_project,
        data=evaluate_op.outputs["output_result"],
        table_id=table_id,
        write_mode=write_mode
    )

@pipeline(name="concept-drift-analysis-pipeline", description="Concept drift detection via performance & score trends")
def run_concept_drift_analysis_pipeline(
    compare_to_date: str,
    bq_project: str,
    predict_table: str,
    backtesting_table: str,
    table_id: str,
    write_mode: str
):
    # Anchor date is typically "yesterday" in production; simplified here
    anchor_date_op = calculate_anchor_date()

    perf_op = compute_model_performance_drift(
        predict_date_str=anchor_date_op.output,
        project=bq_project,
        predict_table=predict_table,
        backtesting_table=backtesting_table
    )

    score_op = compute_prediction_score_drift(
        project=bq_project,
        predict_table=predict_table,
        current_date=anchor_date_op.output,
        compare_to_date=compare_to_date
    )

    label_op = compute_label_distribution_drift(
        project=bq_project,
        date=anchor_date_op.output
    )

    merge_op = merge_and_evaluate_concept_drift(
        model_drift=perf_op.outputs["output"],
        score_drift=score_op.outputs["output"],
        label_drift=label_op.outputs["output"]
    )

    write_to_bq(
        project=bq_project,
        data=merge_op.outputs["output_result"],
        table_id=table_id,
        write_mode=write_mode
    )

@pipeline(name="drift-detection-pipeline", description="Combined data + concept drift monitoring")
def full_drift_analysis_pipeline(
    bq_project: str,
    train_query: str,
    predict_query: str,
    gcs_project: str,
    model_bucket: str,
    model_folder: str,
    top_n: int,
    psi_bins: int,
    unseen_threshold: float,
    data_drift_table_id: str,
    concept_drift_table_id: str,
    compare_to_date: str,
    predict_table: str,
    backtesting_table: str,
    write_mode: str
):
    run_data_drift_analysis_pipeline(
        bq_project=bq_project,
        train_query=train_query,
        predict_query=predict_query,
        gcs_project=gcs_project,
        model_bucket=model_bucket,
        model_folder=model_folder,
        table_id=data_drift_table_id,
        write_mode=write_mode,
        top_n=top_n,
        psi_bins=psi_bins,
        unseen_threshold=unseen_threshold
    )

    run_concept_drift_analysis_pipeline(
        compare_to_date=compare_to_date,
        bq_project=bq_project,
        predict_table=predict_table,
        backtesting_table=backtesting_table,
        table_id=concept_drift_table_id,
        write_mode=write_mode
    )
