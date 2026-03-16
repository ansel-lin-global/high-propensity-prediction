"""
Vertex AI Pipeline — Training (Sliding Windows, Multi-Model)

What this pipeline showcases:
- Split data into rolling time windows with a leakage-safe gap
- Train LightGBM / XGBoost / CatBoost **in parallel** per window
- Evaluate each model, aggregate metrics to BigQuery, and export the daily best model to GCS

Notes for reviewers:
- This is a sanitized, showcase-only pipeline. All GCP project IDs, table names,
  and bucket paths must be supplied as pipeline parameters.
- The pipeline emphasizes orchestration (ParallelFor, If, fan-out/fan-in)
  and production concerns (schema tracking, model versioning).
- Some imported components are NOT included in the published components/ folder.
  They are listed below for structural completeness — see components/README.md
  for descriptions of each.
"""

from kfp import dsl
from kfp.dsl import pipeline

# ── Published components (included in components/train.py) ──────────────
from components.train import (
    split_data_by_time_series,
    count_total_windows,
    train_lgb_model,
)

# ── Production-only components (not published; see components/README.md) ─
# These imports would resolve in the full production codebase.
# They are kept here to show the complete pipeline orchestration.
#
# from components.utils import fetch_raw_data
# from components.train import (
#     inspect_schema,
#     store_schema_features,
#     resample_data,
#     preprocess_data,
#     extract_window_data,
#     train_xgb_model,
#     train_catboost_model,
#     evaluate_model_to_file,
#     merge_and_write_to_bq,
#     summarize_eval_from_bq,
#     export_best_model,
#     model_eval_done,
#     wait_for_all_models,
# )

@pipeline(
    name="training-pipeline",
    description="Train and evaluate multiple models using sliding window cross-validation"
)
def training_pipeline(
    bq_project: str = "",          # GCP project for BigQuery operations
    fetch_raw_data_query: str = "",  # SQL query to fetch training data
    date_col: str = "date",
    gap: int = 3,
    prediction_window: int = 1,
    k: int = 40000,
    bq_table: str = "",            # Destination table for evaluation results
    selection_metric: str = "recall",
    gcs_project: str = "",         # GCP project for GCS operations
    export_bucket: str = "",       # GCS bucket for model artifacts
):
    """
    End-to-end training pipeline with sliding-window cross-validation.

    This pipeline:
    1. Fetches raw training data from BigQuery
    2. Inspects schema and persists feature metadata to GCS
    3. Creates rolling train/test splits with a configurable gap
    4. For each window, resamples, preprocesses, and trains 3 models in parallel
    5. Evaluates all models, selects the best, and exports to GCS

    All parameters must be supplied at runtime — there are no hardcoded
    project IDs, table names, or bucket paths.
    """
    # Step 1: Fetch raw data
    raw_data = fetch_raw_data(query=fetch_raw_data_query, project=bq_project)
    raw_data.set_cpu_limit('2')
    raw_data.set_memory_limit('32Gi')

    # Step 2: Inspect schema and store to GCS
    schema = inspect_schema(input_dataset=raw_data.outputs["output_dataset"])
    store_schema_features(
        numeric=schema.outputs["numeric"],
        cat=schema.outputs["cat"],
        output_bucket=export_bucket,
        project=gcs_project
    )

    # Step 3: Create sliding window splits
    splits = split_data_by_time_series(
        input_dataset=raw_data.outputs["output_dataset"],
        date_col=date_col,
        gap=gap,
        prediction_window=prediction_window
    )
    index_list = count_total_windows(splits_path=splits.outputs["output_splits_path"])

    # Step 4: Loop over each window
    with dsl.ParallelFor(index_list.output) as window_index:
        window_data = extract_window_data(
            input_dataset=raw_data.outputs["output_dataset"],
            splits_path=splits.outputs["output_splits_path"],
            window_index=window_index,
            numeric=schema.outputs["numeric"],
            cat=schema.outputs["cat"]
        )

        resampled = resample_data(
            X_path=window_data.outputs["X_train_path"],
            y_path=window_data.outputs["y_train_path"],
            numeric=schema.outputs["numeric"],
            cat=schema.outputs["cat"],
            do_smote=False,
            undersample_ratio=0.05,
            smote_ratio=0.1
        )

        preprocessed = preprocess_data(
            X_train_path=resampled.outputs["X_res_path"],
            X_test_path=window_data.outputs["X_test_path"],
            numeric=schema.outputs["numeric"],
            cat=schema.outputs["cat"]
        )

        done_signals = []

        # Step 5: Train and evaluate 3 models
        with dsl.ParallelFor(["lgb", "xgb", "catboost"]) as model_type:
            with dsl.If(model_type == "lgb"):
                train = train_lgb_model(
                    X_train_path=preprocessed.outputs["X_train_scaled_path"],
                    y_train_path=resampled.outputs["y_res_path"],
                    X_valid_path=preprocessed.outputs["X_test_scaled_path"],
                    y_valid_path=window_data.outputs["y_test_path"],
                    cat=schema.outputs["cat"]
                )
            with dsl.If(model_type == "xgb"):
                train = train_xgb_model(
                    X_train_path=preprocessed.outputs["X_train_scaled_path"],
                    y_train_path=resampled.outputs["y_res_path"],
                    X_valid_path=preprocessed.outputs["X_test_scaled_path"],
                    y_valid_path=window_data.outputs["y_test_path"],
                    cat=schema.outputs["cat"]
                )
            with dsl.If(model_type == "catboost"):
                train = train_catboost_model(
                    X_train_path=preprocessed.outputs["X_train_scaled_path"],
                    y_train_path=resampled.outputs["y_res_path"],
                    X_valid_path=preprocessed.outputs["X_test_scaled_path"],
                    y_valid_path=window_data.outputs["y_test_path"],
                    cat=schema.outputs["cat"]
                )

            eval_model = evaluate_model_to_file(
                model_path=train.outputs["model_output_path"],
                scaler_path=preprocessed.outputs["scaler_output_path"],
                X_test_path=preprocessed.outputs["X_test_scaled_path"],
                y_test_path=window_data.outputs["y_test_path"],
                cat=schema.outputs["cat"],
                k=k,
                model_type=model_type,
                window_index=window_index,
                bucket_name=export_bucket
            ).after(train)

            done = model_eval_done(model_type=model_type).after(eval_model)
            done_signals.append(done)

    # Step 6: Wait until all model evals complete
    wait = wait_for_all_models()
    for done in done_signals:
        wait.after(done)

    # Step 7: Aggregate evaluation results and export best model
    merged = merge_and_write_to_bq(
        result_root_path=f'/gcs/{export_bucket}/eval_results',
        project=bq_project,
        bq_table=bq_table
    ).after(wait)

    summarized = summarize_eval_from_bq(
        project=bq_project,
        bq_table=bq_table
    ).after(merged)

    export_best_model(
        project=bq_project,
        bq_table=bq_table,
        selection_metric=selection_metric,
        gcs_project=gcs_project,
        export_bucket=export_bucket
    ).after(summarized)
