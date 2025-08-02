"""
Component: Retrain Trigger (Vertex AI Pipeline)

📌 Purpose:
Dynamically launch retraining pipeline on Vertex AI based on config

🧾 Input:
- Precompiled training pipeline (YAML on GCS)
- Parameter overrides (BQ table, prediction window, export location, etc.)

📤 Output:
- Training job triggered on Vertex AI
"""

from kfp.dsl import component

@component(
    base_image='python:3.10',
    packages_to_install=["google-cloud-aiplatform"]
)
def trigger_training_pipeline(
    project: str,
    region: str,
    pipeline_uri: str,
    pipeline_params: dict,
    staging_bucket: str,
    service_account: str,
    encryption_key: str
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)

    job = aiplatform.PipelineJob(
        display_name="retrain-pipeline",
        template_path=pipeline_uri,
        pipeline_root=f"{staging_bucket}/pipeline_root",
        enable_caching=False,
        parameter_values=pipeline_params,
        encryption_spec_key_name=encryption_key
    )

    job.run(service_account=service_account, sync=False)
