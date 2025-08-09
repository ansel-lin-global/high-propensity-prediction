"""
Submit a Vertex AI PipelineJob (showcase)
- Reads compiled JSON/YAML
- Submits a run with parameter values
"""

import argparse, os
from datetime import datetime
from google.cloud import aiplatform

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--staging-bucket", required=True)          
    p.add_argument("--service-account", required=True)
    p.add_argument("--pipeline-spec", required=True)           # artifacts/training-*.json
    p.add_argument("--job-display-name", default="training-pipeline-job")
    p.add_argument("--enable-caching", action="store_true")
    p.add_argument("--encryption-key", default="")             # projects/.../cryptoKeys/...
    p.add_argument("--param", action="append", default=[], help="k=v pairs")
    args = p.parse_args()

    # parse params k=v
    params = {}
    for kv in args.param:
        k, v = kv.split("=", 1)

        if v.isdigit(): v = int(v)
        params[k] = v

    aiplatform.init(project=args.project, location=args.region, staging_bucket=args.staging_bucket)

    job = aiplatform.PipelineJob(
        display_name=args.job_display_name,
        template_path=args.pipeline_spec,
        pipeline_root=f"{args.staging_bucket}/pipeline_root",
        parameter_values=params,
        enable_caching=args.enable_caching,
        encryption_spec_key_name=args.encryption_key or None,
    )
    job.run(service_account=args.service_account, sync=False)
    print("[SUBMITTED]", args.job_display_name, "->", args.pipeline_spec)

if __name__ == "__main__":
    main()
