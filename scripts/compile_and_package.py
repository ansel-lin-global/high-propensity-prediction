"""
Compile & package Vertex AI Pipelines (showcase)
- Compiles pipelines/* into JSON/YAML specs under artifacts/
- File names include git sha and UTC timestamp for traceability
"""

import argparse, os, subprocess
from datetime import datetime
from kfp import compiler

# === Import your pipelines ===
from pipelines.training_pipeline import training_pipeline
from pipelines.predict_pipeline import daily_predict_pipeline
from pipelines.drift_pipeline import (
    run_data_drift_analysis_pipeline,
    run_concept_drift_analysis_pipeline,
    full_drift_analysis_pipeline,
)
from pipelines.retrain_pipeline import daily_drift_check_and_retrain

PIPELINES = {
    "training": training_pipeline,
    "predict": daily_predict_pipeline,
    "drift-data": run_data_drift_analysis_pipeline,
    "drift-concept": run_concept_drift_analysis_pipeline,
    "drift-full": full_drift_analysis_pipeline,
    "retrain": daily_drift_check_and_retrain,
}

def git_sha():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def stamp(name):  # training-20250101-120000-abc123.json
    return f"{name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{git_sha()}.json"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(PIPELINES.keys()))
    p.add_argument("--out-dir", default="artifacts")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = [args.only] if args.only else list(PIPELINES.keys())

    for key in targets:
        out = os.path.join(args.out_dir, stamp(key))
        compiler.Compiler().compile(pipeline_func=PIPELINES[key], package_path=out)
        print(f"[COMPILED] {key} -> {out}")

if __name__ == "__main__":
    main()
