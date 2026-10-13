"""Submit the GR-HSTU GPU smoke job to SageMaker.

SMOKE ONLY: proves IAM + image pull + logging on ml.g5.xlarge with a few
synthetic training steps. Not a paper reproduction. Proven working
2026-10-13 (gr-hstu-smoke-20261013-145521 -> Completed, SMOKE OK).
See scripts/gr_hstu_smoke/README.md for the runbook.

Requires a checkout of the (internal) aai-labs-signal_foundry repo for the
sf.iex_ml_sagemaker submission helper; submits with ambient env boto3
credentials (`cloud aws get-credentials`).
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parents[1]
SF_ROOT = os.environ.get(
    "SIGNAL_FOUNDRY_ROOT", os.path.expanduser("~/aai-labs-signal_foundry")
)
sys.path.insert(0, SF_ROOT)
from sf.iex_ml_sagemaker import (  # noqa: E402
    TrainingInputChannel,
    TrainingJobSpec,
    job_name_for,
)

BUCKET = "aai-sf-datasets-prod-us-west-2"
REGION = "us-west-2"
PREFIX = "v1/_scratch/gr-hstu"
ROLE = (
    "arn:aws:iam::254587346374:role/"
    "signal-foundry-gr-sagemaker-execution-us-west-2"
)
IMAGE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "pytorch-training:2.4-gpu-py311"
)
GR_REPO = str(REPO_ROOT)
SMOKE_DIR = Path(__file__).resolve().parent / "gr_hstu_smoke"


def upload_code(job_name: str) -> tuple[str, str]:
    key = f"{PREFIX}/{job_name}/code.tar.gz"
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(f"{SMOKE_DIR}/smoke_train.py", arcname="smoke_train.py")
            tar.add(f"{SMOKE_DIR}/requirements.txt", arcname="requirements.txt")
            tar.add(f"{GR_REPO}/generative_recommenders", arcname="generative_recommenders")
        boto3.client("s3", region_name=REGION).upload_file(tmp.name, BUCKET, key)
    uri = f"s3://{BUCKET}/{key}"
    print(f"code -> {uri}", flush=True)
    return uri, f"s3://{BUCKET}/{PREFIX}/{job_name}/"


def wait(job_name: str, timeout_s: int = 1800) -> dict:
    sm = boto3.client("sagemaker", region_name=REGION)
    deadline = time.monotonic() + timeout_s
    while True:
        d = sm.describe_training_job(TrainingJobName=job_name)
        status = d["TrainingJobStatus"]
        print(f"  {status} / {d.get('SecondaryStatus', '')}", flush=True)
        if status == "Completed":
            return d
        if status in ("Failed", "Stopped"):
            raise RuntimeError(
                f"{job_name} ended {status}: {d.get('FailureReason', 'no reason')}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"{job_name} still {status} after {timeout_s}s")
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", default=ROLE)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--steps", default="5")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    job_name = job_name_for("gr-hstu-smoke")
    code_uri, code_prefix = upload_code(job_name)
    spec = TrainingJobSpec(
        job_name=job_name,
        image_uri=args.image,
        role_arn=args.role,
        features_s3_uri=None,
        output_s3_uri=f"s3://{BUCKET}/{PREFIX}/{job_name}/output/",
        hyperparameters={
            "sagemaker_program": "smoke_train.py",
            "sagemaker_submit_directory": code_uri,
            "steps": args.steps,
        },
        instance_type="ml.g5.xlarge",
        volume_size_gb=30,
        max_runtime_s=3600,
        metric_definitions=(
            {"MetricName": "loss", "Regex": r" loss=([0-9.]+)"},
        ),
        input_channels=(TrainingInputChannel("code", code_prefix),),
        component="gr-training",
        container_entrypoint=None,
        container_arguments=None,
    )
    sm = boto3.client("sagemaker", region_name=REGION)
    resp = sm.create_training_job(**spec.as_request())
    print(f"submitted {spec.job_name}", flush=True)
    print(f"  arn: {resp.get('TrainingJobArn')}", flush=True)
    if args.wait:
        final = wait(spec.job_name)
        print(f"  Completed: {final['ModelArtifacts']['S3ModelArtifacts']}", flush=True)


if __name__ == "__main__":
    main()
