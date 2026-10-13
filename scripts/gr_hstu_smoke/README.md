# GR-HSTU SageMaker GPU smoke

Runs a minimal GPU training job on SageMaker to prove the whole chain works:
IAM role, image pull, code upload to S3, `ml.g5.xlarge` compute, and CloudWatch
logging. The job trains a tiny HSTU + sampled-softmax model for a few steps on
synthetic data, then prints `SMOKE OK`. Not a paper reproduction.

Proven working 2026-10-13: `gr-hstu-smoke-20261013-145521` → `Completed`.

## Files

- `scripts/submit_gr_hstu_sagemaker.py` — submission script. Tars the code,
  uploads to S3, calls `CreateTrainingJob`.
- `scripts/gr_hstu_smoke/smoke_train.py` — the training script that runs inside
  the container.
- `scripts/gr_hstu_smoke/requirements.txt` — pip deps installed in the
  container at runtime. **Pins are load-bearing** (see below).

## Prerequisites

- A devserver with the `cloud` CLI (`cloud aws ...`).
- This repo checked out, on branch `nwr-2026-10-13`, with `.venv`
  (needs `boto3`; this repo's `.venv` has it).
- A checkout of the internal `aai-labs-signal_foundry` repo, used only for
  the `sf.iex_ml_sagemaker` submission helper. Default location
  `~/aai-labs-signal_foundry`; override with `SIGNAL_FOUNDRY_ROOT`.
- The `generative_recommenders/` package tarred into the job comes from this
  repo itself.
- Target AWS account `254587346374`, region `us-west-2`.

## Step 1 — credentials

SageMaker/boto3 calls use ambient env credentials from the `cloud` CLI
(valid 1 hour; re-run when they expire):

```bash
eval "$(cloud aws get-credentials 254587346374 -o env)"
aws sts get-caller-identity  # expect Account 254587346374
```

## Step 2 — submit

From the repo root:

```bash
.venv/bin/python scripts/submit_gr_hstu_sagemaker.py \
  --image 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.4-gpu-py311 \
  --role arn:aws:iam::254587346374:role/signal-foundry-gr-sagemaker-execution-us-west-2
```

This prints the S3 code URI, the job name (`gr-hstu-smoke-<timestamp>`), and
the job ARN. Everything lands under `s3://aai-sf-datasets-prod-us-west-2/v1/_scratch/gr-hstu/<job>/`.
Defaults: `--steps 5`, `ml.g5.xlarge`, 30 GB volume, 1 h max runtime.
Pass `--wait` to block until `Completed` (raises with the `FailureReason` on
failure).

| Parameter | Value |
|---|---|
| Image | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.4-gpu-py311` (torch 2.4.0+cu124, py311) |
| Role | `arn:aws:iam::254587346374:role/signal-foundry-gr-sagemaker-execution-us-west-2` |
| Instance | `ml.g5.xlarge` (1x NVIDIA A10G) |
| Bucket/prefix | `s3://aai-sf-datasets-prod-us-west-2/v1/_scratch/gr-hstu/` |

## Step 3 — monitor

```bash
JOB=gr-hstu-smoke-<timestamp>  # from the submit output
aws sagemaker describe-training-job --training-job-name $JOB --region us-west-2 \
  --query '{Status:TrainingJobStatus, Secondary:SecondaryStatus, FailureReason:FailureReason}' --output json
aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix $JOB --region us-west-2 \
  --query 'logStreams[*].logStreamName' --output text
aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name "$JOB/algo-1-<id>" --region us-west-2 --limit 100 \
  --query 'events[*].message' --output text | tr '\t' '\n' | grep -E 'torch=|cuda_available|device=|METRIC|SMOKE OK|SUCCESS|Error'
```

Success looks like: status `Completed`, plus in the logs
`torch=2.4.0+cu124`, `cuda_available=True`, `device=NVIDIA A10G`,
`[METRIC] step=N loss=...` lines with decreasing loss, `SMOKE OK`, and
`Reporting training SUCCESS`. Provisioning + `fbgemm` wheel download
(~330 MB) takes several minutes; `InProgress` with no log streams yet is
normal in the first minutes.

## Reference — image URIs

The image URI comes from the SageMaker SDK lookup. Import path depends on
SDK major version (v3 moved it to `sagemaker.core`):

```bash
# SDK v3 (3.x installed on this host's system python)
python3 -c "from sagemaker.core import image_uris; print(image_uris.retrieve('pytorch', region='us-west-2', version='2.4', image_scope='training', instance_type='ml.g5.xlarge'))"
# SDK v2 — original import path still works there:
#   uv run --no-project --with 'sagemaker==2.257.7' python -c "from sagemaker import image_uris; ..."
```

Note the framework is `'pytorch'` with `image_scope='training'` —
`'pytorch-training'` is the ECR *repository* name, not a valid framework.

## Reference — why requirements.txt is pinned

`fbgemm-gpu` tracks PyTorch releases with no forward/backward ABI guarantee;
an unpinned install pulls the newest wheel, which segfaults/`undefined symbol`
against the container's torch (`torch::Library::_def`). Per the
[FBGEMM compatibility table](https://docs.pytorch.org/FBGEMM/general/Releases.html):

| torch (container) | fbgemm-gpu |
|---|---|
| 2.4.x | 0.8.0 |
| 2.5.x | 1.0.0 |
| 2.6.x | 1.1.0 (= this repo's own `torch>=2.6` / `fbgemm_gpu>=1.1` pins) |

`gin-config` is needed because the GR research modules do `import gin`.
If you move to a different framework version, re-pin `fbgemm-gpu` to match.

## Troubleshooting

- `FailureReason` is the source of truth; fetch it verbatim via the
  `describe-training-job` query above.
- `undefined symbol: ... torch::Library::_def` in `fbgemm_gpu*.so` →
  fbgemm/torch version mismatch; fix the pin, don't touch infra.
- `ModuleNotFoundError` for a third-party import → add it to
  `requirements.txt` (container only has torch + SageMaker toolkit).
- `NoCredentials` / `ExpiredToken` → re-run the `cloud aws get-credentials`
  eval (1 h lifetime).
- Cost: `ml.g5.xlarge` is ~$1/hr; a smoke run takes ~10 min. The 1 h
  `MaxRuntimeInSeconds` is a runaway guard.
