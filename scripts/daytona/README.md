# GPU runs on Daytona

This devserver has no GPU. `scripts/daytona/run.py` runs any shell command
from this checkout on an ephemeral Daytona GPU sandbox ("box") and brings the
logs and outputs back. With no `--cmd` it runs a GPU smoke: a tiny HSTU +
sampled-softmax model trained for a few steps on synthetic data, ending in
`SMOKE OK`. The smoke is not a paper reproduction.

Proven working 2026-10-28: run `smoke-20261028-120847` on 1x H100 80GB HBM3
(torch 2.6.0+cu124, fbgemm-gpu 1.1.0): loss 8.83 to 0.91 over 20 steps, then
`SMOKE OK`. The box started in 15 s and the run took 1.3 min (about $0.07).
A separate check that day imported `main.py`, the data preprocessor and
TensorBoard's `SummaryWriter` on a box with the default requirements: numpy
2.2.2, pandas 2.3.3, tensorboard 2.21.0, absl-py 2.5.1, gin-config 0.5.0.
The same day's tests also covered `--download`, a failing command (exit code
reported, runner exits 1), and cleanup on SIGTERM.

## Files

- `scripts/daytona/run.py`: the runner (`run`, `list`, `stop`).
- `scripts/daytona/smoke_train.py`: the smoke training script (runs on the box).
- `scripts/daytona/requirements.txt`: pip packages installed on the box before
  the command runs: what the repo's research code imports that the image
  lacks. The pins matter; see the FBGEMM section below.

## Prerequisites

- `DAYTONA_KEY` in the repo's `.env` (gitignored) or in the environment.
  The runner reads it, never prints it, and never sends it to the box.
- This repo's `.venv` with the Daytona SDK. It is already installed on this
  devserver. On a fresh checkout:
  `uv pip install --python .venv/bin/python 'daytona==0.225.0' python-dotenv`.
- No AWS credentials and no other checkout are needed.

## Step 1: smoke

From the repo root:

```bash
.venv/bin/python scripts/daytona/run.py run
```

The runner tars the checkout, creates a 1x H100 box, installs
`scripts/daytona/requirements.txt`, runs the smoke, polls every 60 s, pulls the
results into `daytona_runs/<run-id>/`, and stops the box. A stopped box is
removed, because boxes are created ephemeral. Boxes started in 2 to 16 s in
testing. A new `--image` can take longer while Daytona pulls it, and GPU
capacity is not guaranteed.

Success looks like this: the summary line shows `"status": "DONE"` and
`"exit_code": 0`, and the tail of `run.log` shows `cuda_available=True`,
`device=NVIDIA H100 ...`, `[METRIC] step=N loss=...` lines, and `SMOKE OK`.
The process exits 0.

## Step 2: run your own command

`--cmd` is any shell command. It runs in the repo root on the box
(`/workspace/gr/repo`) with `PYTHONPATH` set to the repo root. The box gets the
checkout as git sees it: tracked files plus untracked files that are not
ignored. Local edits are included without committing; `.env` and ignored
paths (`tmp/`, `exps/`, `ckpts/`, `.venv`) are not uploaded. On the devserver
the runner writes only under `daytona_runs/` in the repo.

```bash
mkdir -p daytona_runs && nohup .venv/bin/python scripts/daytona/run.py run \
  --name <short-name> \
  --cmd "<command 1> && <command 2>" \
  --requirements <repo-relative requirements file, or none> \
  --download <repo-relative paths to pull back, e.g. output dirs> \
  --max-minutes <hard limit> --max-cost <dollars> \
  > daytona_runs/<short-name>.out 2>&1 &
```

- Anything the command writes that is not under `--download` is lost when the
  box is removed. The box's disk is 50 GiB by default (`--disk`).
- `--max-minutes` is a hard wall-clock limit for the box (default 60), counted
  from creation and enforced by Daytona even if the runner dies. The runner
  stops polling 10 minutes before the limit and pulls whatever exists, so set
  it well above the expected runtime plus setup.
- `--max-cost` (default $25) refuses to create a box whose limit times the
  hourly rate exceeds it.
- Run long jobs under `nohup` as above. Ctrl-C or `kill` (SIGTERM) stops the
  box without pulling results. `kill -9` leaves the box running until its limit.
- One run is one box. To run several things, either chain them in one `--cmd`
  or start several runners.

### What lands in `daytona_runs/<run-id>/`

| Path | Contents |
|---|---|
| `out/run.log` | stdout and stderr of `--cmd` |
| `out/pip.log` | the `pip install -r <requirements>` output |
| `out/exit_code` | exit code of `--cmd` (or of pip, if pip failed) |
| `out/nvidia-smi.txt`, `out/gpu.csv` | GPU at start; utilization and memory every 30 s |
| `<download paths>` | the `--download` paths, at their repo-relative locations |
| `request.json`, `run.sh`, `code.tar.gz`, `summary.json` | what was asked, the script and checkout the box ran, the outcome |

`daytona_runs/ledger.jsonl` keeps one line per run: status, exit code, GPU,
minutes, estimated cost, command.

## Box environment

| | Default | Flag |
|---|---|---|
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` | `--image` |
| GPU | 1x H100 80 GB | `--gpu-type`, `--gpus` |
| CPU / memory / disk | 8 cores / 32 GiB / 50 GiB | `--cpu`, `--memory`, `--disk` |
| Requirements | `scripts/daytona/requirements.txt` | `--requirements` |

Seen on the box on 2026-10-28: Python 3.11 (conda, `/opt/conda`), torch
2.6.0+cu124, NVIDIA driver 595.91.07, running as root. PyPI speed varies by
box: the 417 MB fbgemm-gpu wheel usually downloaded in about 40 s, took 8 min on
one box, and one box could not reach PyPI at all. The runner gives pip 60 s
timeouts and 3 attempts. Budget a few minutes of pip per run. pip installs into
the image's Python, so each run starts from the clean image.

## Monitoring and cleanup

The runner prints the last line of `run.log` (or `pip.log` while pip runs) at
every poll. To check for boxes left behind, for example after a `kill -9`:

```bash
.venv/bin/python scripts/daytona/run.py list
.venv/bin/python scripts/daytona/run.py stop --id <box-id>   # or: stop --all
```

`list` and `stop --all` only see boxes labeled with your user and this project.

## Cost

Rates are Daytona's public prices (estimates; AAI billing may differ). With the
default CPU, memory and disk, a 1x H100 box costs about $3.20/h, so the default
60-minute limit caps a run at about $3.20. A smoke run costs about $0.06. The
rate table is in `run.py` (`GPU_PER_H`).

## FBGEMM and torch versions

`fbgemm-gpu` has no ABI compatibility across PyTorch releases. A wheel built for
a different torch fails at import with `undefined symbol` (for example
`torch::Library::_def`). Pin it to the image's torch
([FBGEMM releases](https://docs.pytorch.org/FBGEMM/general/Releases.html)):

| torch | fbgemm-gpu |
|---|---|
| 2.4.x | 0.8.0 |
| 2.5.x | 1.0.0 |
| 2.6.x | 1.1.0 (this image; matches the repo's `torch>=2.6`, `fbgemm_gpu>=1.1`) |

The repo's top-level `requirements.txt` leaves torch unpinned (`torch>=2.6.0`),
so installing it on the box can upgrade torch past the image's version.
The default requirements file covers what the GR research code (`main.py`,
`preprocess_public_data.py`, `generative_recommenders/research/`) imports
beyond the image: gin-config, pandas (kept below 3, because that code predates
pandas 3's changed defaults), tensorboard and absl-py.

## Troubleshooting

- `"status": "CREATE_FAILED"`: the box never started. A timeout usually means
  no GPU capacity. Retry later or try another `--gpu-type`.
- Non-zero `exit_code`: read `out/run.log`, then `out/pip.log`.
- `pip install failed` with `Read timed out` in `out/pip.log`: the box could
  not reach PyPI. Rerun; the next box usually can.
- `ModuleNotFoundError` on the box: add the package to the requirements file.
- `DAYTONA_KEY is not set`: the repo `.env` is missing or lacks the key.
