"""Run a command from this repo on a Daytona GPU sandbox.

One `run` = one ephemeral GPU box: tar this checkout (git-tracked plus
untracked-but-not-ignored files, so local edits ride along and `.env` never
does), create the box, pip install a requirements file, run one shell
command detached in the repo root, poll its log, pull back the logs plus
any --download paths into daytona_runs/<run-id>/, then stop the box (an
ephemeral box is removed when stopped; this key may not delete boxes).
With no --cmd it runs the GR-HSTU GPU smoke (scripts/daytona/smoke_train.py).
See scripts/daytona/README.md for the runbook.

Cost guards: the box gets a hard wall-clock TTL (--max-minutes) and is
refused if that TTL at table rates exceeds --max-cost; an idle auto-stop
backs that up. The key (DAYTONA_KEY) comes from the environment or the
repo .env, and is never logged or sent to the box.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "daytona_runs"
LEDGER = RUNS_DIR / "ledger.jsonl"
PROJECT_LABEL = "generative-recommenders"

IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
REQUIREMENTS = "scripts/daytona/requirements.txt"
SMOKE_CMD = "python scripts/daytona/smoke_train.py --steps 20"

BOX_DIR = "/workspace/gr"
BOX_REPO = f"{BOX_DIR}/repo"
BOX_OUT = f"{BOX_DIR}/out"
# Poll stops this long before the TTL so the results can still be pulled.
DOWNLOAD_MARGIN_MIN = 10
IDLE_STOP_MIN = 60

#: Public Daytona rates ($/h, daytona.io/pricing, preemptible column) as
#: recorded in aai-labs-signal_foundry sf/daytona.py on 2026-10-20. AAI
#: billing may differ: treat every $ figure here as an estimate.
GPU_PER_H = {
    "B300": 4.08,
    "B200": 3.59,
    "MI355X": 3.44,
    "H200": 2.61,
    "H100": 2.27,
    "RTX_PRO_6000": 1.74,
    "RTX_5090": 0.74,
    "RTX_4090": 0.57,
}
CPU_PER_H = 0.0504
MEM_PER_GBH = 0.0162
DISK_PER_GBH = 0.000108
DISK_FREE_GB = 5


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def client():
    from daytona import Daytona, DaytonaConfig
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
    key = os.environ.get("DAYTONA_KEY", "").strip()
    if not key:
        raise SystemExit("DAYTONA_KEY is not set (environment or repo .env)")
    return Daytona(DaytonaConfig(api_key=key))


def hourly_rate(args) -> float:
    return (
        args.gpus * GPU_PER_H[args.gpu_type]
        + args.cpu * CPU_PER_H
        + args.memory * MEM_PER_GBH
        + max(0, args.disk - DISK_FREE_GB) * DISK_PER_GBH
    )


def make_tarball(dest: Path) -> int:
    """Tar the checkout as git sees it: tracked + untracked, minus ignored."""
    names = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    n = 0
    with tarfile.open(dest, "w:gz") as tar:
        for raw in names:
            name = raw.decode()
            path = REPO_ROOT / name
            # .env is gitignored already; refuse it by name too.
            if not name or Path(name).name.startswith(".env") or not path.is_file():
                continue
            tar.add(path, arcname=name)
            n += 1
    return n


def box_script(cmd: str, requirements: str | None) -> str:
    """Shell script run detached on the box; writes DONE when finished."""
    # PyPI from the box can stall (seen 2026-10-28): long timeouts, 3 attempts.
    pip = (
        f"""for i in 1 2 3; do
    pip install --timeout 60 --retries 10 -r {shlex.quote(requirements)} >> {BOX_OUT}/pip.log 2>&1 && return 0
    echo "pip attempt $i failed" >> {BOX_OUT}/pip.log; sleep 30
  done
  return 1"""
        if requirements
        else "true"
    )
    return f"""#!/bin/bash
cd {BOX_REPO}
export PYTHONPATH={BOX_REPO} PYTHONUNBUFFERED=1
nvidia-smi > {BOX_OUT}/nvidia-smi.txt 2>&1
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \\
  --format=csv -l 30 > {BOX_OUT}/gpu.csv 2>&1 &
SMI=$!
install() {{
  {pip}
}}
if install; then
  bash -c {shlex.quote(cmd)} > {BOX_OUT}/run.log 2>&1
  rc=$?
else
  rc=$?
  echo "pip install failed (rc=$rc), see pip.log" > {BOX_OUT}/run.log
fi
echo $rc > {BOX_OUT}/exit_code
kill $SMI
echo DONE > {BOX_DIR}/DONE
"""


def sh(sb, cmd: str, timeout_s: int = 600) -> str:
    """Run cmd on the box; return stdout, raise on non-zero exit."""
    r = sb.process.exec(cmd, timeout=timeout_s)
    if r.exit_code not in (0, None):
        raise RuntimeError(f"exit {r.exit_code}: {(r.result or '')[-2000:]}")
    return r.result or ""


def poll(sb, deadline: float, poll_s: int) -> str:
    """Wait for DONE; return DONE, DEADLINE, or LOST (box unreachable)."""
    failures = 0
    while time.time() < deadline:
        try:
            done = sh(sb, f"cat {BOX_DIR}/DONE 2>/dev/null || true").strip()
            tail = sh(
                sb,
                f"cd {BOX_OUT} && f=$(ls -t run.log pip.log 2>/dev/null | head -1);"
                ' [ -n "$f" ] && tail -n 1 "$f"; true',
            )
            failures = 0
            if done == "DONE":
                return "DONE"
            log(f"running | {tail.strip()[-200:]}")
        except Exception as e:  # noqa: BLE001 - transient API errors
            failures += 1
            log(f"poll failed ({failures}/5): {str(e)[:160]}")
            if failures >= 5:
                return "LOST"
        time.sleep(poll_s)
    return "DEADLINE"


def fetch_results(sb, download: list[str], local: Path) -> None:
    """Tar out/ plus the existing --download paths on the box; pull and unpack."""
    parts = [f"tar cf {BOX_DIR}/results.tar -C {BOX_DIR} out"]
    for p in download:
        q = shlex.quote(p)
        parts.append(f"if [ -e {q} ]; then tar rf {BOX_DIR}/results.tar {q}; fi")
    parts.append(f"gzip -f {BOX_DIR}/results.tar")
    sh(sb, f"cd {BOX_REPO} && " + " && ".join(parts), timeout_s=1800)
    archive = local / "results.tar.gz"
    sb.fs.download_file(f"{BOX_DIR}/results.tar.gz", str(archive), 3600)
    with tarfile.open(archive) as tf:
        tf.extractall(local, filter="data")
    log(f"results: {archive.stat().st_size / 1e6:.1f} MB unpacked into {local}")
    archive.unlink()


def cmd_run(args) -> int:
    from daytona import CreateSandboxFromImageParams, GpuType, Resources, SessionExecuteRequest

    if args.max_minutes <= DOWNLOAD_MARGIN_MIN + 5:
        raise SystemExit(f"--max-minutes must exceed {DOWNLOAD_MARGIN_MIN + 5}")
    rate = hourly_rate(args)
    worst = rate * args.max_minutes / 60
    log(f"rate ~${rate:.2f}/h; worst case ~${worst:.2f} ({args.max_minutes} min TTL)")
    if worst > args.max_cost:
        raise SystemExit(f"worst case ${worst:.2f} exceeds --max-cost ${args.max_cost:.2f}")

    run_id = f"{re.sub(r'[^a-z0-9-]+', '-', args.name.lower())}-{time.strftime('%Y%m%d-%H%M%S')}"
    local = RUNS_DIR / run_id
    local.mkdir(parents=True, exist_ok=True)
    cmd = args.cmd or SMOKE_CMD
    requirements = None if args.requirements == "none" else args.requirements
    script = local / "run.sh"
    script.write_text(box_script(cmd, requirements))
    code = local / "code.tar.gz"
    n_files = make_tarball(code)
    (local / "request.json").write_text(
        json.dumps({**vars(args), "cmd": cmd, "run_id": run_id}, indent=1, default=str)
    )
    log(f"run {run_id}: {n_files} files ({code.stat().st_size / 1e6:.1f} MB); cmd: {cmd}")

    d = client()
    params = CreateSandboxFromImageParams(
        name=f"gr-{run_id}",
        image=args.image,
        resources=Resources(
            cpu=args.cpu,
            memory=args.memory,
            disk=args.disk,
            gpu=args.gpus,
            gpu_type=GpuType[args.gpu_type],
        ),
        labels={"user": args.user, "project": PROJECT_LABEL, "purpose": args.name},
        ephemeral=True,
        ttl_minutes=args.max_minutes,
        auto_stop_interval=IDLE_STOP_MIN,
    )
    t_create = time.time()
    status, exit_code, error, sb = "STARTING", None, None, None
    try:
        log(f"creating {args.gpus}x {args.gpu_type} box from {args.image} (can take minutes)")
        sb = d.create(params, timeout=args.create_timeout)
        log(f"box {sb.name} ({sb.id}) started after {time.time() - t_create:.0f}s")
        sh(sb, f"mkdir -p {BOX_REPO} {BOX_OUT}")
        sb.fs.upload_file(str(code), f"{BOX_DIR}/code.tar.gz")
        sb.fs.upload_file(str(script), f"{BOX_DIR}/run.sh")
        sh(sb, f"tar xzf {BOX_DIR}/code.tar.gz -C {BOX_REPO}")
        sb.process.create_session("run")
        sb.process.execute_session_command(
            "run",
            SessionExecuteRequest(
                command=f"bash {BOX_DIR}/run.sh > {BOX_DIR}/run.out 2>&1", run_async=True
            ),
        )
        log(f"launched; polling every {args.poll}s")
        deadline = t_create + (args.max_minutes - DOWNLOAD_MARGIN_MIN) * 60
        status = poll(sb, deadline, args.poll)
        if status != "LOST":
            fetch_results(sb, args.download, local)
            rc = local / "out" / "exit_code"
            exit_code = int(rc.read_text()) if rc.exists() else None
    except BaseException as e:
        error = f"{type(e).__name__}: {str(e)[:500]}"
        status = "CREATE_FAILED" if sb is None else "ERROR"
        log(f"error: {error}")
        raise
    finally:
        if sb is not None:
            try:
                name = sb.name
                sb.stop()
                log(f"stopped box {name} (ephemeral: removed)")
            except Exception as e:  # noqa: BLE001 - TTL/auto-stop are the backstop
                log(f"stop failed ({e}); TTL and idle auto-stop will reclaim it")
        minutes = (time.time() - t_create) / 60
        row = {
            "run_id": run_id,
            "status": status,
            "exit_code": exit_code,
            "gpu": f"{args.gpus}x{args.gpu_type}",
            "minutes": round(minutes, 1),
            "est_cost_usd": round(rate * minutes / 60, 2),
            "cmd": cmd,
            "error": error,
        }
        RUNS_DIR.mkdir(exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")
        (local / "summary.json").write_text(json.dumps(row, indent=1))
        log(f"summary: {json.dumps(row)}")

    run_log = local / "out" / "run.log"
    if run_log.exists():
        print("---- tail of run.log ----", flush=True)
        print("\n".join(run_log.read_text(errors="replace").splitlines()[-40:]), flush=True)
    return 0 if status == "DONE" and exit_code == 0 else 1


def mine(d, user: str) -> list:
    from daytona import ListSandboxesQuery

    boxes = d.list(ListSandboxesQuery(labels={"user": user, "project": PROJECT_LABEL}))
    return [b for b in boxes if "DESTROY" not in str(b.state).upper()]


def cmd_list(args) -> int:
    boxes = mine(client(), args.user)
    for b in boxes:
        print(f"{b.name}  {b.id}  {b.state}  {b.labels.get('purpose', '')}", flush=True)
    print(f"{len(boxes)} box(es) labeled user={args.user} project={PROJECT_LABEL}", flush=True)
    return 0


def cmd_stop(args) -> int:
    d = client()
    boxes = mine(d, args.user) if args.all else [d.get(args.id)]
    for b in boxes:
        name = b.name
        b.stop()
        print(f"stopped {name} ({b.id})", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default=getpass.getuser(), help="'user' label (required by the shared account)")
    sub = ap.add_subparsers(dest="sub", required=True)

    r = sub.add_parser("run", help="run one command on a fresh GPU box")
    r.add_argument("--name", default="smoke", help="short run name; prefixes the run id")
    r.add_argument("--cmd", help=f"shell command run in the repo root (default: {SMOKE_CMD!r})")
    r.add_argument("--requirements", default=REQUIREMENTS, help="repo-relative pip requirements file, or 'none'")
    r.add_argument("--download", nargs="*", default=[], help="repo-relative paths on the box to pull back")
    r.add_argument("--max-minutes", type=int, default=60, help="hard TTL of the box, minutes")
    r.add_argument("--max-cost", type=float, default=25.0, help="refuse if TTL x rate exceeds this ($)")
    r.add_argument("--gpu-type", default="H100", choices=sorted(GPU_PER_H))
    r.add_argument("--gpus", type=int, default=1)
    r.add_argument("--cpu", type=int, default=8)
    r.add_argument("--memory", type=int, default=32, help="GiB")
    r.add_argument("--disk", type=int, default=50, help="GiB")
    r.add_argument("--image", default=IMAGE)
    r.add_argument("--poll", type=int, default=60, help="seconds between log polls")
    r.add_argument("--create-timeout", type=int, default=1200, help="seconds for image pull + GPU queueing")

    sub.add_parser("list", help="list my boxes for this project")
    s = sub.add_parser("stop", help="stop (and so remove) a box, or all of mine")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--id")
    g.add_argument("--all", action="store_true")

    args = ap.parse_args(argv)
    # nohup/kill sends SIGTERM: turn it into SystemExit so the box is stopped.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    return {"run": cmd_run, "list": cmd_list, "stop": cmd_stop}[args.sub](args)


if __name__ == "__main__":
    sys.exit(main())
