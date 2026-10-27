#!/usr/bin/env python3
"""Render (and optionally submit) Slurm batch scripts for a GPU cluster.

Cluster settings and job definitions live in ``slurm/cluster.toml``; personal settings (the Slurm
account) in ``slurm/cluster.local.toml``, which is not committed and overrides ``[cluster]``.
Every job is rendered to ``slurm/generated/<job>.sbatch``; array jobs also get ``<job>.args``
with one argument line per array task, exposed to the command as ``$ARRAY_ARGS``.

    python3 slurm/generate.py --list
    python3 slurm/generate.py train                      # render only
    python3 slurm/generate.py train --submit             # render and sbatch
    python3 slurm/generate.py train --set time=0-02:00:00 --set gpus=2
    python3 slurm/generate.py --all

Only the standard library is used, so this runs with the cluster's bare Python module.
Render on the cluster itself: the scripts contain absolute paths to the project checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SLURM_DIR.parent
CONFIG_PATH = SLURM_DIR / "cluster.toml"
LOCAL_PATH = SLURM_DIR / "cluster.local.toml"
OUT_DIR = SLURM_DIR / "generated"
LOG_DIR = SLURM_DIR / "logs"

JOB_KEYS = {
    "description",
    "partition",
    "time",
    "cpus",
    "mem",
    "gpus",
    "constraint",
    "command",
    "array",
    "max_parallel",
    "env",
    "checkpoint_signal",
}
INT_KEYS = {"cpus", "gpus", "max_parallel"}
BOOL_KEYS = {"checkpoint_signal"}


def load_config(path: Path = CONFIG_PATH, local: Path = LOCAL_PATH) -> dict:
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if local.exists():  # personal [cluster] settings, e.g. the Slurm account
        with local.open("rb") as f:
            cfg["cluster"] = {**cfg.get("cluster", {}), **tomllib.load(f).get("cluster", {})}
    return cfg


def resolve_job(cfg: dict, name: str, overrides: dict[str, str] | None = None) -> dict:
    jobs = cfg.get("jobs", {})
    if name not in jobs:
        raise SystemExit(f"unknown job '{name}', available: {', '.join(sorted(jobs))}")
    job = {**cfg.get("defaults", {}), **jobs[name]}
    for key, value in (overrides or {}).items():
        if key in INT_KEYS:
            job[key] = int(value)
        elif key in BOOL_KEYS:
            job[key] = value.lower() in {"1", "true", "yes"}
        else:
            job[key] = value

    unknown = set(job) - JOB_KEYS
    if unknown:
        raise SystemExit(f"job '{name}': unknown keys {sorted(unknown)}")
    if not job.get("command", "").strip():
        raise SystemExit(f"job '{name}': missing 'command'")
    if job.get("gpus", 0) and job.get("partition") != "GPUQ":
        raise SystemExit(f"job '{name}': GPU jobs must run on partition GPUQ")
    if job.get("checkpoint_signal") and "\n" in job["command"].strip():
        raise SystemExit(f"job '{name}': checkpoint_signal needs a single-line command (it is exec'd)")
    return job


def render(cfg: dict, name: str, job: dict) -> str:
    cluster = cfg["cluster"]
    if cluster.get("account", "CHANGE_ME") == "CHANGE_ME":
        raise SystemExit('set your account in slurm/cluster.local.toml first: [cluster] account = "..."')

    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --account={cluster['account']}",
        f"#SBATCH --partition={job['partition']}",
        f"#SBATCH --time={job['time']}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={job['cpus']}",
        f"#SBATCH --mem={job['mem']}",
    ]
    gpus = int(job.get("gpus", 0))
    if gpus:
        header.append(f"#SBATCH --gres=gpu:{gpus}")
        if job.get("constraint"):
            header.append(f'#SBATCH --constraint="{job["constraint"]}"')

    array = job.get("array") or []
    if array:
        spec = f"0-{len(array) - 1}"
        if job.get("max_parallel"):
            spec += f"%{job['max_parallel']}"
        header.append(f"#SBATCH --array={spec}")
        header.append(f"#SBATCH --output={LOG_DIR}/%x-%A_%a.out")
    else:
        header.append(f"#SBATCH --output={LOG_DIR}/%x-%j.out")
    if job.get("checkpoint_signal"):
        # Python is exec'd below, so it receives USR1 three minutes before the time limit
        # and can write a final checkpoint. Resubmitting the job resumes from it.
        header.append("#SBATCH --signal=B:USR1@180")
    if cluster.get("mail_user"):
        header += [f"#SBATCH --mail-user={cluster['mail_user']}", "#SBATCH --mail-type=END,FAIL"]

    env = {**cluster.get("env", {}), **job.get("env", {})}
    body = [
        "",
        "set -eo pipefail",
        "",
        "module purge",
        f"module load {cluster['python_module']}",
        *[f"module load {m}" for m in cluster.get("extra_modules", [])],
        f'source "{cluster["venv"]}/bin/activate"',
        "set -u",
        "",
        f'export WORK="{cluster["workdir"]}"',
        'export DATA="$WORK/data" RUNS="$WORK/runs"',
        'export HF_HOME="$WORK/.cache/huggingface" TORCH_HOME="$WORK/.cache/torch"',
        "export PYTHONUNBUFFERED=1",
        *[f'export {key}="{value}"' for key, value in env.items()],
        f'cd "{PROJECT_ROOT}"',
        "",
        'echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"',
    ]
    if gpus:
        body.append("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true")
    if array:
        args_file = OUT_DIR / f"{name}.args"
        body += [
            f'ARRAY_ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{args_file}")',
            'ARRAY_ARGS=$(eval echo "$ARRAY_ARGS")  # expand $DATA, $RUNS, ... in the args line',
            'echo "array task $SLURM_ARRAY_TASK_ID: $ARRAY_ARGS"',
        ]
    command = job["command"].strip()
    body.append(f"exec {command}" if job.get("checkpoint_signal") else command)
    return "\n".join(header + body) + "\n"


def write_job(cfg: dict, name: str, job: dict) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    if job.get("array"):
        (OUT_DIR / f"{name}.args").write_text("\n".join(job["array"]) + "\n")
    path = OUT_DIR / f"{name}.sbatch"
    path.write_text(render(cfg, name, job))
    path.chmod(0o755)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render Slurm scripts for the cluster.")
    parser.add_argument("jobs", nargs="*", help="job names from slurm/cluster.toml")
    parser.add_argument("--all", action="store_true", help="render every job")
    parser.add_argument("--list", action="store_true", help="list jobs and exit")
    parser.add_argument("--submit", action="store_true", help="submit with sbatch after rendering")
    parser.add_argument("--print", action="store_true", help="print the script instead of writing")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--shell-env", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.shell_env:  # consumed by scripts/setup_cluster.sh and scripts/download_data.sh
        cluster = cfg["cluster"]
        work = cluster["workdir"]
        print(f'VENV="{cluster["venv"]}"\nWORK="{work}"\nDATA="{work}/data"\nRUNS="{work}/runs"')
        return
    if args.list:
        for name, job in cfg.get("jobs", {}).items():
            print(f"{name:28s} {job.get('description', '')}")
        return

    names = list(cfg.get("jobs", {})) if args.all else args.jobs
    if not names:
        parser.error("give job names, --all or --list")
    overrides = dict(item.split("=", 1) for item in args.set)

    for name in names:
        job = resolve_job(cfg, name, overrides)
        if args.print:
            sys.stdout.write(render(cfg, name, job))
            continue
        path = write_job(cfg, name, job)
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")
        if args.submit:
            subprocess.run(["sbatch", str(path)], check=True)


if __name__ == "__main__":
    main()
