#!/bin/bash
# One-time setup on a cluster login node: venv, project install and work directories.
#
#   bash scripts/setup_cluster.sh
set -eo pipefail
cd "$(dirname "$0")/.."

if ! type module &>/dev/null; then
    source /etc/profile
fi
PY_MODULE=$(sed -n 's/^python_module *= *"\([^"]*\)".*/\1/p' slurm/cluster.local.toml slurm/cluster.toml 2>/dev/null | head -1)
module purge
module load "$PY_MODULE"
eval "$(python3 slurm/generate.py --shell-env)"

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel
# CUDA 12.6 wheels run on every GPU generation of the cluster (V100 to H200)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[ml,onnx,dev]" "$@"

mkdir -p "$WORK/data" "$WORK/runs" "$WORK/.cache"
python3 slurm/generate.py --all > /dev/null || true

echo "venv : $VENV"
echo "work : $WORK"
