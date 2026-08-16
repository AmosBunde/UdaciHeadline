#!/usr/bin/env bash
# Create a reproducible Python environment for the UdaciHeadline project and lessons.
#
# Usage:
#   ./setup_env.sh            # auto-detects CUDA via nvidia-smi
#   ./setup_env.sh cpu        # force CPU-only torch wheels
#   ./setup_env.sh cuda       # force CUDA torch wheels
#
# Requires `uv` (https://docs.astral.sh/uv/). Falls back to python -m venv + pip.
set -euo pipefail

cd "$(dirname "$0")"
MODE="${1:-auto}"
PY="${PYTHON_VERSION:-3.12}"

if [[ "$MODE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then MODE=cuda; else MODE=cpu; fi
fi
echo ">> Setting up environment in .venv (mode: $MODE)"

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PY" .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  python3 -m venv .venv
  PIP=(.venv/bin/python -m pip install)
  "${PIP[@]}" --upgrade pip
fi

if [[ "$MODE" == "cpu" ]]; then
  "${PIP[@]}" --index-url https://download.pytorch.org/whl/cpu torch
else
  "${PIP[@]}" torch
fi
"${PIP[@]}" -r requirements.txt

# DeepSpeed is optional (used in Lesson 4 / distributed section). Best effort.
if [[ "${INSTALL_DEEPSPEED:-1}" == "1" ]]; then
  DS_BUILD_OPS=0 "${PIP[@]}" deepspeed || echo "!! DeepSpeed install failed (optional) - continuing"
fi

# Register a Jupyter kernel so notebooks can pick this env
.venv/bin/python -m ipykernel install --user --name udaciheadline --display-name "Python (udaciheadline)"

echo ">> Done. Activate with: source .venv/bin/activate"
echo ">> Verify with:        .venv/bin/python env_check.py"
