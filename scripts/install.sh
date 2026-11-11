#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
cd "$project_root"
uv sync --frozen --python "${FOURIER_BOOTSTRAP_PYTHON:-3.11}" --extra planning
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
.venv/bin/fourier-wm doctor
