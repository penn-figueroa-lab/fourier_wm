#!/usr/bin/env bash
# Optional isolated DINO-WM Maze runtime; the main uv environment stays Python3.11.
set -euo pipefail
: "${FOURIER_LEGACY_ROOT:?Set an external directory for the legacy runtime}"
uv="${FOURIER_UV:-uv}"
"$uv" --version
mkdir -p "$FOURIER_LEGACY_ROOT"
runtime="$FOURIER_LEGACY_ROOT/python39"
export UV_PYTHON_INSTALL_DIR="$FOURIER_LEGACY_ROOT/python-managed"
if [[ ! -x "$runtime/bin/python" ]]; then
  "$uv" venv --python 3.9 --python-preference only-managed "$runtime"
fi
archive="$FOURIER_LEGACY_ROOT/mujoco210-linux-x86_64.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --output "$archive.partial" https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  mv "$archive.partial" "$archive"
fi
printf '%s  %s\n' a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012 "$archive" | sha256sum --check
if [[ ! -d "$FOURIER_LEGACY_ROOT/mujoco210" ]]; then
  tar -xzf "$archive" -C "$FOURIER_LEGACY_ROOT"
fi
"$uv" pip install --python "$runtime/bin/python" numpy==1.26.4 cython==0.29.37 \
  setuptools==80.10.2 wheel==0.48.0 patchelf==0.19.3.0 packaging==26.3 \
  mujoco-py==2.1.2.14 cffi==2.0.0 fasteners==0.20 glfw==2.11.0 \
  imageio==2.37.2 pillow==11.3.0 pycparser==2.23
# Install no system packages. If needed, unpack Ubuntu OSMesa libraries/headers
# into the external runtime; use the allocated node's configured apt sources.
osmesa="$FOURIER_LEGACY_ROOT/osmesa"
if [[ ! -f /usr/include/GL/osmesa.h && ! -f "$osmesa/usr/lib/x86_64-linux-gnu/libGL.so.1" ]]; then
  mkdir -p "$osmesa/packages"
  ( cd "$osmesa/packages"
    apt-get download libosmesa6 libosmesa6-dev libgl-dev libglx-dev libgl1 libglx0 libglvnd0
    for package in ./*.deb; do dpkg-deb -x "$package" "$osmesa"; done
    sha256sum ./*.deb > SHA256SUMS
  )
fi
export MUJOCO_PY_FORCE_CPU=1 MUJOCO_PY_MUJOCO_PATH="$FOURIER_LEGACY_ROOT/mujoco210"
export LD_LIBRARY_PATH="$osmesa/usr/lib/x86_64-linux-gnu:$runtime/lib:$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$osmesa/usr/lib/x86_64-linux-gnu:$runtime/lib:$MUJOCO_PY_MUJOCO_PATH/bin:${LIBRARY_PATH:-}"
export CPATH="$osmesa/usr/include:$runtime/include:${CPATH:-}" PATH="$runtime/bin:$PATH"
"$runtime/bin/python" -c 'import mujoco_py; print("LEGACY_ENGINE_READY",mujoco_py.__version__)'
"$uv" pip freeze --python "$runtime/bin/python" > "$FOURIER_LEGACY_ROOT/requirements-resolved.txt"
# mujoco-py's CPU renderer still parses a numeric device ID. Override its GPUS
# fallback so a Slurm MIG UUID in CUDA_VISIBLE_DEVICES remains untouched.
printf 'export FOURIER_LEGACY_PYTHON=%q\nexport MUJOCO_PY_MUJOCO_PATH=%q\nexport GPUS=0\n' \
  "$runtime/bin/python" "$MUJOCO_PY_MUJOCO_PATH" > "$FOURIER_LEGACY_ROOT/runtime.env"
