#!/bin/bash
# Setup local test environment for IREE Metal PJRT plugin using uv.
#
# The pyproject.toml in this directory declares the plugin as an editable
# source install, which transitively pulls JAX and iree-base-compiler from
# local checkouts. This script validates prerequisites, installs the compiler
# Python packages, and runs `uv sync`.
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - IREE compiler build at compiler/build/b (cmake --build)
#   - JAX source checkout at ../jax (sibling to the IREE repo root)
#
# Usage:
#   bash setup_local_test_env.sh [--clean]
#
# Options:
#   --clean    Remove .venv and uv.lock before setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IREE_COMPILER_BUILD="${IREE_ROOT}/compiler/build/b"
PJRT_PLUGIN="${SCRIPT_DIR}/../python_packages/iree_metal_plugin"
JAX_ROOT="${IREE_ROOT}/../jax"

echo "=== IREE Metal PJRT Local Test Environment Setup ==="
echo "IREE_ROOT:            ${IREE_ROOT}"
echo "IREE_COMPILER_BUILD:  ${IREE_COMPILER_BUILD}"
echo "PJRT_PLUGIN:          ${PJRT_PLUGIN}"
echo "JAX_ROOT:             ${JAX_ROOT}"

# ── Validate prerequisites ───────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

if [[ ! -d "${IREE_COMPILER_BUILD}" ]]; then
    echo "ERROR: IREE compiler build not found at ${IREE_COMPILER_BUILD}"
    echo "Build it first:  cmake -G Ninja -B compiler/build/b -S . && cmake --build compiler/build/b"
    exit 1
fi

if [[ ! -d "${JAX_ROOT}" ]]; then
    echo "ERROR: JAX source checkout not found at ${JAX_ROOT}"
    echo "Clone it:  git clone https://github.com/jax-ml/jax.git ${JAX_ROOT}"
    exit 1
fi

if [[ ! -d "${PJRT_PLUGIN}" ]]; then
    echo "ERROR: Metal PJRT plugin not found at ${PJRT_PLUGIN}"
    exit 1
fi

# ── Handle --clean flag ──────────────────────────────────────────────────────

if [[ "${1:-}" == "--clean" ]]; then
    echo ""
    echo "=== Cleaning existing environment ==="
    rm -rf "${SCRIPT_DIR}/.venv" "${SCRIPT_DIR}/uv.lock"
    echo "Removed .venv and uv.lock"
fi

# ── Install compiler Python packages ─────────────────────────────────────────

echo ""
echo "=== Installing compiler Python packages ==="
cmake --install "${IREE_COMPILER_BUILD}" --prefix "${IREE_ROOT}/compiler/build/i" --component IREECompilerPythonPackages
echo "Installed to ${IREE_ROOT}/compiler/build/i"

# ── Sync the environment ─────────────────────────────────────────────────────

echo ""
echo "=== Running uv sync (first run builds the plugin — may take 15-30 min) ==="
cd "${SCRIPT_DIR}"
uv sync

# ── Verify installation ──────────────────────────────────────────────────────

echo ""
echo "=== Verifying installation ==="

uv run python -c "
import sys
print(f'Python: {sys.version}')

import jax
print(f'JAX version: {jax.__version__}')

try:
    import iree.compiler
    print(f'iree-base-compiler: {iree.compiler.__path__}')
except Exception as e:
    print(f'iree-base-compiler: import error: {e}')

try:
    import iree._pjrt_libs.metal as m
    import os
    print(f'Metal PJRT plugin: {os.path.dirname(m.__file__)}')
except ImportError as e:
    print(f'Metal plugin import error: {e}')

# Check Metal devices are visible
try:
    jax.config.update('jax_platforms', 'iree_metal')
    devices = jax.devices()
    print(f'Metal devices: {devices}')
except Exception as e:
    print(f'Metal device probe: {e}')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run tests:"
echo "  cd ${SCRIPT_DIR}"
echo "  uv run python -c \"import jax; jax.config.update('jax_platforms', 'iree_metal'); print(jax.devices())\""
echo "  uv run pytest test_metal.py -v"
echo ""
echo "To run a quick smoke test:"
echo "  cd ${SCRIPT_DIR}"
echo "  uv run python test_add.py"
