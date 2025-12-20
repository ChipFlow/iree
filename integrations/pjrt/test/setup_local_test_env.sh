#!/bin/bash
# Setup local test environment for IREE Metal PJRT plugin
#
# This script creates a clean venv and builds the PJRT plugin with matching
# compiler/runtime versions to avoid ABI mismatches.
#
# Usage:
#   ./setup_local_test_env.sh [--clean]
#
# Options:
#   --clean    Remove existing build and venv directories before setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PJRT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUGIN_DIR="${PJRT_DIR}/python_packages/iree_metal_plugin"
BUILD_DIR="${PLUGIN_DIR}/build"
VENV_DIR="${BUILD_DIR}/.venv"

echo "=== IREE Metal PJRT Local Test Environment Setup ==="
echo "PJRT_DIR: ${PJRT_DIR}"
echo "PLUGIN_DIR: ${PLUGIN_DIR}"
echo "VENV_DIR: ${VENV_DIR}"

# Handle --clean flag
if [[ "$1" == "--clean" ]]; then
    echo ""
    echo "=== Cleaning existing build ==="
    rm -rf "${BUILD_DIR}"
    echo "Removed ${BUILD_DIR}"
fi

# Create venv if it doesn't exist
# Use Python 3.12 as iree-base-compiler doesn't have 3.14 wheels yet
if [[ ! -d "${VENV_DIR}" ]]; then
    echo ""
    echo "=== Creating virtual environment (Python 3.12) ==="
    uv venv --python 3.12 "${VENV_DIR}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"
echo "Activated venv: ${VIRTUAL_ENV}"
echo "Python version: $(python --version)"

# Upgrade pip and install build requirements
echo ""
echo "=== Installing build requirements ==="
uv pip install --upgrade pip wheel setuptools
uv pip install ninja cmake

# Install runtime build requirements (for nanobind, etc.)
RUNTIME_REQS="${PJRT_DIR}/../../runtime/bindings/python/iree/runtime/build_requirements.txt"
if [[ -f "${RUNTIME_REQS}" ]]; then
    uv pip install -r "${RUNTIME_REQS}"
fi

# Install JAX and test dependencies
echo ""
echo "=== Installing JAX and test dependencies ==="
uv pip install numpy jax jaxlib pytest absl-py hypothesis

# Install iree-base-compiler from pip (provides libIREECompiler.dylib)
# This is faster than building from source and works for testing
echo ""
echo "=== Installing IREE compiler ==="
uv pip install iree-base-compiler

# Build and install the Metal PJRT plugin
# We use the pip-installed iree-base-compiler for the compiler library
# and only build the PJRT runtime from source
echo ""
echo "=== Building and installing Metal PJRT plugin ==="
echo "Building PJRT plugin with compiler from source..."
echo "First build takes 15-30 minutes. Subsequent builds with ccache are faster."

# Set build environment variables
export CMAKE_OSX_ARCHITECTURES=arm64
export CMAKE_SYSTEM_PROCESSOR=arm64
# Note: We build the compiler from source (IREE_BUILD_COMPILER=ON by default)
# The pip iree-base-compiler provides libIREECompiler.dylib for JAX JIT at runtime
# Suppress warnings from protobuf's abseil on ARM64
export CFLAGS="-Wno-unused-command-line-argument"
export CXXFLAGS="-Wno-unused-command-line-argument"
export CMAKE_C_FLAGS="-Wno-unused-command-line-argument"
export CMAKE_CXX_FLAGS="-Wno-unused-command-line-argument"

# Use ccache if available
if command -v ccache &> /dev/null; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    echo "Using ccache for faster rebuilds"
fi

# Install the plugin (this triggers the build)
uv pip install -v --no-deps --no-build-isolation "${PLUGIN_DIR}"

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import jax
print(f'JAX version: {jax.__version__}')

# Check Metal plugin
try:
    import iree._pjrt_libs.metal as m
    import os
    print(f'Metal PJRT plugin: {os.path.dirname(m.__file__)}')
except ImportError as e:
    print(f'Metal plugin import error: {e}')

# Check compiler library
try:
    from iree.compiler.api import ctypes_dl
    lib_path = ctypes_dl._probe_iree_compiler_dylib()
    print(f'Compiler library: {lib_path}')
except Exception as e:
    print(f'Compiler probe error: {e}')
"

# Clone JAX for tests if not present
JAX_TEST_DIR="/tmp/jax_metal_test"
if [[ ! -d "${JAX_TEST_DIR}" ]]; then
    echo ""
    echo "=== Cloning JAX for tests ==="
    git clone --depth=1 https://github.com/jax-ml/jax.git "${JAX_TEST_DIR}"
fi

# Apply patches
echo ""
echo "=== Applying IREE Metal patches ==="
cd "${JAX_TEST_DIR}"
git checkout -- . 2>/dev/null || true  # Reset any previous patches
for patch in "${PJRT_DIR}/test/patches/"*.patch; do
    if [[ -f "$patch" ]]; then
        echo "Applying: $(basename $patch)"
        patch -p1 < "$patch" || echo "Patch may have already been applied"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run tests:"
echo "  JAX_PLATFORMS=iree_metal JAX_ENABLE_X64=0 pytest ${JAX_TEST_DIR}/tests/api_test.py::JitTest -v --tb=short"
echo ""
echo "To run a quick smoke test:"
echo "  JAX_PLATFORMS=iree_metal python ${PJRT_DIR}/test/test_add.py"
