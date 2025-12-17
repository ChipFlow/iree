#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeo pipefail

pjrt_platform=$1
run_mode=$2

if [ -z "${pjrt_platform}" ]; then
    set +x
    echo "Usage: run_jax_tests.sh <pjrt_platform> [--full-suite]"
    echo "  <pjrt_platform> can be 'cpu', 'cuda', 'rocm', 'vulkan' or 'metal'"
    echo "  --full-suite: Run full JAX test suite (requires JAX_TEST_DIR env var)"
    exit 1
fi

# cd into the PJRT plugin dir
ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${ROOT_DIR}/integrations/pjrt"

# perform some differential testing
actual_jax_platform=iree_${pjrt_platform}
expected_jax_platform=cpu

# this function will execute the test python script in
# both cpu mode and the IREE PJRT mode,
# and then compare the difference in the output
diff_jax_test() {
    local test_py_file=$1

    echo "executing ${test_py_file} in ${expected_jax_platform}.."
    local expected_tmp_out=$(mktemp /tmp/jax_test_result_expected.XXXXXX)
    JAX_PLATFORMS=$expected_jax_platform python $test_py_file > $expected_tmp_out

    echo "executing ${test_py_file} in ${actual_jax_platform}.."
    local actual_tmp_out=$(mktemp /tmp/jax_test_result_actual.XXXXXX)
    JAX_PLATFORMS=$actual_jax_platform python $test_py_file > $actual_tmp_out

    echo "comparing ${expected_tmp_out} and ${actual_tmp_out}.."
    diff --unified $expected_tmp_out $actual_tmp_out
    echo "no difference found"
}

diff_jax_test test/test_add.py
diff_jax_test test/test_degenerate.py
diff_jax_test test/test_simple.py

# Test Shardy dialect support (JAX 0.8.2+ uses Shardy by default)
# This test verifies the sdy dialect can be deserialized and stripped
echo "Testing Shardy dialect support..."
JAX_PLATFORMS=$actual_jax_platform python test/test_shardy.py

# Platform-specific tests
if [ "${pjrt_platform}" = "metal" ]; then
    echo "Running Metal-specific tests..."
    JAX_PLATFORMS=$actual_jax_platform python test/test_metal.py
fi

# here we test if the compile options is passed to IREE PJRT plugin successfully.
# we pass --iree-scheduling-dump-statistics-format=csv via jax.jit,
# and see if there's statistics in the output
compile_options_test_tmp_out=$(mktemp /tmp/jax_test_result_compile_options.XXXXXX)
JAX_PLATFORMS=$actual_jax_platform python test/test_compile_options.py 2>&1 | tee $compile_options_test_tmp_out
cat $compile_options_test_tmp_out | grep '@main_dispatch'


# Full JAX test suite mode
if [ "${run_mode}" = "--full-suite" ]; then
    if [ -z "${JAX_TEST_DIR}" ]; then
        echo "ERROR: JAX_TEST_DIR environment variable must be set for --full-suite mode"
        echo "Example: JAX_TEST_DIR=/path/to/jax/tests ./run_jax_tests.sh metal --full-suite"
        exit 1
    fi

    echo "Running full JAX test suite from ${JAX_TEST_DIR}..."

    # Configuration
    JOBS=${JOBS:-4}
    TIMEOUT=${TIMEOUT:-120}
    LOG_DIR=${LOG_DIR:-/tmp/jaxtest}
    EXPECTED_FILE="test/${pjrt_platform}_expected_passing.txt"
    PASSING_FILE="${LOG_DIR}/${pjrt_platform}_passing.txt"
    FAILING_FILE="${LOG_DIR}/${pjrt_platform}_failing.txt"

    # Default test files - core JAX tests most likely to work
    TEST_FILES="${JAX_TEST_FILES:-${JAX_TEST_DIR}/lax_test.py}"

    echo "Test configuration:"
    echo "  Platform: ${actual_jax_platform}"
    echo "  Jobs: ${JOBS}"
    echo "  Timeout: ${TIMEOUT}s"
    echo "  Test files: ${TEST_FILES}"

    # Run tests with expected results comparison
    JAX_PLATFORMS=${actual_jax_platform} python test/test_jax.py \
        ${TEST_FILES} \
        -j ${JOBS} \
        -t ${TIMEOUT} \
        -l ${LOG_DIR} \
        -e ${EXPECTED_FILE} \
        -p ${PASSING_FILE} \
        -f ${FAILING_FILE}

    echo ""
    echo "=== Test Results ==="
    echo "Passing: $(wc -l < ${PASSING_FILE} 2>/dev/null || echo 0)"
    echo "Failing: $(wc -l < ${FAILING_FILE} 2>/dev/null || echo 0)"
    echo "Log directory: ${LOG_DIR}"
fi
