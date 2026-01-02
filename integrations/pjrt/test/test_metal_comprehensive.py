#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Comprehensive test suite for IREE Metal PJRT plugin.

This test suite covers functionality gaps in the basic tests:
- Determinism/stability (repeated execution)
- Identity/constant operations
- Control flow primitives
- Data type coverage
- Scatter/gather operations
- Random number generation
- Linear algebra operations
- Buffer donation

Run with: JAX_PLATFORMS=iree_metal python test_metal_comprehensive.py
"""

import sys
import platform
import argparse
from typing import List, Tuple
import traceback

# Check if we're on macOS
if platform.system() != "Darwin":
    print(f"Skipping Metal tests on {platform.system()} (Metal requires macOS)")
    sys.exit(0)

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


class TestResult:
    """Track test results."""
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.skipped: List[Tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)

    def add_fail(self, name: str, reason: str):
        self.failed.append((name, reason))

    def add_skip(self, name: str, reason: str):
        self.skipped.append((name, reason))

    def summary(self) -> str:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        lines = [
            f"\n{'='*60}",
            f"Test Results: {len(self.passed)}/{total} passed",
            f"  Passed:  {len(self.passed)}",
            f"  Failed:  {len(self.failed)}",
            f"  Skipped: {len(self.skipped)}",
        ]
        if self.failed:
            lines.append("\nFailed tests:")
            for name, reason in self.failed:
                lines.append(f"  - {name}: {reason}")
        if self.skipped:
            lines.append("\nSkipped tests:")
            for name, reason in self.skipped:
                lines.append(f"  - {name}: {reason}")
        lines.append("="*60)
        return "\n".join(lines)


results = TestResult()


def run_test(name: str, verbose: bool = False):
    """Decorator to run a test function and track results."""
    def decorator(fn):
        def wrapper():
            try:
                if verbose:
                    print(f"Running {name}...", end=" ", flush=True)
                fn()
                results.add_pass(name)
                if verbose:
                    print("PASS")
            except AssertionError as e:
                results.add_fail(name, str(e))
                if verbose:
                    print(f"FAIL: {e}")
            except Exception as e:
                results.add_fail(name, f"{type(e).__name__}: {e}")
                if verbose:
                    print(f"ERROR: {type(e).__name__}: {e}")
                    traceback.print_exc()
        return wrapper
    return decorator


# =============================================================================
# SECTION 1: Determinism / Stability Tests
# =============================================================================

def test_determinism_eye(iterations: int = 100):
    """Test eye() returns consistent results across iterations."""
    @jax.jit
    def make_eye():
        return jnp.eye(4)

    # Warmup
    _ = make_eye()

    expected = np.eye(4, dtype=np.float32)
    failures = 0
    for i in range(iterations):
        result = np.array(make_eye())
        if not np.array_equal(result, expected):
            failures += 1
            if failures <= 3:
                print(f"  eye() iteration {i}: {result.flatten()}")

    assert failures == 0, f"eye() failed {failures}/{iterations} times"


def test_determinism_zeros(iterations: int = 100):
    """Test zeros() returns consistent results."""
    @jax.jit
    def make_zeros():
        return jnp.zeros((4, 4))

    _ = make_zeros()
    expected = np.zeros((4, 4), dtype=np.float32)

    for i in range(iterations):
        result = np.array(make_zeros())
        assert np.array_equal(result, expected), f"zeros() failed at iteration {i}"


def test_determinism_ones(iterations: int = 100):
    """Test ones() returns consistent results."""
    @jax.jit
    def make_ones():
        return jnp.ones((4, 4))

    _ = make_ones()
    expected = np.ones((4, 4), dtype=np.float32)

    for i in range(iterations):
        result = np.array(make_ones())
        assert np.array_equal(result, expected), f"ones() failed at iteration {i}"


def test_determinism_arange(iterations: int = 100):
    """Test arange() returns consistent results."""
    @jax.jit
    def make_arange():
        return jnp.arange(16).reshape(4, 4)

    _ = make_arange()
    expected = np.arange(16, dtype=np.int32).reshape(4, 4)

    for i in range(iterations):
        result = np.array(make_arange())
        assert np.array_equal(result, expected), f"arange() failed at iteration {i}"


def test_determinism_matmul(iterations: int = 100):
    """Test matrix multiply returns consistent results."""
    @jax.jit
    def matmul(a, b):
        return jnp.dot(a, b)

    a = jnp.ones((8, 8))
    b = jnp.eye(8) * 2.0

    _ = matmul(a, b)
    expected = np.array(matmul(a, b))

    for i in range(iterations):
        result = np.array(matmul(a, b))
        assert np.allclose(result, expected), f"matmul() failed at iteration {i}"


# =============================================================================
# SECTION 2: Identity / Constant Operations
# =============================================================================

def test_eye_sizes():
    """Test eye() with various sizes."""
    for n in [1, 2, 3, 4, 8, 16, 32, 64]:
        result = jnp.eye(n)
        expected = np.eye(n, dtype=np.float32)
        assert np.allclose(result, expected), f"eye({n}) failed"


def test_eye_rectangular():
    """Test eye() with rectangular shapes."""
    for m, n in [(3, 4), (4, 3), (2, 5), (5, 2)]:
        result = jnp.eye(m, n)
        expected = np.eye(m, n, dtype=np.float32)
        assert np.allclose(result, expected), f"eye({m}, {n}) failed"


def test_eye_with_k():
    """Test eye() with diagonal offset."""
    for k in [-2, -1, 0, 1, 2]:
        result = jnp.eye(5, k=k)
        expected = np.eye(5, k=k, dtype=np.float32)
        assert np.allclose(result, expected), f"eye(5, k={k}) failed"


def test_identity():
    """Test identity()."""
    for n in [1, 2, 4, 8]:
        result = jnp.identity(n)
        expected = np.identity(n, dtype=np.float32)
        assert np.allclose(result, expected), f"identity({n}) failed"


def test_zeros_shapes():
    """Test zeros() with various shapes."""
    shapes = [(4,), (4, 4), (2, 3, 4), (2, 2, 2, 2)]
    for shape in shapes:
        result = jnp.zeros(shape)
        expected = np.zeros(shape, dtype=np.float32)
        assert np.allclose(result, expected), f"zeros({shape}) failed"


def test_ones_shapes():
    """Test ones() with various shapes."""
    shapes = [(4,), (4, 4), (2, 3, 4), (2, 2, 2, 2)]
    for shape in shapes:
        result = jnp.ones(shape)
        expected = np.ones(shape, dtype=np.float32)
        assert np.allclose(result, expected), f"ones({shape}) failed"


def test_full():
    """Test full() with various fill values."""
    for val in [0.0, 1.0, -1.0, 3.14, 42.0]:
        result = jnp.full((4, 4), val)
        expected = np.full((4, 4), val, dtype=np.float32)
        assert np.allclose(result, expected), f"full((4,4), {val}) failed"


def test_linspace():
    """Test linspace()."""
    result = jnp.linspace(0, 1, 11)
    expected = np.linspace(0, 1, 11, dtype=np.float32)
    assert np.allclose(result, expected, atol=1e-6), "linspace failed"


def test_arange_variants():
    """Test arange() with various arguments."""
    test_cases = [
        (10,),           # arange(10)
        (0, 10),         # arange(0, 10)
        (0, 10, 2),      # arange(0, 10, 2)
        (5, 0, -1),      # arange(5, 0, -1)
    ]
    for args in test_cases:
        result = jnp.arange(*args)
        expected = np.arange(*args)
        assert np.allclose(result, expected), f"arange{args} failed"


# =============================================================================
# SECTION 3: Data Type Coverage
# =============================================================================

def test_dtype_float16():
    """Test float16 operations (Metal supports this)."""
    a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float16)
    b = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float16)
    result = a + b
    expected = np.array([5.0, 7.0, 9.0], dtype=np.float16)
    assert result.dtype == jnp.float16
    assert np.allclose(result, expected, rtol=1e-2)  # float16 has lower precision


def test_dtype_bfloat16():
    """Test bfloat16 operations (Metal supports this)."""
    a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16)
    b = jnp.array([4.0, 5.0, 6.0], dtype=jnp.bfloat16)
    result = a + b
    # bfloat16 may not be directly comparable, cast to float32
    result_f32 = result.astype(jnp.float32)
    expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    assert result.dtype == jnp.bfloat16
    assert np.allclose(result_f32, expected, rtol=1e-2)


def test_dtype_float32():
    """Test float32 operations."""
    a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    b = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    result = a + b
    expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    assert result.dtype == jnp.float32
    assert np.allclose(result, expected)


def test_dtype_float64_skip():
    """Test float64 operations - SKIPPED on Metal (not supported)."""
    # Metal does not support float64 - this is a hardware limitation
    # Apple GPUs only support 32-bit and smaller floating point types
    raise AssertionError("SKIP: Metal does not support float64")


def test_dtype_int32():
    """Test int32 operations."""
    a = jnp.array([1, 2, 3], dtype=jnp.int32)
    b = jnp.array([4, 5, 6], dtype=jnp.int32)
    result = a + b
    expected = np.array([5, 7, 9], dtype=np.int32)
    assert result.dtype == jnp.int32
    assert np.array_equal(result, expected)


def test_dtype_int64_skip():
    """Test int64 operations - SKIPPED on Metal (not supported)."""
    # Metal does not support int64 - this is a hardware limitation
    raise AssertionError("SKIP: Metal does not support int64")


def test_dtype_bool():
    """Test boolean operations."""
    a = jnp.array([True, False, True])
    b = jnp.array([True, True, False])

    and_result = jnp.logical_and(a, b)
    or_result = jnp.logical_or(a, b)

    assert np.array_equal(and_result, [True, False, False])
    assert np.array_equal(or_result, [True, True, True])


def test_dtype_casting():
    """Test dtype casting."""
    a = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float32)

    # Float to int (truncation)
    b = a.astype(jnp.int32)
    assert np.array_equal(b, [1, 2, 3])

    # Int to float
    c = b.astype(jnp.float32)
    assert np.array_equal(c, [1.0, 2.0, 3.0])


def test_dtype_complex():
    """Test complex number operations."""
    a = jnp.array([1+2j, 3+4j], dtype=jnp.complex64)
    b = jnp.array([5+6j, 7+8j], dtype=jnp.complex64)

    result = a + b
    expected = np.array([6+8j, 10+12j], dtype=np.complex64)
    assert result.dtype == jnp.complex64
    assert np.allclose(result, expected)


# =============================================================================
# SECTION 4: Control Flow
# =============================================================================

def test_cond_true():
    """Test lax.cond with true branch."""
    @jax.jit
    def f(x):
        return lax.cond(x > 0, lambda y: y * 2, lambda y: y * 3, x)

    result = f(jnp.array(5.0))
    assert np.isclose(result, 10.0), f"cond(true) failed: {result}"


def test_cond_false():
    """Test lax.cond with false branch."""
    @jax.jit
    def f(x):
        return lax.cond(x > 0, lambda y: y * 2, lambda y: y * 3, x)

    result = f(jnp.array(-5.0))
    assert np.isclose(result, -15.0), f"cond(false) failed: {result}"


def test_while_loop():
    """Test lax.while_loop."""
    @jax.jit
    def f(n):
        def cond_fn(state):
            i, _ = state
            return i < n

        def body_fn(state):
            i, total = state
            return (i + 1, total + i)

        _, result = lax.while_loop(cond_fn, body_fn, (0, 0))
        return result

    # Sum of 0..9 = 45
    result = f(jnp.array(10))
    assert np.isclose(result, 45), f"while_loop failed: {result}"


def test_fori_loop():
    """Test lax.fori_loop."""
    @jax.jit
    def f(n):
        def body_fn(i, total):
            return total + i
        return lax.fori_loop(0, n, body_fn, 0)

    result = f(10)
    assert np.isclose(result, 45), f"fori_loop failed: {result}"


def test_scan():
    """Test lax.scan.

    KNOWN ISSUE: IREE Metal returns input values instead of accumulated results.
    The scan body function's carry seems to be ignored.
    Expected: [1, 3, 6, 10] (cumulative sum)
    Actual: [1, 2, 3, 4] (just the inputs)
    """
    @jax.jit
    def cumsum(xs):
        def body_fn(carry, x):
            new_carry = carry + x
            return new_carry, new_carry
        _, ys = lax.scan(body_fn, 0.0, xs)
        return ys

    xs = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = cumsum(xs)
    expected = np.array([1.0, 3.0, 6.0, 10.0])
    # TODO: Remove XFAIL when IREE Metal scan is fixed
    if not np.allclose(result, expected):
        raise AssertionError(f"KNOWN ISSUE: scan returns {result}, expected {expected}")


def test_select():
    """Test lax.select (where)."""
    @jax.jit
    def f(cond, a, b):
        return lax.select(cond, a, b)

    cond = jnp.array([True, False, True, False])
    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    b = jnp.array([10.0, 20.0, 30.0, 40.0])

    result = f(cond, a, b)
    expected = np.array([1.0, 20.0, 3.0, 40.0])
    assert np.allclose(result, expected), f"select failed: {result}"


# =============================================================================
# SECTION 5: Scatter / Gather Operations
# =============================================================================

def test_gather_1d():
    """Test 1D gather (indexing)."""
    data = jnp.arange(10)
    indices = jnp.array([0, 2, 4, 6, 8])
    result = data[indices]
    expected = np.array([0, 2, 4, 6, 8])
    assert np.array_equal(result, expected), f"1D gather failed: {result}"


def test_gather_2d():
    """Test 2D gather."""
    data = jnp.arange(16).reshape(4, 4)
    result = data[jnp.array([0, 1, 2]), jnp.array([0, 1, 2])]  # Diagonal
    expected = np.array([0, 5, 10])
    assert np.array_equal(result, expected), f"2D gather failed: {result}"


def test_scatter_1d():
    """Test 1D scatter (index update)."""
    data = jnp.zeros(5)
    indices = jnp.array([1, 3])
    updates = jnp.array([10.0, 30.0])
    result = data.at[indices].set(updates)
    expected = np.array([0.0, 10.0, 0.0, 30.0, 0.0])
    assert np.allclose(result, expected), f"1D scatter failed: {result}"


def test_scatter_add():
    """Test scatter add (accumulate at indices)."""
    data = jnp.zeros(5)
    indices = jnp.array([1, 1, 3])  # Duplicate index
    updates = jnp.array([10.0, 20.0, 30.0])
    result = data.at[indices].add(updates)
    expected = np.array([0.0, 30.0, 0.0, 30.0, 0.0])
    assert np.allclose(result, expected), f"scatter add failed: {result}"


def test_dynamic_slice():
    """Test dynamic_slice.

    KNOWN ISSUE: IREE Metal ignores the start index and always slices from 0.
    Expected: [2, 3, 4] (starting at index 2)
    Actual: [0, 1, 2] (starting at index 0)
    """
    @jax.jit
    def f(data, start):
        return lax.dynamic_slice(data, (start,), (3,))

    data = jnp.arange(10)
    result = f(data, 2)
    expected = np.array([2, 3, 4])
    # TODO: Remove XFAIL when IREE Metal dynamic_slice is fixed
    if not np.array_equal(result, expected):
        raise AssertionError(f"KNOWN ISSUE: dynamic_slice returns {result}, expected {expected}")


def test_dynamic_update_slice():
    """Test dynamic_update_slice.

    KNOWN ISSUE: IREE Metal ignores the start index and updates from index 0.
    Expected: [0, 0, 1, 2, 3, 0, 0, 0, 0, 0]
    Actual: [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    """
    @jax.jit
    def f(data, update, start):
        return lax.dynamic_update_slice(data, update, (start,))

    data = jnp.zeros(10, dtype=jnp.int32)
    update = jnp.array([1, 2, 3], dtype=jnp.int32)
    result = f(data, update, 2)
    expected = np.array([0, 0, 1, 2, 3, 0, 0, 0, 0, 0])
    # TODO: Remove XFAIL when IREE Metal dynamic_update_slice is fixed
    if not np.array_equal(result, expected):
        raise AssertionError(f"KNOWN ISSUE: dynamic_update_slice returns {result}, expected {expected}")


# =============================================================================
# SECTION 6: Random Number Generation
# =============================================================================
# NOTE: Random tests may fail with "dialect 'sdy' does not implement the
# bytecode interface" if there's a version mismatch between JAX (which uses
# Shardy) and the IREE compiler (which may not support the Shardy dialect
# version). This is a known issue when using PyPI iree-base-compiler with
# a development version of JAX.

def _check_random_support():
    """Check if random ops are supported (Shardy compatibility)."""
    try:
        key = jax.random.PRNGKey(0)
        _ = jax.random.uniform(key, shape=(2,))
        return True
    except Exception as e:
        if "sdy" in str(e) or "bytecode" in str(e):
            return False
        raise


def test_random_uniform():
    """Test random.uniform generates values in range."""
    if not _check_random_support():
        raise AssertionError("SKIP: Random ops not supported (Shardy dialect version mismatch)")

    key = jax.random.PRNGKey(0)
    result = jax.random.uniform(key, shape=(1000,))

    assert result.shape == (1000,)
    assert jnp.all(result >= 0.0), "uniform has values < 0"
    assert jnp.all(result <= 1.0), "uniform has values > 1"
    # Check distribution is roughly uniform
    assert 0.4 < jnp.mean(result) < 0.6, f"uniform mean out of range: {jnp.mean(result)}"


def test_random_normal():
    """Test random.normal generates reasonable distribution."""
    if not _check_random_support():
        raise AssertionError("SKIP: Random ops not supported (Shardy dialect version mismatch)")

    key = jax.random.PRNGKey(42)
    result = jax.random.normal(key, shape=(10000,))

    assert result.shape == (10000,)
    # Mean should be close to 0, std close to 1
    assert -0.1 < jnp.mean(result) < 0.1, f"normal mean out of range: {jnp.mean(result)}"
    assert 0.9 < jnp.std(result) < 1.1, f"normal std out of range: {jnp.std(result)}"


def test_random_split():
    """Test random.split produces different keys."""
    if not _check_random_support():
        raise AssertionError("SKIP: Random ops not supported (Shardy dialect version mismatch)")

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Keys should be different
    assert not np.array_equal(key1, key2), "split produced identical keys"

    # Values generated should be different
    v1 = jax.random.uniform(key1, shape=(10,))
    v2 = jax.random.uniform(key2, shape=(10,))
    assert not np.allclose(v1, v2), "split keys produced same random values"


def test_random_choice():
    """Test random.choice."""
    if not _check_random_support():
        raise AssertionError("SKIP: Random ops not supported (Shardy dialect version mismatch)")

    key = jax.random.PRNGKey(0)
    data = jnp.arange(10)
    result = jax.random.choice(key, data, shape=(5,), replace=False)

    assert result.shape == (5,)
    assert len(jnp.unique(result)) == 5, "choice with replace=False has duplicates"


def test_random_permutation():
    """Test random.permutation."""
    if not _check_random_support():
        raise AssertionError("SKIP: Random ops not supported (Shardy dialect version mismatch)")

    key = jax.random.PRNGKey(0)
    data = jnp.arange(10)
    result = jax.random.permutation(key, data)

    assert result.shape == (10,)
    assert set(np.array(result)) == set(range(10)), "permutation changed values"


# =============================================================================
# SECTION 7: Linear Algebra Operations
# =============================================================================

def test_matmul_shapes():
    """Test matrix multiply with various shapes."""
    test_cases = [
        ((4, 4), (4, 4)),
        ((2, 3), (3, 4)),
        ((1, 10), (10, 1)),
        ((8, 16), (16, 8)),
    ]
    for shape_a, shape_b in test_cases:
        a = jnp.ones(shape_a)
        b = jnp.ones(shape_b)
        result = jnp.dot(a, b)
        expected = np.dot(np.ones(shape_a), np.ones(shape_b))
        assert np.allclose(result, expected), f"matmul {shape_a} @ {shape_b} failed"


def test_batched_matmul():
    """Test batched matrix multiply."""
    a = jnp.ones((4, 3, 3))
    b = jnp.eye(3).reshape(1, 3, 3).repeat(4, axis=0)
    result = jnp.matmul(a, b)
    expected = np.ones((4, 3, 3))
    assert np.allclose(result, expected), "batched matmul failed"


def test_transpose():
    """Test matrix transpose."""
    a = jnp.arange(12).reshape(3, 4)
    result = a.T
    expected = np.arange(12).reshape(3, 4).T
    assert np.array_equal(result, expected), "transpose failed"


def test_trace():
    """Test matrix trace."""
    a = jnp.arange(16).reshape(4, 4)
    result = jnp.trace(a)
    expected = 0 + 5 + 10 + 15  # diagonal sum
    assert np.isclose(result, expected), f"trace failed: {result}"


def test_diag():
    """Test diagonal extraction and creation."""
    # Extract diagonal
    a = jnp.arange(16).reshape(4, 4)
    result = jnp.diag(a)
    expected = np.array([0, 5, 10, 15])
    assert np.array_equal(result, expected), "diag extract failed"

    # Create diagonal matrix
    v = jnp.array([1, 2, 3])
    result = jnp.diag(v)
    expected = np.diag([1, 2, 3])
    assert np.array_equal(result, expected), "diag create failed"


def test_outer():
    """Test outer product."""
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0])
    result = jnp.outer(a, b)
    expected = np.outer([1, 2, 3], [4, 5])
    assert np.allclose(result, expected), "outer product failed"


def test_einsum():
    """Test einsum for various contractions."""
    a = jnp.ones((3, 4))
    b = jnp.ones((4, 5))

    # Matrix multiply
    result = jnp.einsum('ij,jk->ik', a, b)
    expected = np.einsum('ij,jk->ik', np.ones((3, 4)), np.ones((4, 5)))
    assert np.allclose(result, expected), "einsum matmul failed"

    # Trace
    c = jnp.arange(9).reshape(3, 3)
    result = jnp.einsum('ii', c)
    expected = 0 + 4 + 8
    assert np.isclose(result, expected), "einsum trace failed"


def test_norm():
    """Test vector/matrix norms."""
    v = jnp.array([3.0, 4.0])

    # L2 norm
    result = jnp.linalg.norm(v)
    assert np.isclose(result, 5.0), f"L2 norm failed: {result}"

    # L1 norm
    result = jnp.linalg.norm(v, ord=1)
    assert np.isclose(result, 7.0), f"L1 norm failed: {result}"


def test_solve():
    """Test linear system solve (if supported)."""
    # Simple 2x2 system: Ax = b
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    b = jnp.array([1.0, 2.0])

    try:
        x = jnp.linalg.solve(A, b)
        # Verify: Ax should equal b
        result = jnp.dot(A, x)
        assert np.allclose(result, b, atol=1e-5), f"solve verification failed: {result}"
    except Exception as e:
        # Solve may not be implemented
        raise AssertionError(f"solve not supported: {e}")


def test_cholesky():
    """Test Cholesky decomposition (if supported)."""
    # Positive definite matrix
    A = jnp.array([[4.0, 2.0], [2.0, 5.0]])

    try:
        L = jnp.linalg.cholesky(A)
        # Verify: L @ L.T should equal A
        result = jnp.dot(L, L.T)
        assert np.allclose(result, A, atol=1e-5), f"cholesky verification failed"
    except Exception as e:
        raise AssertionError(f"cholesky not supported: {e}")


# =============================================================================
# SECTION 8: Reduction Operations
# =============================================================================

def test_sum_axes():
    """Test sum reduction along various axes."""
    a = jnp.arange(24).reshape(2, 3, 4)

    # Full reduction
    result = jnp.sum(a)
    assert np.isclose(result, 276), f"sum all failed: {result}"

    # Axis 0
    result = jnp.sum(a, axis=0)
    assert result.shape == (3, 4), f"sum axis=0 shape wrong: {result.shape}"

    # Axis 1
    result = jnp.sum(a, axis=1)
    assert result.shape == (2, 4), f"sum axis=1 shape wrong: {result.shape}"

    # Multiple axes
    result = jnp.sum(a, axis=(0, 2))
    assert result.shape == (3,), f"sum axis=(0,2) shape wrong: {result.shape}"


def test_mean_axes():
    """Test mean reduction along various axes."""
    a = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

    result = jnp.mean(a)
    assert np.isclose(result, 5.5), f"mean all failed: {result}"

    result = jnp.mean(a, axis=0)
    expected = np.array([4.0, 5.0, 6.0, 7.0])
    assert np.allclose(result, expected), f"mean axis=0 failed: {result}"


def test_prod():
    """Test product reduction."""
    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = jnp.prod(a)
    assert np.isclose(result, 24.0), f"prod failed: {result}"


def test_min_max():
    """Test min/max reductions."""
    a = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])

    assert np.isclose(jnp.min(a), 1.0), "min failed"
    assert np.isclose(jnp.max(a), 9.0), "max failed"
    assert np.isclose(jnp.argmin(a), 1), "argmin failed"
    assert np.isclose(jnp.argmax(a), 5), "argmax failed"


def test_all_any():
    """Test all/any reductions."""
    a = jnp.array([True, True, True])
    b = jnp.array([True, False, True])
    c = jnp.array([False, False, False])

    assert jnp.all(a) == True, "all(True,True,True) failed"
    assert jnp.all(b) == False, "all(True,False,True) failed"
    assert jnp.any(b) == True, "any(True,False,True) failed"
    assert jnp.any(c) == False, "any(False,False,False) failed"


# =============================================================================
# SECTION 9: Buffer Donation Tests
# =============================================================================

def test_donation_basic():
    """Test basic buffer donation."""
    @jax.jit
    def f(x):
        return x + 1

    x = jnp.array([1.0, 2.0, 3.0])
    result = f(x)
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(result, expected), f"donation basic failed: {result}"


def test_donation_with_donate_argnums():
    """Test explicit buffer donation."""
    @jax.jit
    def f(x):
        return x * 2

    # Create array and donate it
    x = jnp.ones((100, 100))
    result = jax.jit(f, donate_argnums=(0,))(x)

    # Result should be correct
    assert np.allclose(result, 2.0), "donation with donate_argnums failed"


# =============================================================================
# Main test runner
# =============================================================================

def run_all_tests(verbose: bool = True):
    """Run all tests."""
    print(f"JAX version: {jax.__version__}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Devices: {jax.devices()}")
    print()

    # Section 1: Determinism
    print("=== Section 1: Determinism/Stability Tests ===")
    run_test("determinism_eye", verbose)(test_determinism_eye)()
    run_test("determinism_zeros", verbose)(test_determinism_zeros)()
    run_test("determinism_ones", verbose)(test_determinism_ones)()
    run_test("determinism_arange", verbose)(test_determinism_arange)()
    run_test("determinism_matmul", verbose)(test_determinism_matmul)()

    # Section 2: Constants
    print("\n=== Section 2: Identity/Constant Operations ===")
    run_test("eye_sizes", verbose)(test_eye_sizes)()
    run_test("eye_rectangular", verbose)(test_eye_rectangular)()
    run_test("eye_with_k", verbose)(test_eye_with_k)()
    run_test("identity", verbose)(test_identity)()
    run_test("zeros_shapes", verbose)(test_zeros_shapes)()
    run_test("ones_shapes", verbose)(test_ones_shapes)()
    run_test("full", verbose)(test_full)()
    run_test("linspace", verbose)(test_linspace)()
    run_test("arange_variants", verbose)(test_arange_variants)()

    # Section 3: Data Types
    print("\n=== Section 3: Data Type Coverage ===")
    run_test("dtype_float16", verbose)(test_dtype_float16)()
    run_test("dtype_bfloat16", verbose)(test_dtype_bfloat16)()
    run_test("dtype_float32", verbose)(test_dtype_float32)()
    # float64 not supported on Metal - mark as skipped
    results.add_skip("dtype_float64", "Metal does not support float64")
    if verbose:
        print("Skipping dtype_float64... SKIP (Metal limitation)")
    run_test("dtype_int32", verbose)(test_dtype_int32)()
    # int64 not supported on Metal - mark as skipped
    results.add_skip("dtype_int64", "Metal does not support int64")
    if verbose:
        print("Skipping dtype_int64... SKIP (Metal limitation)")
    run_test("dtype_bool", verbose)(test_dtype_bool)()
    run_test("dtype_casting", verbose)(test_dtype_casting)()
    run_test("dtype_complex", verbose)(test_dtype_complex)()

    # Section 4: Control Flow
    print("\n=== Section 4: Control Flow ===")
    run_test("cond_true", verbose)(test_cond_true)()
    run_test("cond_false", verbose)(test_cond_false)()
    run_test("while_loop", verbose)(test_while_loop)()
    run_test("fori_loop", verbose)(test_fori_loop)()
    run_test("scan", verbose)(test_scan)()
    run_test("select", verbose)(test_select)()

    # Section 5: Scatter/Gather
    print("\n=== Section 5: Scatter/Gather Operations ===")
    run_test("gather_1d", verbose)(test_gather_1d)()
    run_test("gather_2d", verbose)(test_gather_2d)()
    run_test("scatter_1d", verbose)(test_scatter_1d)()
    run_test("scatter_add", verbose)(test_scatter_add)()
    run_test("dynamic_slice", verbose)(test_dynamic_slice)()
    run_test("dynamic_update_slice", verbose)(test_dynamic_update_slice)()

    # Section 6: Random Numbers
    print("\n=== Section 6: Random Number Generation ===")
    run_test("random_uniform", verbose)(test_random_uniform)()
    run_test("random_normal", verbose)(test_random_normal)()
    run_test("random_split", verbose)(test_random_split)()
    run_test("random_choice", verbose)(test_random_choice)()
    run_test("random_permutation", verbose)(test_random_permutation)()

    # Section 7: Linear Algebra
    print("\n=== Section 7: Linear Algebra ===")
    run_test("matmul_shapes", verbose)(test_matmul_shapes)()
    run_test("batched_matmul", verbose)(test_batched_matmul)()
    run_test("transpose", verbose)(test_transpose)()
    run_test("trace", verbose)(test_trace)()
    run_test("diag", verbose)(test_diag)()
    run_test("outer", verbose)(test_outer)()
    run_test("einsum", verbose)(test_einsum)()
    run_test("norm", verbose)(test_norm)()
    run_test("solve", verbose)(test_solve)()
    run_test("cholesky", verbose)(test_cholesky)()

    # Section 8: Reductions
    print("\n=== Section 8: Reduction Operations ===")
    run_test("sum_axes", verbose)(test_sum_axes)()
    run_test("mean_axes", verbose)(test_mean_axes)()
    run_test("prod", verbose)(test_prod)()
    run_test("min_max", verbose)(test_min_max)()
    run_test("all_any", verbose)(test_all_any)()

    # Section 9: Buffer Donation
    print("\n=== Section 9: Buffer Donation ===")
    run_test("donation_basic", verbose)(test_donation_basic)()
    run_test("donation_with_donate_argnums", verbose)(test_donation_with_donate_argnums)()

    # Print summary
    print(results.summary())

    return len(results.failed) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive Metal PJRT tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet output")
    args = parser.parse_args()

    success = run_all_tests(verbose=not args.quiet)
    sys.exit(0 if success else 1)
