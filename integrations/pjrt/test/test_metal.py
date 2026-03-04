# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test Metal PJRT plugin functionality.

These tests run EXCLUSIVELY on the IREE Metal backend.
Run: uv run python -m pytest test_metal.py -v

The test configures JAX_PLATFORMS=iree_metal internally — no env var needed.
"""

import os
import platform
import sys

import pytest

# Skip on non-macOS
if platform.system() != "Darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)

# Configure Metal backend BEFORE importing JAX
os.environ["JAX_PLATFORMS"] = "iree_metal"

import jax
import jax.numpy as jnp

# Verify we actually have a Metal device.
# With JAX_PLATFORMS=iree_metal, any device found IS a Metal device.
# The device string may be an opaque ID like "00000001000004ce".
_devices = jax.devices()
if not _devices:
    pytest.skip("No IREE Metal device available", allow_module_level=True)


class TestMetalBasic:
    """Basic Metal PJRT plugin tests — all ops run on Metal GPU."""

    def test_array_addition(self):
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([5.0, 6.0, 7.0, 8.0])
        c = a + b
        assert jnp.allclose(c, jnp.array([6.0, 8.0, 10.0, 12.0]))

    def test_matmul(self):
        m1 = jnp.ones((4, 4))
        m2 = jnp.eye(4) * 2.0
        result = jnp.dot(m1, m2)
        expected = jnp.ones((4, 4)) * 2.0
        assert jnp.allclose(result, expected)

    def test_jit_matmul(self):
        @jax.jit
        def jit_matmul(x, y):
            return jnp.dot(x, y)

        m1 = jnp.ones((4, 4))
        m2 = jnp.eye(4) * 2.0
        result = jit_matmul(m1, m2)
        expected = jnp.ones((4, 4)) * 2.0
        assert jnp.allclose(result, expected)

    def test_large_matmul(self):
        large_a = jnp.ones((256, 256))
        large_b = jnp.ones((256, 256))

        @jax.jit
        def large_matmul(x, y):
            return jnp.dot(x, y)

        result = large_matmul(large_a, large_b)
        assert result.shape == (256, 256)
        assert jnp.allclose(jnp.sum(result), 256.0 * 256.0 * 256.0)

    def test_elementwise(self):
        x = jnp.linspace(0, 1, 100)
        y = jnp.sin(x * jnp.pi)
        z = jnp.exp(-x)
        result = y * z
        assert result.shape == (100,)

    def test_reduction(self):
        data = jnp.arange(100).reshape(10, 10).astype(jnp.float32)
        assert jnp.isclose(jnp.sum(data), 4950.0)
        assert jnp.isclose(jnp.mean(data), 49.5)
        assert jnp.isclose(jnp.max(data), 99.0)

    def test_grad(self):
        def loss_fn(x):
            return jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        grad_result = jax.grad(loss_fn)(x)
        assert jnp.allclose(grad_result, 2.0 * x)

    def test_vmap(self):
        def single_dot(a, b):
            return jnp.dot(a, b)

        batch_a = jnp.ones((8, 4, 4))
        batch_b = jnp.eye(4).reshape(1, 4, 4).repeat(8, axis=0)
        result = jax.vmap(single_dot)(batch_a, batch_b)
        assert result.shape == (8, 4, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
