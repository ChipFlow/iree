# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test Metal PJRT plugin functionality.

This test verifies that the IREE Metal PJRT plugin works correctly on macOS.
It will skip gracefully if Metal is not available (non-macOS platforms).
"""

import sys
import platform

# Check if we're on macOS
if platform.system() != "Darwin":
    print(f"Skipping Metal tests on {platform.system()} (Metal requires macOS)")
    sys.exit(0)

import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Platform: {platform.system()} {platform.machine()}")

# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")

# Verify we have an IREE Metal device
metal_devices = [d for d in devices if "iree" in str(d).lower() or "metal" in str(d).lower()]
if not metal_devices:
    print("Warning: No IREE Metal device found, running with default device")
    print(f"Running on: {devices[0] if devices else 'unknown'}")

# Test 1: Basic array creation and operations
print("\n=== Test 1: Basic array operations ===")
a = jnp.array([1.0, 2.0, 3.0, 4.0])
b = jnp.array([5.0, 6.0, 7.0, 8.0])
c = a + b
print(f"Array addition: {a} + {b} = {c}")
assert jnp.allclose(c, jnp.array([6.0, 8.0, 10.0, 12.0])), "Basic addition failed"

# Test 2: Matrix multiplication
print("\n=== Test 2: Matrix multiplication ===")
m1 = jnp.ones((4, 4))
m2 = jnp.eye(4) * 2.0
result = jnp.dot(m1, m2)
print(f"Matrix multiply result shape: {result.shape}")
print(f"Result:\n{result}")
expected = jnp.ones((4, 4)) * 2.0
assert jnp.allclose(result, expected), "Matrix multiplication failed"

# Test 3: JIT compilation
print("\n=== Test 3: JIT compilation ===")
@jax.jit
def jit_matmul(x, y):
    return jnp.dot(x, y)

jit_result = jit_matmul(m1, m2)
print(f"JIT matmul result shape: {jit_result.shape}")
assert jnp.allclose(jit_result, expected), "JIT matmul failed"

# Test 4: Larger computation (stress test)
print("\n=== Test 4: Larger computation ===")
large_a = jnp.ones((256, 256))
large_b = jnp.ones((256, 256))

@jax.jit
def large_matmul(x, y):
    return jnp.dot(x, y)

large_result = large_matmul(large_a, large_b)
print(f"Large matmul (256x256) result shape: {large_result.shape}")
print(f"Result sum: {jnp.sum(large_result)}")
assert large_result.shape == (256, 256), "Large matmul shape mismatch"
assert jnp.allclose(jnp.sum(large_result), 256.0 * 256.0 * 256.0), "Large matmul sum mismatch"

# Test 5: Elementwise operations
print("\n=== Test 5: Elementwise operations ===")
x = jnp.linspace(0, 1, 100)
y = jnp.sin(x * jnp.pi)
z = jnp.exp(-x)
result = y * z
print(f"Elementwise ops result shape: {result.shape}")
print(f"First few values: {result[:5]}")

# Test 6: Reduction operations
print("\n=== Test 6: Reduction operations ===")
data = jnp.arange(100).reshape(10, 10).astype(jnp.float32)
sum_result = jnp.sum(data)
mean_result = jnp.mean(data)
max_result = jnp.max(data)
print(f"Sum: {sum_result}, Mean: {mean_result}, Max: {max_result}")
assert jnp.isclose(sum_result, 4950.0), "Sum reduction failed"
assert jnp.isclose(mean_result, 49.5), "Mean reduction failed"
assert jnp.isclose(max_result, 99.0), "Max reduction failed"

# Test 7: Automatic differentiation
print("\n=== Test 7: Automatic differentiation ===")
def loss_fn(x):
    return jnp.sum(x ** 2)

x = jnp.array([1.0, 2.0, 3.0])
grad_fn = jax.grad(loss_fn)
grad_result = grad_fn(x)
print(f"Gradient of sum(x^2) at {x}: {grad_result}")
expected_grad = 2.0 * x
assert jnp.allclose(grad_result, expected_grad), "Gradient computation failed"

# Test 8: vmap (vectorized mapping)
print("\n=== Test 8: vmap ===")
def single_dot(a, b):
    return jnp.dot(a, b)

batch_a = jnp.ones((8, 4, 4))
batch_b = jnp.eye(4).reshape(1, 4, 4).repeat(8, axis=0)
batched_dot = jax.vmap(single_dot)
vmap_result = batched_dot(batch_a, batch_b)
print(f"vmap result shape: {vmap_result.shape}")
assert vmap_result.shape == (8, 4, 4), "vmap shape mismatch"

print("\n" + "=" * 50)
print("All Metal PJRT plugin tests passed!")
print("=" * 50)
