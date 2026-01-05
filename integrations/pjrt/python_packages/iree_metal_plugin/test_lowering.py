#!/usr/bin/env python3
"""Test that the spsolve lowering produces correct MLIR."""

import os

# Point to local compiler
os.environ["IREE_PJRT_COMPILER_LIB_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "../../../../compiler/build/b/lib/libIREECompiler.dylib"
)

import numpy as np
import jax
import jax.numpy as jnp

# Register the lowerings
from jax_plugins.iree_metal import _register_linalg_lowerings
_register_linalg_lowerings()

# Test if spsolve primitive exists
try:
    from jax.experimental.sparse.linalg import spsolve_p
    print(f"Found spsolve primitive: {spsolve_p}")
except ImportError as e:
    print(f"Cannot import spsolve_p: {e}")
    exit(1)

# Try to lower a simple call
print("\nTesting lowering...")

def test_fn(data, indices, indptr, b):
    from jax.experimental.sparse import linalg as sparse_linalg
    return sparse_linalg.spsolve(data, indices, indptr, b)

# Create sample inputs
data = jnp.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0], dtype=jnp.float32)
indices = jnp.array([0, 1, 0, 1, 2, 1, 2], dtype=jnp.int32)
indptr = jnp.array([0, 2, 5, 7], dtype=jnp.int32)
b = jnp.array([6.0, 12.0, 14.0], dtype=jnp.float32)

# Get the jaxpr
print("Lowering to jaxpr...")
closed_jaxpr = jax.make_jaxpr(test_fn)(data, indices, indptr, b)
print(closed_jaxpr)

# Try to lower to HLO (without running)
print("\nLowering to HLO...")
lowered = jax.jit(test_fn).lower(data, indices, indptr, b)
print(lowered.as_text()[:2000])
print("...")
