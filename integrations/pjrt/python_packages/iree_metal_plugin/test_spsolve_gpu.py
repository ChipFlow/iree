#!/usr/bin/env python3
"""Test sparse solve on IREE Metal GPU backend.

This tests the full pipeline:
1. JAX spsolve with CSR matrix
2. MLIR lowering emits stablehlo.custom_call("iree_spsolve")
3. StableHLOCustomCalls converts to sparse_solver.spsolve
4. SparseSolver dialect conversion to VM calls
5. BaSpaCho runtime executes on GPU
"""

import os
import sys

# Set up environment for IREE Metal
os.environ["JAX_PLATFORMS"] = "iree_metal"
os.environ["IREE_PJRT_COMPILER_LIB_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "../../../../compiler/build/b/lib/libIREECompiler.dylib"
)

import numpy as np

# Import JAX after setting platform
import jax
import jax.numpy as jnp
from jax.experimental import sparse


def create_csr_matrix():
    """Create a simple 3x3 sparse matrix in CSR format.

    Matrix A:
    [[ 4  1  0 ]
     [ 1  4  1 ]
     [ 0  1  4 ]]

    This is a tridiagonal SPD matrix (Poisson-like).
    """
    data = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    indptr = np.array([0, 2, 5, 7], dtype=np.int32)
    return data, indices, indptr


def test_spsolve():
    """Test sparse solve with known solution."""
    print("Testing sparse solve on IREE Metal GPU...")

    # Create sparse matrix
    data, indices, indptr = create_csr_matrix()
    n = len(indptr) - 1

    # Known solution
    x_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Compute b = A @ x
    # A = [[4,1,0], [1,4,1], [0,1,4]]
    # b = [4*1+1*2, 1*1+4*2+1*3, 1*2+4*3] = [6, 12, 14]
    b = np.array([6.0, 12.0, 14.0], dtype=np.float32)

    print(f"Matrix dimension: {n}x{n}")
    print(f"Number of non-zeros: {len(data)}")
    print(f"True solution: {x_true}")
    print(f"RHS vector b: {b}")

    # Convert to JAX arrays
    data_jax = jnp.array(data)
    indices_jax = jnp.array(indices)
    indptr_jax = jnp.array(indptr)
    b_jax = jnp.array(b)

    # Create CSR matrix object
    csr = sparse.CSR((data_jax, indices_jax, indptr_jax), shape=(n, n))

    # Solve Ax = b
    print("\nSolving Ax = b...")
    try:
        x = sparse.linalg.spsolve(csr.data, csr.indices, csr.indptr, b_jax)
        x_np = np.array(x)

        print(f"Computed solution: {x_np}")
        print(f"Expected solution: {x_true}")

        # Check error
        error = np.linalg.norm(x_np - x_true) / np.linalg.norm(x_true)
        print(f"Relative error: {error:.2e}")

        if error < 1e-5:
            print("\n[PASS] Sparse solve succeeded!")
            return True
        else:
            print(f"\n[FAIL] Error too large: {error}")
            return False

    except Exception as e:
        print(f"\n[FAIL] Exception during solve: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jit_spsolve():
    """Test JIT-compiled sparse solve."""
    print("\n" + "="*60)
    print("Testing JIT-compiled sparse solve...")

    data, indices, indptr = create_csr_matrix()
    n = len(indptr) - 1
    b = np.array([6.0, 12.0, 14.0], dtype=np.float32)
    x_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    @jax.jit
    def solve(data, indices, indptr, b):
        return sparse.linalg.spsolve(data, indices, indptr, b)

    try:
        print("Lowering and compiling...")
        x = solve(
            jnp.array(data),
            jnp.array(indices),
            jnp.array(indptr),
            jnp.array(b)
        )
        x_np = np.array(x)

        print(f"Computed solution: {x_np}")
        error = np.linalg.norm(x_np - x_true) / np.linalg.norm(x_true)
        print(f"Relative error: {error:.2e}")

        if error < 1e-5:
            print("\n[PASS] JIT sparse solve succeeded!")
            return True
        else:
            print(f"\n[FAIL] Error too large: {error}")
            return False

    except Exception as e:
        print(f"\n[FAIL] Exception during JIT solve: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("IREE Metal GPU Sparse Solver Test")
    print("="*60)

    # Show JAX devices
    print(f"\nJAX devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")

    # Run tests
    success = True
    success = test_spsolve() and success
    success = test_jit_spsolve() and success

    print("\n" + "="*60)
    if success:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)
