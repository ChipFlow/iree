#!/usr/bin/env python3
"""Test sparse solve with a large sparse system.

Uses a 2D Poisson problem (5-point stencil Laplacian) which generates
a symmetric positive definite sparse matrix.
"""

import os
import sys
import time

# Set up environment for IREE Metal
os.environ["JAX_PLATFORMS"] = "iree_metal"
os.environ["IREE_PJRT_COMPILER_LIB_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "../../../../compiler/build/b/lib/libIREECompiler.dylib"
)

import numpy as np
import scipy.sparse as sp

# Import JAX after setting platform
import jax
import jax.numpy as jnp
from jax.experimental import sparse


def create_poisson_2d(n):
    """Create 2D Poisson matrix (5-point stencil) of size n^2 x n^2.

    This is a classic sparse SPD matrix from discretizing:
        -nabla^2 u = f
    on a unit square with Dirichlet boundary conditions.

    The matrix has:
    - n^2 rows/columns
    - ~5 * n^2 non-zeros (sparse!)
    - Condition number O(n^2)
    """
    # Create 1D second derivative matrix (n x n)
    main_diag = -2 * np.ones(n)
    off_diag = np.ones(n - 1)
    T = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n, n), format='csr')

    # Create 2D Laplacian via Kronecker product
    I = sp.eye(n, format='csr')
    A = sp.kron(I, T) + sp.kron(T, I)

    # Make it positive definite by negating (Laplacian has negative eigenvalues)
    A = -A

    return A.tocsr()


def test_large_spsolve(grid_size=100):
    """Test sparse solve with a large Poisson matrix."""
    n = grid_size
    N = n * n  # Total matrix dimension

    print(f"Creating {N}x{N} sparse Poisson matrix (grid: {n}x{n})...")
    A = create_poisson_2d(n)
    nnz = A.nnz

    print(f"Matrix dimension: {N}x{N}")
    print(f"Number of non-zeros: {nnz}")
    print(f"Sparsity: {100 * nnz / (N*N):.4f}%")
    print(f"Avg non-zeros per row: {nnz / N:.1f}")

    # Create a known solution and compute RHS
    x_true = np.sin(np.linspace(0, 2*np.pi, N)).astype(np.float32)
    b = (A @ x_true).astype(np.float32)

    # Extract CSR components
    data = A.data.astype(np.float32)
    indices = A.indices.astype(np.int32)
    indptr = A.indptr.astype(np.int32)

    print(f"\nTrue solution norm: {np.linalg.norm(x_true):.6f}")
    print(f"RHS norm: {np.linalg.norm(b):.6f}")

    # Convert to JAX arrays
    data_jax = jnp.array(data)
    indices_jax = jnp.array(indices)
    indptr_jax = jnp.array(indptr)
    b_jax = jnp.array(b)

    # Solve using IREE Metal
    print("\nSolving with IREE Metal + BaSpaCho...")

    @jax.jit
    def solve(data, indices, indptr, b):
        return sparse.linalg.spsolve(data, indices, indptr, b)

    try:
        # Warm-up / compile
        t0 = time.perf_counter()
        x = solve(data_jax, indices_jax, indptr_jax, b_jax)
        x.block_until_ready()
        compile_time = time.perf_counter() - t0
        print(f"First call (compile + solve): {compile_time*1000:.1f} ms")

        # Timed run
        t0 = time.perf_counter()
        x = solve(data_jax, indices_jax, indptr_jax, b_jax)
        x.block_until_ready()
        solve_time = time.perf_counter() - t0
        print(f"Second call (solve only): {solve_time*1000:.2f} ms")

        x_np = np.array(x)

        # Compute errors
        abs_error = np.linalg.norm(x_np - x_true)
        rel_error = abs_error / np.linalg.norm(x_true)

        # Compute residual: ||Ax - b|| / ||b||
        residual = A @ x_np - b
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(b)

        print(f"\nResults:")
        print(f"  Absolute error: {abs_error:.2e}")
        print(f"  Relative error: {rel_error:.2e}")
        print(f"  Relative residual: {rel_residual:.2e}")

        if rel_error < 1e-3:
            print(f"\n[PASS] Large sparse solve succeeded!")
            return True
        else:
            print(f"\n[FAIL] Error too large")
            return False

    except Exception as e:
        print(f"\n[FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scaling():
    """Test how solve time scales with problem size."""
    print("\n" + "="*60)
    print("Scaling Test")
    print("="*60)

    sizes = [20, 50, 100, 150]

    @jax.jit
    def solve(data, indices, indptr, b):
        return sparse.linalg.spsolve(data, indices, indptr, b)

    for n in sizes:
        N = n * n
        A = create_poisson_2d(n)

        x_true = np.ones(N, dtype=np.float32)
        b = (A @ x_true).astype(np.float32)

        data_jax = jnp.array(A.data.astype(np.float32))
        indices_jax = jnp.array(A.indices.astype(np.int32))
        indptr_jax = jnp.array(A.indptr.astype(np.int32))
        b_jax = jnp.array(b)

        try:
            # Warm-up
            x = solve(data_jax, indices_jax, indptr_jax, b_jax)
            x.block_until_ready()

            # Timed run (average of 3)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                x = solve(data_jax, indices_jax, indptr_jax, b_jax)
                x.block_until_ready()
                times.append(time.perf_counter() - t0)

            avg_time = np.mean(times) * 1000
            x_np = np.array(x)
            rel_error = np.linalg.norm(x_np - x_true) / np.linalg.norm(x_true)

            print(f"Grid {n:3d}x{n:3d} (N={N:5d}, nnz={A.nnz:6d}): "
                  f"{avg_time:7.2f} ms, error={rel_error:.2e}")
        except Exception as e:
            print(f"Grid {n:3d}x{n:3d}: FAILED - {e}")


if __name__ == "__main__":
    print("="*60)
    print("IREE Metal Large Sparse Solver Test")
    print("="*60)

    # Show JAX devices
    print(f"\nJAX devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")

    # Run main test with 100x100 grid (10,000 unknowns)
    success = test_large_spsolve(grid_size=100)

    # Run scaling test
    if success:
        test_scaling()

    print("\n" + "="*60)
    sys.exit(0 if success else 1)
