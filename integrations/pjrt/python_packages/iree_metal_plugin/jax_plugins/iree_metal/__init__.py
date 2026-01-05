# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
from pathlib import Path
import platform
import sys

import numpy as np

import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)


def _lu_iree_metal(a):
    """Pure JAX LU decomposition that avoids problematic scatter patterns.

    This is an unblocked LU with partial pivoting, designed to work
    correctly on IREE Metal backend by avoiding the gather/scatter patterns
    that IREE doesn't handle properly (specifically scatter with empty indices).

    Uses unrolled loops which works well for small-to-medium matrices.
    """
    import jax
    import jax.numpy as jnp

    m, n = a.shape
    k = min(m, n)

    # Initialize outputs
    pivot = jnp.zeros(k, dtype=jnp.int32)
    perm = jnp.arange(m, dtype=jnp.int32)

    def body(state, i):
        a, pivot, perm = state

        # Find pivot row (row with max abs value in column i, from row i onwards)
        col = jnp.abs(a[i:, i])
        j = jnp.argmax(col) + i

        # Swap rows i and j in matrix
        row_i = a[i, :]
        row_j = a[j, :]
        a = a.at[i, :].set(row_j)
        a = a.at[j, :].set(row_i)

        # Update pivot and permutation
        pivot = pivot.at[i].set(j)
        perm_i = perm[i]
        perm_j = perm[j]
        perm = perm.at[i].set(perm_j)
        perm = perm.at[j].set(perm_i)

        # For rows below i: compute multipliers and update submatrix
        for row_idx in range(i + 1, m):
            multiplier = a[row_idx, i] / a[i, i]
            a = a.at[row_idx, i].set(multiplier)
            for col_idx in range(i + 1, n):
                a = a.at[row_idx, col_idx].add(-multiplier * a[i, col_idx])

        return (a, pivot, perm), None

    # Unroll the loop (works for small matrices, JAX will trace through)
    for i in range(k):
        (a, pivot, perm), _ = body((a, pivot, perm), i)

    return a, pivot, perm


def _qr_iree_metal(a, *, full_matrices, pivoting, use_magma):
    """Pure JAX Householder QR decomposition.

    This implements the Householder QR algorithm directly in JAX,
    avoiding the custom_call @Qr that IREE doesn't support.

    Uses full matrix operations to avoid dynamic scatter/gather ops
    that IREE has trouble with.

    Note: Uses jnp.diag(jnp.ones(m)) instead of jnp.eye(m) to work around
    an IREE Metal codegen bug where eye + matmul fusion causes sitofp errors.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    if pivoting:
        raise NotImplementedError(
            "Column-pivoted QR (geqp3) is not yet supported on IREE Metal"
        )

    m, n = a.shape
    k = min(m, n)

    r = a.astype(a.dtype)

    # Helper to create identity matrix - use diag(ones) to avoid IREE Metal bug
    # where jnp.eye() followed by matmul causes sitofp errors
    def identity(size, dtype):
        return jnp.diag(jnp.ones(size, dtype=dtype))

    # Start with Q = I, we'll build it up as a product of Householder matrices
    q = identity(m, a.dtype)

    # Householder QR: for each column, compute and apply reflector
    for j in range(k):
        # Extract column j from row j onwards
        # Use + 0 to force a copy - workaround for IREE bug where
        # .at[].add() on a slice returns wrong results
        x = r[j:, j] + 0

        # Compute norm of x
        norm_x = jnp.sqrt(jnp.sum(x * x))

        # Sign for numerical stability - we want sign(x[0]) or 1 if x[0] == 0
        # To avoid boolean-to-float conversion, use: x[0] / (|x[0]| + eps)
        # This gives approximately sign(x[0]) and avoids division by zero
        sign_x0 = x[0] / (jnp.abs(x[0]) + 1e-30)
        alpha = sign_x0 * norm_x

        # Build Householder vector v (in the subspace from j onwards)
        # v = x + sign(x[0]) * ||x|| * e_1
        v = x.at[0].add(alpha)

        # Compute tau = 2 / (v^T * v)
        norm_v_sq = jnp.sum(v * v)
        # Avoid division by zero - add tiny epsilon
        tau = 2.0 / (norm_v_sq + 1e-30)

        # Build the full m×m Householder matrix H = I - tau * v_full * v_full^T
        # where v_full is v padded with zeros in positions 0..j-1
        # Use concatenate to avoid dynamic scatter and potential boolean ops in pad
        zeros_prefix = jnp.zeros(j, dtype=a.dtype)
        v_full = jnp.concatenate([zeros_prefix, v])

        # H = I - tau * v_full * v_full^T
        # Use diag(ones) instead of eye to avoid IREE Metal bug
        H = identity(m, a.dtype) - tau * jnp.outer(v_full, v_full)

        # Update R = H @ R and Q = Q @ H
        r = H @ r
        q = q @ H

    # Extract upper triangular part of R
    if full_matrices:
        r = jnp.triu(r)
    else:
        q = q[:, :k]
        r = jnp.triu(r[:k, :])

    return q, r


def _scipy_solve_iree_metal(a, b, lower=False, overwrite_a=False, overwrite_b=False,
                            debug=False, check_finite=True, assume_a='gen'):
    """Replacement for jax.scipy.linalg.solve that avoids nested JIT issue.

    IREE Metal has a bug where nested JIT with static arguments causes
    FlatBuffer verification errors. This implementation avoids that by
    providing a non-nested solve implementation.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax._src.lax import linalg as lax_linalg
    from jax._src.numpy.util import promote_dtypes_inexact

    del overwrite_a, overwrite_b, debug, check_finite  # unused

    a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))
    lax_linalg._check_solve_shapes(a, b)

    if assume_a == 'pos':
        # Use Cholesky for positive-definite matrices
        # chol @ chol.T @ x = b
        chol = jnp.linalg.cholesky(a)
        # Solve chol @ y = b for y, then chol.T @ x = y for x
        y = lax_linalg.triangular_solve(chol, b, left_side=True, lower=True)
        x = lax_linalg.triangular_solve(chol, y, left_side=True, lower=True,
                                         transpose_a=True)
        return x
    else:
        # Use LU decomposition for general matrices
        # Broadcast leading dimensions of b to the shape of a
        out_shape = tuple(d_a if d_b == 1 else d_b
                          for d_a, d_b in zip(a.shape[:-1] + (1,), b.shape))
        b = lax.broadcast_in_dim(b, out_shape, range(b.ndim))

        lu_, _, permutation = lax_linalg.lu(a)

        # Apply permutation to b
        # P @ b where P is the permutation matrix
        m = a.shape[-1]
        row_indices = jnp.arange(m, dtype=permutation.dtype)[:, None]
        col_indices = permutation[None, :]
        P = jnp.where(row_indices == col_indices,
                      jnp.ones((), dtype=a.dtype),
                      jnp.zeros((), dtype=a.dtype))
        pb = P @ b

        # Solve L @ y = P @ b for y (forward substitution)
        L = jnp.tril(lu_, -1) + jnp.eye(m, dtype=a.dtype)
        y = lax_linalg.triangular_solve(L, pb, left_side=True, lower=True)

        # Solve U @ x = y for x (back substitution)
        U = jnp.triu(lu_)
        x = lax_linalg.triangular_solve(U, y, left_side=True, lower=False)

        return x.squeeze(-1) if a.ndim == b.ndim else x


def _scipy_lu_iree_metal(a, permute_l=False, overwrite_a=False, check_finite=True):
    """Replacement for jax.scipy.linalg.lu that avoids nested JIT issue.

    IREE Metal has a bug where nested JIT with static arguments causes
    FlatBuffer verification errors. This implementation avoids that by
    not using a nested JIT structure.

    Also avoids boolean-to-float conversion (arith.sitofp) which IREE Metal
    doesn't support, by using jnp.where instead of direct boolean array conversion.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax._src.lax import linalg as lax_linalg

    del overwrite_a, check_finite  # unused

    # Call the primitive directly (our custom lowering handles it)
    lu, _, permutation = lax_linalg.lu(a)
    dtype = lax.dtype(a)
    m, n = a.shape
    k = min(m, n)

    # Build permutation matrix without boolean-to-float conversion
    # Use jnp.where instead of direct boolean array conversion to avoid
    # arith.sitofp which IREE Metal doesn't support
    row_indices = jnp.arange(m, dtype=permutation.dtype)[:, None]
    col_indices = permutation[None, :]
    # Create 1.0 where row == col, 0.0 elsewhere
    p = jnp.where(row_indices == col_indices, jnp.ones((), dtype=dtype), jnp.zeros((), dtype=dtype))

    # Extract L and U
    l = jnp.tril(lu, -1)[:, :k] + jnp.eye(m, k, dtype=dtype)
    u = jnp.triu(lu)[:k, :]

    if permute_l:
        return jnp.matmul(p, l, precision=lax.Precision.HIGHEST), u
    else:
        return p, l, u


def _spsolve_iree_metal_lowering(ctx, data, indices, indptr, b, *, tol, reorder):
    """MLIR lowering for spsolve that emits mhlo.custom_call.

    Routes to the BaSpaCho sparse solver runtime module:
    1. Emit MHLO custom_call("iree_spsolve")
    2. StableHLOCustomCalls converts to sparse_solver.spsolve op
    3. SparseSolver to VM conversion generates vm.call
    4. VM calls @sparse_solver.spsolve_complete runtime function
    5. BaSpaCho executes on GPU (Metal/CUDA/OpenCL)

    The BaSpaCho solver uses LU factorization with partial pivoting
    for general sparse matrices, or Cholesky for SPD matrices.

    See: iree/compiler/Dialect/SparseSolver/ for compiler dialect
         iree/modules/sparse_solver/ for runtime module
    """
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import mhlo
    from jax._src.interpreters import mlir as jax_mlir

    del tol, reorder  # Currently not passed through to BaSpaCho

    # Get result type from context
    result_type = jax_mlir.aval_to_ir_type(ctx.avals_out[0])

    # Emit mhlo.custom_call which will be converted by StableHLOCustomCalls
    # Note: MHLO gets converted to StableHLO by JAX before reaching IREE
    result = mhlo.custom_call(
        [result_type],
        [data, indices, indptr, b],
        call_target_name=ir.StringAttr.get("iree_spsolve"),
        has_side_effect=ir.BoolAttr.get(False),
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 4),  # TYPED_FFI
    )

    # mhlo.custom_call with single result returns the result directly, not a list
    return [result]


def _spsolve_iree_metal_dense(data, indices, indptr, b):
    """Dense fallback for sparse solve when GPU path unavailable.

    This converts CSR to dense and uses LU solve.
    Works but is O(n²) in memory and doesn't leverage GPU sparsity.
    """
    import jax.numpy as jnp
    from jax import vmap

    n = b.shape[0]

    # Convert CSR to dense matrix
    def csr_to_dense_element(row_col):
        row, col = row_col
        row_start = indptr[row]
        row_end = indptr[row + 1]

        # Linear search in the row (up to 64 elements per row)
        result = jnp.float32(0.0)
        for offset in range(64):
            pos = row_start + offset
            in_range = pos < row_end
            safe_pos = jnp.where(in_range, pos, 0)
            stored_col = indices[safe_pos]
            stored_val = data[safe_pos]
            match = jnp.logical_and(in_range, stored_col == col)
            result = jnp.where(match, stored_val, result)
        return result

    # Build dense matrix using vmap
    rows = jnp.arange(n)
    cols = jnp.arange(n)
    row_grid, col_grid = jnp.meshgrid(rows, cols, indexing='ij')
    row_col_pairs = jnp.stack([row_grid.ravel(), col_grid.ravel()], axis=1)

    # Use vmap to compute all elements
    dense_flat = vmap(csr_to_dense_element)(row_col_pairs)
    dense = dense_flat.reshape((n, n))

    # Use dense solve (which uses our working LU implementation)
    return jnp.linalg.solve(dense, b)


def _register_linalg_lowerings():
    """Register platform-specific lowerings for linear algebra operations.

    Instead of using LAPACK FFI custom calls (which have bugs in IREE's rewriter),
    we use a pure JAX implementation that avoids problematic scatter patterns.

    Also patches jax.scipy.linalg.lu to avoid nested JIT issues on IREE Metal.
    """
    from jax._src.interpreters import mlir
    from jax._src.lax import linalg as lax_linalg
    import jax.scipy.linalg

    # Register the lowering for LU using our pure JAX implementation
    mlir.register_lowering(
        lax_linalg.lu_p,
        mlir.lower_fun(_lu_iree_metal, multiple_results=True),
        platform="iree_metal"
    )

    # Register the lowering for QR using our pure JAX implementation
    mlir.register_lowering(
        lax_linalg.qr_p,
        mlir.lower_fun(_qr_iree_metal, multiple_results=True),
        platform="iree_metal"
    )

    # Register the lowering for sparse solve (spsolve)
    # Uses BaSpaCho GPU-accelerated solver via iree_spsolve custom call
    try:
        from jax.experimental.sparse.linalg import spsolve_p
        mlir.register_lowering(
            spsolve_p,
            _spsolve_iree_metal_lowering,
            platform="iree_metal"
        )
        logger.debug("Registered IREE Metal lowering for sparse solve (BaSpaCho GPU)")
    except ImportError:
        logger.debug("jax.experimental.sparse not available, skipping spsolve lowering")

    # Patch jax.scipy.linalg functions to avoid nested JIT issue
    # The original functions have nested JIT decorators which cause FlatBuffer
    # verification errors when called from a JIT context on IREE Metal
    jax.scipy.linalg.lu = _scipy_lu_iree_metal
    jax.scipy.linalg.solve = _scipy_solve_iree_metal

    logger.debug("Registered IREE Metal lowerings for linear algebra operations")


def probe_iree_compiler_dylib() -> str:
    """Probes an installed iree.compiler for the compiler dylib.

    On macOS, also checks the IREE_PJRT_COMPILER_LIB_PATH environment variable.
    """
    # Check environment variable first (useful for development builds)
    env_path = os.environ.get("IREE_PJRT_COMPILER_LIB_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Fall back to probing installed iree.compiler
    try:
        from iree.compiler.api import ctypes_dl
        return ctypes_dl._probe_iree_compiler_dylib()
    except ImportError:
        logger.warning(
            "Could not import iree.compiler. Set IREE_PJRT_COMPILER_LIB_PATH "
            "environment variable to point to libIREECompiler.dylib"
        )
        raise


def initialize():
    # Metal is only available on macOS
    if platform.system() != "Darwin":
        logger.warning(
            f"Metal PJRT plugin is only available on macOS, "
            f"but running on {platform.system()}"
        )
        return

    import iree._pjrt_libs.metal as lib_package

    # On macOS, the library is a .dylib
    path = Path(lib_package.__file__).resolve().parent / "pjrt_plugin_iree_metal.dylib"
    if not path.exists():
        # Also try .so extension for compatibility
        path = Path(lib_package.__file__).resolve().parent / "pjrt_plugin_iree_metal.so"
    if not path.exists():
        logger.warning(
            f"WARNING: Native library {path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )
    xb.register_plugin(
        "iree_metal",
        priority=500,
        library_path=str(path),
        options={
            "COMPILER_LIB_PATH": str(probe_iree_compiler_dylib()),
        },
    )

    # Register custom lowerings for linear algebra operations
    # This must be done after the plugin is registered so JAX knows the platform
    _register_linalg_lowerings()
