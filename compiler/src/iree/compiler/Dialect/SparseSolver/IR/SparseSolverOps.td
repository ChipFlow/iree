//===- SparseSolverOps.td - Sparse Solver operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operation definitions for the SparseSolver dialect.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_DIALECT_SPARSESOLVER_IR_SPARSESOLVEROPS_TD
#define IREE_DIALECT_SPARSESOLVER_IR_SPARSESOLVEROPS_TD

include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverDialect.td"
include "iree/compiler/Dialect/HAL/IR/HALTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SparseSolver_Op<string mnemonic, list<Trait> traits = []> :
    Op<SparseSolver_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// sparse_solver.spsolve
//===----------------------------------------------------------------------===//

def SparseSolver_SpsolveOp : SparseSolver_Op<"spsolve", [Pure]> {
  let summary = "Sparse linear solve using LU factorization";
  let description = [{
    Solves Ax = b where A is a sparse matrix in CSR format.

    This operation performs:
    1. Symbolic analysis of the sparsity pattern
    2. Numeric LU factorization with partial pivoting
    3. Forward/backward substitution to solve

    The sparse matrix A is provided in CSR (Compressed Sparse Row) format:
    - `data`: Non-zero values (nnz elements)
    - `indices`: Column indices for each non-zero (nnz elements)
    - `indptr`: Row pointers (n+1 elements, where n is matrix dimension)

    Example:
    ```mlir
    %x = sparse_solver.spsolve %data, %indices, %indptr, %b
        : (tensor<?xf32>, tensor<?xi32>, tensor<?xi32>, tensor<?xf32>)
        -> tensor<?xf32>
    ```

    The operation is lowered to calls into the BaSpaCho sparse solver module,
    which automatically selects the appropriate backend (Metal, CUDA, OpenCL, CPU).

    Note: Operand types may be tensors (before HAL conversion) or buffer_views
    (after HAL conversion). This flexibility allows the op to survive through
    the conversion pipeline until it reaches VM conversion.
  }];

  let arguments = (ins
    AnyType:$data,      // CSR values (nnz,) - tensor or buffer_view
    AnyType:$indices,   // CSR column indices (nnz,) - tensor or buffer_view
    AnyType:$indptr,    // CSR row pointers (n+1,) - tensor or buffer_view
    AnyType:$rhs        // Right-hand side (n,) - tensor or buffer_view
  );

  let results = (outs
    AnyType:$solution   // Solution vector (n,) - tensor or buffer_view
  );

  let assemblyFormat = [{
    $data `,` $indices `,` $indptr `,` $rhs attr-dict
    `:` `(` type($data) `,` type($indices) `,` type($indptr) `,` type($rhs) `)`
    `->` type($solution)
  }];
}

//===----------------------------------------------------------------------===//
// sparse_solver.cholesky_solve (for symmetric positive definite matrices)
//===----------------------------------------------------------------------===//

def SparseSolver_CholeskySolveOp : SparseSolver_Op<"cholesky_solve", [Pure]> {
  let summary = "Sparse Cholesky solve for symmetric positive definite matrices";
  let description = [{
    Solves Ax = b where A is a sparse symmetric positive definite matrix in CSR format.

    Uses Cholesky factorization (A = L * L^T) which is more efficient than LU
    when A is known to be symmetric positive definite.

    The lower triangular part of A should be provided in CSR format.

    Note: Like spsolve, operand types may be tensors or buffer_views depending
    on the conversion stage.
  }];

  let arguments = (ins
    AnyType:$data,
    AnyType:$indices,
    AnyType:$indptr,
    AnyType:$rhs
  );

  let results = (outs
    AnyType:$solution
  );

  let assemblyFormat = [{
    $data `,` $indices `,` $indptr `,` $rhs attr-dict
    `:` `(` type($data) `,` type($indices) `,` type($indptr) `,` type($rhs) `)`
    `->` type($solution)
  }];
}

//===----------------------------------------------------------------------===//
// sparse_solver.spsolve_complete (HAL-level, takes buffer_views)
//===----------------------------------------------------------------------===//

def SparseSolver_SpsolveCompleteOp : SparseSolver_Op<"spsolve_complete", []> {
  let summary = "Complete sparse LU solve with HAL buffer views";
  let description = [{
    HAL-level sparse solve operation that takes buffer_views directly.
    This is the interface to the BaSpaCho sparse solver runtime module.

    Performs: analyze + factor_lu + solve_lu + release in a single call.

    Arguments:
    - n: Matrix dimension (number of rows/columns)
    - nnz: Number of non-zero elements
    - row_ptr: CSR row pointers buffer view (n+1 elements, i32/i64)
    - col_idx: CSR column indices buffer view (nnz elements, i32/i64)
    - values: CSR non-zero values buffer view (nnz elements, f32/f64)
    - rhs: Right-hand side vector buffer view (n elements, f32/f64)
    - solution: Output solution vector buffer view (n elements, f32/f64)

    This op is lowered to vm.call @sparse_solver.spsolve_complete.
  }];

  let arguments = (ins
    I64:$n,
    I64:$nnz,
    HAL_BufferView:$row_ptr,
    HAL_BufferView:$col_idx,
    HAL_BufferView:$values,
    HAL_BufferView:$rhs,
    HAL_BufferView:$solution
  );

  let results = (outs);

  let assemblyFormat = [{
    `n` `(` $n `)` `nnz` `(` $nnz `)`
    `row_ptr` `(` $row_ptr `:` type($row_ptr) `)`
    `col_idx` `(` $col_idx `:` type($col_idx) `)`
    `values` `(` $values `:` type($values) `)`
    `rhs` `(` $rhs `:` type($rhs) `)`
    `solution` `(` $solution `:` type($solution) `)`
    attr-dict
  }];
}

def SparseSolver_SpsolveCompleteF64Op : SparseSolver_Op<"spsolve_complete.f64", []> {
  let summary = "Complete sparse LU solve with HAL buffer views (f64)";
  let description = [{
    Same as spsolve_complete but for double precision (f64) matrices.
    Lowered to vm.call @sparse_solver.spsolve_complete.f64.
  }];

  let arguments = (ins
    I64:$n,
    I64:$nnz,
    HAL_BufferView:$row_ptr,
    HAL_BufferView:$col_idx,
    HAL_BufferView:$values,
    HAL_BufferView:$rhs,
    HAL_BufferView:$solution
  );

  let results = (outs);

  let assemblyFormat = [{
    `n` `(` $n `)` `nnz` `(` $nnz `)`
    `row_ptr` `(` $row_ptr `:` type($row_ptr) `)`
    `col_idx` `(` $col_idx `:` type($col_idx) `)`
    `values` `(` $values `:` type($values) `)`
    `rhs` `(` $rhs `:` type($rhs) `)`
    `solution` `(` $solution `:` type($solution) `)`
    attr-dict
  }];
}

#endif // IREE_DIALECT_SPARSESOLVER_IR_SPARSESOLVEROPS_TD
