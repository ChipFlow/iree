// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Sparse Solver Module Exports
//===----------------------------------------------------------------------===//
// Function signature type codes:
//   r = ref (vm.ref)
//   i = i32
//   I = i64
//   f = f32
//   F = f64
//   v = void

//===----------------------------------------------------------------------===//
// Symbolic Analysis
//===----------------------------------------------------------------------===//

// Analyze sparsity pattern (done once per pattern).
// Args: device, n (matrix dimension), nnz (number of non-zeros),
//       row_ptr (CSR row pointers), col_idx (CSR column indices)
// Returns: analysis handle (ref)
// Type: ref, i64, i64, ref, ref -> ref
EXPORT_FN("analyze", iree_sparse_solver_analyze, rIIrr, r)

// Release an analysis handle.
// Args: analysis_handle
// Type: ref -> void
EXPORT_FN("release", iree_sparse_solver_release, r, v)

//===----------------------------------------------------------------------===//
// Numeric Factorization
//===----------------------------------------------------------------------===//

// Perform Cholesky factorization with f32 values.
// Args: analysis_handle, values buffer (CSR values)
// Returns: status (0 = success, >0 = not positive definite at row info)
// Type: ref, ref -> i32
EXPORT_FN("factor", iree_sparse_solver_factor, rr, i)

// Perform Cholesky factorization with f64 values.
// Args: analysis_handle, values buffer
// Returns: status
// Type: ref, ref -> i32
EXPORT_FN("factor.f64", iree_sparse_solver_factor_f64, rr, i)

//===----------------------------------------------------------------------===//
// Solve Operations
//===----------------------------------------------------------------------===//

// Solve A*x = b using the factored matrix (f32).
// Args: analysis_handle, rhs buffer, solution buffer
// Type: ref, ref, ref -> void
EXPORT_FN("solve", iree_sparse_solver_solve, rrr, v)

// Solve A*x = b using the factored matrix (f64).
// Args: analysis_handle, rhs buffer, solution buffer
// Type: ref, ref, ref -> void
EXPORT_FN("solve.f64", iree_sparse_solver_solve_f64, rrr, v)

// Batched solve: solve A*X = B for multiple right-hand sides.
// Args: analysis_handle, rhs buffer, solution buffer, num_rhs
// Type: ref, ref, ref, i64 -> void
EXPORT_FN("solve.batched", iree_sparse_solver_solve_batched, rrrI, v)

// Batched solve with f64 precision.
// Args: analysis_handle, rhs buffer, solution buffer, num_rhs
// Type: ref, ref, ref, i64 -> void
EXPORT_FN("solve.batched.f64", iree_sparse_solver_solve_batched_f64, rrrI, v)

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

// Get the number of non-zeros in the factored matrix.
// Useful for memory allocation planning.
// Args: analysis_handle
// Returns: nnz in L factor
// Type: ref -> i64
EXPORT_FN("get_factor_nnz", iree_sparse_solver_get_factor_nnz, r, I)

// Get the number of supernodes in the factored matrix.
// Args: analysis_handle
// Returns: number of supernodes
// Type: ref -> i64
EXPORT_FN("get_num_supernodes", iree_sparse_solver_get_num_supernodes, r, I)
