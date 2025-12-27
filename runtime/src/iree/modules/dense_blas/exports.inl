// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Dense BLAS Module Exports
//===----------------------------------------------------------------------===//
// Function signature type codes:
//   r = ref (vm.ref)
//   i = i32
//   I = i64
//   f = f32
//   F = f64
//   v = void

// GEMM: C = alpha * op(A) * op(B) + beta * C
// Args: device, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
// Type: ref, i32, i32, i64, i64, i64, f32, ref, i64, ref, i64, f32, ref, i64
EXPORT_FN("gemm", iree_dense_blas_gemm, riiIIIfIrIfIrI, v)

// GEMM with f64 precision
EXPORT_FN("gemm.f64", iree_dense_blas_gemm_f64, riiIIIFIrIFIrI, v)

// SYRK: C = alpha * A * A^T + beta * C (symmetric rank-k update)
// Args: device, uplo, trans, N, K, alpha, A, lda, beta, C, ldc
// Type: ref, i32, i32, i64, i64, f32, ref, i64, f32, ref, i64
EXPORT_FN("syrk", iree_dense_blas_syrk, riiIIfIrfIrI, v)

// SYRK with f64 precision
EXPORT_FN("syrk.f64", iree_dense_blas_syrk_f64, riiIIFIrFIrI, v)

// TRSM: Solve op(A) * X = alpha * B or X * op(A) = alpha * B
// Args: device, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb
// Type: ref, i32, i32, i32, i32, i64, i64, f32, ref, i64, ref, i64
EXPORT_FN("trsm", iree_dense_blas_trsm, riiiiIIfIrIrI, v)

// TRSM with f64 precision
EXPORT_FN("trsm.f64", iree_dense_blas_trsm_f64, riiiiIIFIrIrI, v)

// POTRF: Cholesky factorization of symmetric positive-definite matrix
// Args: device, uplo, N, A, lda
// Returns: info (0 = success, >0 = not positive definite)
// Type: ref, i32, i64, ref, i64 -> i32
EXPORT_FN("potrf", iree_dense_blas_potrf, riIrI, i)

// POTRF with f64 precision
EXPORT_FN("potrf.f64", iree_dense_blas_potrf_f64, riIrI, i)
