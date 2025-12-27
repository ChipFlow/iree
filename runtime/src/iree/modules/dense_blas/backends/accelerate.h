// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_DENSE_BLAS_BACKENDS_ACCELERATE_H_
#define IREE_MODULES_DENSE_BLAS_BACKENDS_ACCELERATE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Apple Accelerate Framework Backend (CPU)
//===----------------------------------------------------------------------===//
// Uses Apple's Accelerate framework for CPU-based dense BLAS via CBLAS.
// This is the CPU fallback for Apple platforms when GPU execution is not
// available or not beneficial (small matrices).
//
// Note: Accelerate provides CPU-only acceleration through:
//   - NEON SIMD on Apple Silicon
//   - AMX (Apple Matrix coprocessor) instructions
//
// All standard BLAS operations are supported including:
//   - gemm, syrk, trsm (BLAS Level 3)
//   - potrf (LAPACK)

// GEMM: C = alpha * op(A) * op(B) + beta * C
// Wraps cblas_sgemm.
iree_status_t iree_dense_blas_accelerate_gemm_f32(
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    float beta,
    float* C, int64_t ldc);

// GEMM with f64 precision.
// Wraps cblas_dgemm.
iree_status_t iree_dense_blas_accelerate_gemm_f64(
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    double alpha,
    const double* A, int64_t lda,
    const double* B, int64_t ldb,
    double beta,
    double* C, int64_t ldc);

// SYRK: C = alpha * A * A^T + beta * C
// Wraps cblas_ssyrk.
iree_status_t iree_dense_blas_accelerate_syrk_f32(
    bool upper, bool trans,
    int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t lda,
    float beta,
    float* C, int64_t ldc);

// SYRK with f64 precision.
// Wraps cblas_dsyrk.
iree_status_t iree_dense_blas_accelerate_syrk_f64(
    bool upper, bool trans,
    int64_t N, int64_t K,
    double alpha,
    const double* A, int64_t lda,
    double beta,
    double* C, int64_t ldc);

// TRSM: Solve op(A) * X = alpha * B or X * op(A) = alpha * B
// Wraps cblas_strsm.
iree_status_t iree_dense_blas_accelerate_trsm_f32(
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    float alpha,
    const float* A, int64_t lda,
    float* B, int64_t ldb);

// TRSM with f64 precision.
// Wraps cblas_dtrsm.
iree_status_t iree_dense_blas_accelerate_trsm_f64(
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    double alpha,
    const double* A, int64_t lda,
    double* B, int64_t ldb);

// POTRF: Cholesky factorization of symmetric positive-definite matrix.
// Wraps LAPACK spotrf/dpotrf.
// Returns: 0 on success, >0 if matrix is not positive definite.
int iree_dense_blas_accelerate_potrf_f32(
    bool upper,
    int64_t N,
    float* A, int64_t lda);

int iree_dense_blas_accelerate_potrf_f64(
    bool upper,
    int64_t N,
    double* A, int64_t lda);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_DENSE_BLAS_BACKENDS_ACCELERATE_H_
