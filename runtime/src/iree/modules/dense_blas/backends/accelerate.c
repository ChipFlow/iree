// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/dense_blas/backends/accelerate.h"

#if defined(__APPLE__) && defined(IREE_DENSE_BLAS_HAVE_ACCELERATE)

#include <Accelerate/Accelerate.h>

//===----------------------------------------------------------------------===//
// GEMM via cblas_sgemm/cblas_dgemm
//===----------------------------------------------------------------------===//

iree_status_t iree_dense_blas_accelerate_gemm_f32(
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    float beta,
    float* C, int64_t ldc) {
  cblas_sgemm(
      CblasColMajor,
      trans_a ? CblasTrans : CblasNoTrans,
      trans_b ? CblasTrans : CblasNoTrans,
      (int)M, (int)N, (int)K,
      alpha,
      A, (int)lda,
      B, (int)ldb,
      beta,
      C, (int)ldc);
  return iree_ok_status();
}

iree_status_t iree_dense_blas_accelerate_gemm_f64(
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    double alpha,
    const double* A, int64_t lda,
    const double* B, int64_t ldb,
    double beta,
    double* C, int64_t ldc) {
  cblas_dgemm(
      CblasColMajor,
      trans_a ? CblasTrans : CblasNoTrans,
      trans_b ? CblasTrans : CblasNoTrans,
      (int)M, (int)N, (int)K,
      alpha,
      A, (int)lda,
      B, (int)ldb,
      beta,
      C, (int)ldc);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// SYRK via cblas_ssyrk/cblas_dsyrk
//===----------------------------------------------------------------------===//

iree_status_t iree_dense_blas_accelerate_syrk_f32(
    bool upper, bool trans,
    int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t lda,
    float beta,
    float* C, int64_t ldc) {
  cblas_ssyrk(
      CblasColMajor,
      upper ? CblasUpper : CblasLower,
      trans ? CblasTrans : CblasNoTrans,
      (int)N, (int)K,
      alpha,
      A, (int)lda,
      beta,
      C, (int)ldc);
  return iree_ok_status();
}

iree_status_t iree_dense_blas_accelerate_syrk_f64(
    bool upper, bool trans,
    int64_t N, int64_t K,
    double alpha,
    const double* A, int64_t lda,
    double beta,
    double* C, int64_t ldc) {
  cblas_dsyrk(
      CblasColMajor,
      upper ? CblasUpper : CblasLower,
      trans ? CblasTrans : CblasNoTrans,
      (int)N, (int)K,
      alpha,
      A, (int)lda,
      beta,
      C, (int)ldc);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// TRSM via cblas_strsm/cblas_dtrsm
//===----------------------------------------------------------------------===//

iree_status_t iree_dense_blas_accelerate_trsm_f32(
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    float alpha,
    const float* A, int64_t lda,
    float* B, int64_t ldb) {
  cblas_strsm(
      CblasColMajor,
      left_side ? CblasLeft : CblasRight,
      upper ? CblasUpper : CblasLower,
      trans_a ? CblasTrans : CblasNoTrans,
      unit_diag ? CblasUnit : CblasNonUnit,
      (int)M, (int)N,
      alpha,
      A, (int)lda,
      B, (int)ldb);
  return iree_ok_status();
}

iree_status_t iree_dense_blas_accelerate_trsm_f64(
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    double alpha,
    const double* A, int64_t lda,
    double* B, int64_t ldb) {
  cblas_dtrsm(
      CblasColMajor,
      left_side ? CblasLeft : CblasRight,
      upper ? CblasUpper : CblasLower,
      trans_a ? CblasTrans : CblasNoTrans,
      unit_diag ? CblasUnit : CblasNonUnit,
      (int)M, (int)N,
      alpha,
      A, (int)lda,
      B, (int)ldb);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// POTRF via LAPACK spotrf_/dpotrf_
//===----------------------------------------------------------------------===//

// LAPACK function declarations (Accelerate provides these).
extern int spotrf_(char* uplo, int* n, float* a, int* lda, int* info);
extern int dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);

int iree_dense_blas_accelerate_potrf_f32(
    bool upper,
    int64_t N,
    float* A, int64_t lda) {
  char uplo = upper ? 'U' : 'L';
  int n = (int)N;
  int lda_int = (int)lda;
  int info = 0;
  spotrf_(&uplo, &n, A, &lda_int, &info);
  return info;
}

int iree_dense_blas_accelerate_potrf_f64(
    bool upper,
    int64_t N,
    double* A, int64_t lda) {
  char uplo = upper ? 'U' : 'L';
  int n = (int)N;
  int lda_int = (int)lda;
  int info = 0;
  dpotrf_(&uplo, &n, A, &lda_int, &info);
  return info;
}

#else  // !__APPLE__ || !IREE_DENSE_BLAS_HAVE_ACCELERATE

// Stub implementations for non-Apple platforms.

iree_status_t iree_dense_blas_accelerate_gemm_f32(
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    float beta,
    float* C, int64_t ldc) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Accelerate backend not available on this platform");
}

iree_status_t iree_dense_blas_accelerate_gemm_f64(
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    double alpha,
    const double* A, int64_t lda,
    const double* B, int64_t ldb,
    double beta,
    double* C, int64_t ldc) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Accelerate backend not available on this platform");
}

iree_status_t iree_dense_blas_accelerate_syrk_f32(
    bool upper, bool trans,
    int64_t N, int64_t K,
    float alpha,
    const float* A, int64_t lda,
    float beta,
    float* C, int64_t ldc) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Accelerate backend not available on this platform");
}

iree_status_t iree_dense_blas_accelerate_syrk_f64(
    bool upper, bool trans,
    int64_t N, int64_t K,
    double alpha,
    const double* A, int64_t lda,
    double beta,
    double* C, int64_t ldc) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Accelerate backend not available on this platform");
}

iree_status_t iree_dense_blas_accelerate_trsm_f32(
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    float alpha,
    const float* A, int64_t lda,
    float* B, int64_t ldb) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Accelerate backend not available on this platform");
}

iree_status_t iree_dense_blas_accelerate_trsm_f64(
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    double alpha,
    const double* A, int64_t lda,
    double* B, int64_t ldb) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Accelerate backend not available on this platform");
}

int iree_dense_blas_accelerate_potrf_f32(
    bool upper,
    int64_t N,
    float* A, int64_t lda) {
  return -1;  // Unavailable
}

int iree_dense_blas_accelerate_potrf_f64(
    bool upper,
    int64_t N,
    double* A, int64_t lda) {
  return -1;  // Unavailable
}

#endif  // __APPLE__ && IREE_DENSE_BLAS_HAVE_ACCELERATE
