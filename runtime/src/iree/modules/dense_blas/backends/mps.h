// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_DENSE_BLAS_BACKENDS_MPS_H_
#define IREE_MODULES_DENSE_BLAS_BACKENDS_MPS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Metal Performance Shaders (MPS) Backend
//===----------------------------------------------------------------------===//
// Uses Apple's Metal Performance Shaders for GPU-accelerated dense BLAS.
// MPS provides highly optimized matrix operations for Apple GPUs.
//
// Supported operations:
//   - gemm: MPSMatrixMultiplication
//   - syrk: (via gemm + triangular extraction)
//   - trsm: MPSMatrixSolveTriangular
//
// Unsupported (falls back to Accelerate):
//   - potrf: Cholesky factorization (MPS doesn't have this)

// MPS context handle (opaque pointer to Objective-C objects).
typedef struct iree_dense_blas_mps_context_t* iree_dense_blas_mps_context_t;

// Creates an MPS context for the given Metal device.
// |out_context| must be destroyed with iree_dense_blas_mps_context_destroy.
iree_status_t iree_dense_blas_mps_context_create(
    void* metal_device,  // id<MTLDevice>
    iree_allocator_t allocator,
    iree_dense_blas_mps_context_t* out_context);

// Destroys an MPS context and releases associated resources.
void iree_dense_blas_mps_context_destroy(
    iree_dense_blas_mps_context_t context);

// GEMM: C = alpha * op(A) * op(B) + beta * C
// Uses MPSMatrixMultiplication.
// All buffers must be MTLBuffer objects.
iree_status_t iree_dense_blas_mps_gemm_f32(
    iree_dense_blas_mps_context_t context,
    void* command_buffer,  // id<MTLCommandBuffer>
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    void* A,  // id<MTLBuffer>
    int64_t lda,
    void* B,  // id<MTLBuffer>
    int64_t ldb,
    float beta,
    void* C,  // id<MTLBuffer>
    int64_t ldc);

// TRSM: Solve op(A) * X = alpha * B
// Uses MPSMatrixSolveTriangular.
iree_status_t iree_dense_blas_mps_trsm_f32(
    iree_dense_blas_mps_context_t context,
    void* command_buffer,  // id<MTLCommandBuffer>
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    float alpha,
    void* A,  // id<MTLBuffer>
    int64_t lda,
    void* B,  // id<MTLBuffer>
    int64_t ldb);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_DENSE_BLAS_BACKENDS_MPS_H_
