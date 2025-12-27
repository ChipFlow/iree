// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/dense_blas/gemm_metal.h"
#include "iree/modules/dense_blas/hal_buffer_helper.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Include only the CBLAS header to avoid simd type conflicts.
// The full Accelerate.h includes vecLib which has clang/SDK compatibility issues
// with the current macOS SDK and clang version.
extern "C" {
#include <vecLib/cblas.h>
}

#include "iree/hal/drivers/metal/metal_buffer.h"

//===----------------------------------------------------------------------===//
// Accelerate BLAS GEMM Implementation
//===----------------------------------------------------------------------===//
// Uses Apple's Accelerate framework (cblas_sgemm) which leverages the AMX
// coprocessor on Apple Silicon for excellent BLAS performance.
//
// TODO: Add MPS (Metal Performance Shaders) support for GPU-accelerated GEMM
// once the clang/SDK simd header compatibility issues are resolved.

iree_status_t iree_dense_blas_gemm_metal(
    iree_hal_device_t* device,
    iree_hal_buffer_view_t* lhs,   // A: [M, K]
    iree_hal_buffer_view_t* rhs,   // B: [K, N]
    iree_hal_buffer_view_t* out,   // C: [M, N]
    float alpha,
    float beta,
    bool transpose_lhs,
    bool transpose_rhs,
    iree_allocator_t host_allocator) {

  // Validate inputs.
  if (!device || !lhs || !rhs || !out) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "null argument to gemm_metal");
  }

  // Get dimensions from buffer views.
  iree_host_size_t lhs_rank = iree_hal_buffer_view_shape_rank(lhs);
  iree_host_size_t rhs_rank = iree_hal_buffer_view_shape_rank(rhs);
  iree_host_size_t out_rank = iree_hal_buffer_view_shape_rank(out);

  if (lhs_rank != 2 || rhs_rank != 2 || out_rank != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GEMM requires 2D matrices");
  }

  // Get dimensions: A[M,K] * B[K,N] = C[M,N]
  iree_hal_dim_t M = iree_hal_buffer_view_shape_dim(lhs, transpose_lhs ? 1 : 0);
  iree_hal_dim_t K_lhs = iree_hal_buffer_view_shape_dim(lhs, transpose_lhs ? 0 : 1);
  iree_hal_dim_t K_rhs = iree_hal_buffer_view_shape_dim(rhs, transpose_rhs ? 1 : 0);
  iree_hal_dim_t N = iree_hal_buffer_view_shape_dim(rhs, transpose_rhs ? 0 : 1);

  if (K_lhs != K_rhs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GEMM dimension mismatch: K_lhs=%" PRId64 " != K_rhs=%" PRId64,
                            (int64_t)K_lhs, (int64_t)K_rhs);
  }
  iree_hal_dim_t K = K_lhs;

  // Verify output dimensions.
  iree_hal_dim_t out_M = iree_hal_buffer_view_shape_dim(out, 0);
  iree_hal_dim_t out_N = iree_hal_buffer_view_shape_dim(out, 1);
  if (out_M != M || out_N != N) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GEMM output dimension mismatch");
  }

  // Check element type.
  iree_hal_element_type_t element_type = iree_hal_buffer_view_element_type(lhs);
  if (element_type != IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "GEMM only supports f32 on Metal");
  }

  // Get underlying Metal buffers.
  iree_hal_buffer_t* lhs_buffer = iree_hal_buffer_view_buffer(lhs);
  iree_hal_buffer_t* rhs_buffer = iree_hal_buffer_view_buffer(rhs);
  iree_hal_buffer_t* out_buffer = iree_hal_buffer_view_buffer(out);

  id<MTLBuffer> mtl_lhs = iree_hal_metal_buffer_handle(
      iree_hal_buffer_allocated_buffer(lhs_buffer) ?: lhs_buffer);
  id<MTLBuffer> mtl_rhs = iree_hal_metal_buffer_handle(
      iree_hal_buffer_allocated_buffer(rhs_buffer) ?: rhs_buffer);
  id<MTLBuffer> mtl_out = iree_hal_metal_buffer_handle(
      iree_hal_buffer_allocated_buffer(out_buffer) ?: out_buffer);

  if (!mtl_lhs || !mtl_rhs || !mtl_out) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to get Metal buffers for GEMM");
  }

  // Get buffer offsets.
  NSUInteger lhs_offset = iree_hal_buffer_byte_offset(lhs_buffer);
  NSUInteger rhs_offset = iree_hal_buffer_byte_offset(rhs_buffer);
  NSUInteger out_offset = iree_hal_buffer_byte_offset(out_buffer);

  // Get pointers to buffer contents.
  // For shared storage mode buffers, contents returns a valid CPU pointer.
  // For private storage mode, we would need to copy (not implemented yet).
  const float* A = (const float*)((uint8_t*)mtl_lhs.contents + lhs_offset);
  const float* B = (const float*)((uint8_t*)mtl_rhs.contents + rhs_offset);
  float* C = (float*)((uint8_t*)mtl_out.contents + out_offset);

  if (!A || !B || !C) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "Metal buffers not accessible from CPU. "
                            "Private storage mode not yet supported.");
  }

  // Use Accelerate framework's cblas_sgemm.
  // cblas_sgemm performs: C = alpha * op(A) * op(B) + beta * C
  //
  // Parameters for row-major storage:
  // - CblasRowMajor: Row-major layout
  // - CblasNoTrans/CblasTrans: Whether to transpose A/B
  // - M: Number of rows in op(A) and C
  // - N: Number of columns in op(B) and C
  // - K: Number of columns in op(A) and rows in op(B)
  // - lda: Leading dimension of A (number of columns in A's storage)
  // - ldb: Leading dimension of B (number of columns in B's storage)
  // - ldc: Leading dimension of C (number of columns in C's storage)

  CBLAS_TRANSPOSE transA = transpose_lhs ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = transpose_rhs ? CblasTrans : CblasNoTrans;

  // Leading dimensions based on original storage layout.
  // For a matrix stored as [rows, cols] in row-major, ld = cols.
  int lda = transpose_lhs ? (int)M : (int)K;  // A is [M,K] or [K,M] if transposed
  int ldb = transpose_rhs ? (int)K : (int)N;  // B is [K,N] or [N,K] if transposed
  int ldc = (int)N;  // C is always [M,N]

  cblas_sgemm(CblasRowMajor, transA, transB,
              (int)M, (int)N, (int)K,
              alpha,
              A, lda,
              B, ldb,
              beta,
              C, ldc);

  return iree_ok_status();
}
