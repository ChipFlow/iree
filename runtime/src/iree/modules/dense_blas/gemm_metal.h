// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_DENSE_BLAS_GEMM_METAL_H_
#define IREE_MODULES_DENSE_BLAS_GEMM_METAL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Performs GEMM using Metal Performance Shaders.
// C = alpha * op(A) * op(B) + beta * C
//
// Parameters:
//   device: HAL device (must be Metal)
//   lhs: Left-hand side matrix A [M, K] or [K, M] if transposed
//   rhs: Right-hand side matrix B [K, N] or [N, K] if transposed
//   out: Output matrix C [M, N] - will be overwritten
//   alpha: Scalar multiplier for A*B
//   beta: Scalar multiplier for C (0 means C is not read)
//   transpose_lhs: If true, use A^T
//   transpose_rhs: If true, use B^T
//   host_allocator: Allocator for temporary allocations
//
// Currently only supports f32 matrices.
iree_status_t iree_dense_blas_gemm_metal(
    iree_hal_device_t* device,
    iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs,
    iree_hal_buffer_view_t* out,
    float alpha,
    float beta,
    bool transpose_lhs,
    bool transpose_rhs,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_DENSE_BLAS_GEMM_METAL_H_
