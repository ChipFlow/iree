// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MPS-based dense solve operations for Metal.
// Separated from module_metal.m because MetalPerformanceShaders headers
// conflict with IREE's -fno-lax-vector-conversions compiler flag.

#ifndef IREE_MODULES_SPARSE_SOLVER_MODULE_METAL_MPS_H_
#define IREE_MODULES_SPARSE_SOLVER_MODULE_METAL_MPS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encodes a dense NxN linear solve (Ax = b) into a Metal command buffer using
// MPS (MetalPerformanceShaders). All operations are recorded as GPU commands
// without reading buffer contents, making this compatible with IREE's
// streamable recording pipeline.
//
// Pipeline: MPSMatrixDecompositionLU → pivot permutation → forward solve → back solve
//
// |mtl_cmd_buf|: MTLCommandBuffer (as void*) to encode operations into.
// |mtl_device|:  MTLDevice (as void*) for resource allocation.
// |mtl_matrix|:  MTLBuffer (as void*) containing the NxN matrix (row-major f32).
//                Modified in-place with LU factors.
// |matrix_byte_off|: Byte offset into mtl_matrix.
// |mtl_rhs|:     MTLBuffer (as void*) containing the N-element rhs vector (f32).
// |rhs_byte_off|: Byte offset into mtl_rhs.
// |mtl_solution|: MTLBuffer (as void*) to receive the N-element solution (f32).
// |sol_byte_off|: Byte offset into mtl_solution.
// |n|:           Matrix dimension.
iree_status_t iree_sparse_solver_metal_mps_dense_solve(
    void* mtl_cmd_buf, void* mtl_device,
    void* mtl_matrix, uint64_t matrix_byte_off,
    void* mtl_rhs, uint64_t rhs_byte_off,
    void* mtl_solution, uint64_t sol_byte_off,
    int64_t n);

#ifdef __cplusplus
}
#endif

#endif  // IREE_MODULES_SPARSE_SOLVER_MODULE_METAL_MPS_H_
