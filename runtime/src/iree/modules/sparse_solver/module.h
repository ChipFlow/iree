// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_SPARSE_SOLVER_MODULE_H_
#define IREE_MODULES_SPARSE_SOLVER_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Sparse Solver Module
//===----------------------------------------------------------------------===//
// Provides sparse Cholesky factorization and solve operations via BaSpaCho.
//
// BaSpaCho (Batched Sparse Cholesky) is optimized for:
// - Levenberg-Marquardt algorithms in nonlinear optimization
// - Bundle adjustment, SLAM, and similar problems
// - Batched solves with the same sparsity pattern
//
// Backend selection (via NumericCtx/SolveCtx abstraction):
// - Tier 1: Native Metal (Apple), CUDA (NVIDIA)
// - Tier 2: OpenCL (generic fallback)
//
// Workflow:
// 1. analyze() - Symbolic analysis of sparsity pattern (done once)
// 2. factor() - Numeric factorization (done when values change)
// 3. solve() - Forward/backward substitution (can be batched)

// Flags controlling module behavior.
typedef uint32_t iree_sparse_solver_module_flags_t;
enum iree_sparse_solver_module_flags_bits_t {
  IREE_SPARSE_SOLVER_MODULE_FLAG_NONE = 0u,
  // Use supernodal factorization (better for larger supernodes).
  IREE_SPARSE_SOLVER_MODULE_FLAG_SUPERNODAL = 1u << 0,
  // Use left-looking factorization (lower memory).
  IREE_SPARSE_SOLVER_MODULE_FLAG_LEFT_LOOKING = 1u << 1,
};

// Creates a sparse solver module that can be used with the given |device|.
// The device determines which BaSpaCho backend to use:
//   - Metal device -> Metal compute shaders
//   - CUDA device -> CUDA kernels + cuBLAS
//   - Other -> OpenCL/CLBlast fallback
//
// |out_module| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_sparse_solver_module_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_sparse_solver_module_flags_t flags, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_SPARSE_SOLVER_MODULE_H_
