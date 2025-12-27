// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_DENSE_BLAS_MODULE_H_
#define IREE_MODULES_DENSE_BLAS_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Dense BLAS Module
//===----------------------------------------------------------------------===//
// Provides GPU-accelerated dense linear algebra operations via external
// libraries (cuBLAS, rocBLAS, MPS, CLBlast) with zero-copy HAL buffer sharing.
//
// Tier 1 (Optimized): cuBLAS (CUDA), rocBLAS (ROCm), MPS (Metal)
// Tier 2 (Generic): CLBlast (OpenCL fallback)
// CPU Fallback: Accelerate (Apple), OpenBLAS (others)

// Flags controlling module behavior.
typedef uint32_t iree_dense_blas_module_flags_t;
enum iree_dense_blas_module_flags_bits_t {
  IREE_DENSE_BLAS_MODULE_FLAG_NONE = 0u,
  // Prefer GPU backends even for small matrices.
  IREE_DENSE_BLAS_MODULE_FLAG_PREFER_GPU = 1u << 0,
  // Enable auto-tuning for CLBlast backend.
  IREE_DENSE_BLAS_MODULE_FLAG_CLBLAST_TUNE = 1u << 1,
};

// Creates a dense BLAS module that can be used with the given |device|.
// The device determines which backend to use:
//   - CUDA device -> cuBLAS
//   - HIP device -> rocBLAS
//   - Metal device -> MPS
//   - Vulkan/OpenCL device -> CLBlast
//   - CPU device -> Accelerate/OpenBLAS
//
// |out_module| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_dense_blas_module_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_dense_blas_module_flags_t flags, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_DENSE_BLAS_MODULE_H_
