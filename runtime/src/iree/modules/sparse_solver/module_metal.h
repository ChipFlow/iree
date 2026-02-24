// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_SPARSE_SOLVER_MODULE_METAL_H_
#define IREE_MODULES_SPARSE_SOLVER_MODULE_METAL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Check if this is a Metal-based unified memory device where direct
// buffer access is possible.
bool iree_sparse_solver_metal_is_unified_memory(iree_hal_device_t* device);

// Get the Metal device handle from an IREE HAL device.
// Returns NULL if not a Metal device.
void* iree_sparse_solver_metal_get_device(iree_hal_device_t* device);

// Get the contents pointer from a Metal buffer.
// This works on unified memory systems where CPU can access GPU buffers directly.
// Returns NULL if the buffer is not accessible.
void* iree_sparse_solver_metal_buffer_contents(iree_hal_buffer_t* buffer);

// Get the underlying Metal buffer handle from an IREE buffer.
// Returns NULL if not a Metal buffer.
void* iree_sparse_solver_metal_buffer_handle(iree_hal_buffer_t* buffer);

// Synchronize Metal operations (wait for GPU to finish).
void iree_sparse_solver_metal_synchronize(iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_MODULES_SPARSE_SOLVER_MODULE_METAL_H_
