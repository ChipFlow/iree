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

// Register an IREE Metal buffer with BaSpaCho's buffer registry.
// Uses the allocated (root) buffer to get the MTLBuffer base pointer.
void iree_sparse_solver_metal_register_iree_buffer(iree_hal_buffer_t* buffer);

// Unregister an IREE Metal buffer from BaSpaCho's buffer registry.
void iree_sparse_solver_metal_unregister_iree_buffer(iree_hal_buffer_t* buffer);

// Streamable sparse solve: records BaSpaCho dispatches into IREE's
// command buffer encoder. This enables the solve to live inside a
// stream.cmd.execute region alongside other GPU dispatches.
//
// The function:
// 1. Ends IREE's current compute encoder
// 2. Creates a new encoder for BaSpaCho
// 3. Records LU factor + solve into the encoder
// 4. Ends the encoder (IREE lazily creates a new one for subsequent dispatches)
iree_status_t iree_sparse_solver_metal_spsolve_gpu(
    iree_hal_command_buffer_t* cmd_buf,
    iree_hal_buffer_t* data_buf, int64_t data_off, int64_t data_len,
    iree_hal_buffer_t* indices_buf, int64_t indices_off, int64_t indices_len,
    iree_hal_buffer_t* indptr_buf, int64_t indptr_off, int64_t indptr_len,
    iree_hal_buffer_t* rhs_buf, int64_t rhs_off, int64_t rhs_len,
    iree_hal_buffer_t* solution_buf, int64_t solution_off,
    int64_t solution_len, iree_allocator_t host_allocator);

// Streamable dense solve: records BaSpaCho dispatches into IREE's
// command buffer encoder. Uses BaSpaCho's dense LU path (fully on GPU).
iree_status_t iree_sparse_solver_metal_dense_solve_gpu(
    iree_hal_command_buffer_t* cmd_buf,
    iree_hal_buffer_t* matrix_buf, int64_t matrix_off, int64_t matrix_len,
    iree_hal_buffer_t* rhs_buf, int64_t rhs_off, int64_t rhs_len,
    iree_hal_buffer_t* solution_buf, int64_t solution_off,
    int64_t solution_len, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_MODULES_SPARSE_SOLVER_MODULE_METAL_H_
