// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/sparse_solver/module_metal.h"

#import <Metal/Metal.h>

#include "iree/hal/drivers/metal/direct_command_buffer.h"
#include "iree/hal/drivers/metal/metal_buffer.h"
#include "iree/hal/drivers/metal/metal_device.h"
#include "iree/modules/sparse_solver/baspacho_wrapper.h"

bool iree_sparse_solver_metal_is_unified_memory(iree_hal_device_t* device) {
  if (!device) return false;

  // Check if device ID starts with "metal"
  iree_string_view_t device_id = iree_hal_device_id(device);
  if (!iree_string_view_starts_with(device_id, IREE_SV("metal"))) {
    return false;
  }

  // Get the Metal device and check for unified memory
  id<MTLDevice> mtl_device = iree_hal_metal_device_handle(device);
  if (!mtl_device) return false;

  return [mtl_device hasUnifiedMemory];
}

void* iree_sparse_solver_metal_get_device(iree_hal_device_t* device) {
  if (!device) return NULL;

  // Check if device ID starts with "metal"
  iree_string_view_t device_id = iree_hal_device_id(device);
  if (!iree_string_view_starts_with(device_id, IREE_SV("metal"))) {
    return NULL;
  }

  return (__bridge void*)iree_hal_metal_device_handle(device);
}

void* iree_sparse_solver_metal_buffer_contents(iree_hal_buffer_t* buffer) {
  if (!buffer) return NULL;

  // Get the underlying allocated buffer (handles subspans/views).
  iree_hal_buffer_t* allocated_buffer = iree_hal_buffer_allocated_buffer(buffer);
  if (!allocated_buffer) return NULL;

  id<MTLBuffer> mtl_buffer = iree_hal_metal_buffer_handle(allocated_buffer);
  if (!mtl_buffer) return NULL;

  // Get byte offset for subspans.
  iree_device_size_t byte_offset = iree_hal_buffer_byte_offset(buffer);

  // On unified memory, [MTLBuffer contents] gives direct CPU-accessible pointer.
  // Add byte offset for subspans.
  uint8_t* base_ptr = (uint8_t*)[mtl_buffer contents];
  return base_ptr + byte_offset;
}

void* iree_sparse_solver_metal_buffer_handle(iree_hal_buffer_t* buffer) {
  if (!buffer) return NULL;
  // Get the underlying allocated buffer (handles subspans/views).
  iree_hal_buffer_t* allocated_buffer = iree_hal_buffer_allocated_buffer(buffer);
  if (!allocated_buffer) return NULL;
  return (__bridge void*)iree_hal_metal_buffer_handle(allocated_buffer);
}

void iree_sparse_solver_metal_synchronize(iree_hal_device_t* device) {
  // Metal operations on unified memory don't need explicit synchronization
  // for CPU access - the [MTLBuffer contents] pointer is always coherent.
  // However, if we wanted to wait for pending GPU work, we would need
  // to commit and wait on command buffers.
  (void)device;
}

void iree_sparse_solver_metal_register_iree_buffer(iree_hal_buffer_t* buffer) {
  if (!buffer) return;
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  if (!allocated) return;
  id<MTLBuffer> mtl_buffer = iree_hal_metal_buffer_handle(allocated);
  if (!mtl_buffer) return;
  void* base = [mtl_buffer contents];
  size_t size = [mtl_buffer length];
  baspacho_register_metal_buffer(base, (__bridge void*)mtl_buffer, size);
}

void iree_sparse_solver_metal_unregister_iree_buffer(iree_hal_buffer_t* buffer) {
  if (!buffer) return;
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  if (!allocated) return;
  id<MTLBuffer> mtl_buffer = iree_hal_metal_buffer_handle(allocated);
  if (!mtl_buffer) return;
  void* base = [mtl_buffer contents];
  baspacho_unregister_metal_buffer(base);
}

iree_status_t iree_sparse_solver_metal_spsolve_gpu(
    iree_hal_command_buffer_t* cmd_buf,
    iree_hal_buffer_t* data_buf, int64_t data_off, int64_t data_len,
    iree_hal_buffer_t* indices_buf, int64_t indices_off, int64_t indices_len,
    iree_hal_buffer_t* indptr_buf, int64_t indptr_off, int64_t indptr_len,
    iree_hal_buffer_t* rhs_buf, int64_t rhs_off, int64_t rhs_len,
    iree_hal_buffer_t* solution_buf, int64_t solution_off,
    int64_t solution_len, iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  baspacho_handle_t baspacho = NULL;
  int64_t* row_ptr_i64 = NULL;
  int64_t* col_idx_i64 = NULL;
  int64_t* pivots = NULL;

  // Get buffer content pointers (unified memory, offset-adjusted).
  float* data_ptr = (float*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                                 data_buf) +
                             data_off);
  int32_t* indices_ptr =
      (int32_t*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                     indices_buf) +
                 indices_off);
  int32_t* indptr_ptr =
      (int32_t*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                     indptr_buf) +
                 indptr_off);
  float* rhs_ptr = (float*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                                rhs_buf) +
                            rhs_off);
  float* solution_ptr =
      (float*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                   solution_buf) +
               solution_off);

  if (!data_ptr || !indices_ptr || !indptr_ptr || !rhs_ptr || !solution_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to get Metal buffer contents for spsolve");
  }

  // Infer dimensions from buffer lengths.
  int64_t nnz = data_len / (int64_t)sizeof(float);
  int64_t n_plus_1 = indptr_len / (int64_t)sizeof(int32_t);
  int64_t n = n_plus_1 - 1;

  // Convert int32 indices to int64 for BaSpaCho.
  status = iree_allocator_malloc(host_allocator,
                                 (n + 1) * sizeof(int64_t),
                                 (void**)&row_ptr_i64);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_allocator_malloc(host_allocator,
                                 nnz * sizeof(int64_t),
                                 (void**)&col_idx_i64);
  if (!iree_status_is_ok(status)) goto cleanup;

  for (int64_t i = 0; i <= n; ++i) row_ptr_i64[i] = indptr_ptr[i];
  for (int64_t i = 0; i < nnz; ++i) col_idx_i64[i] = indices_ptr[i];

  // Create BaSpaCho context with Metal backend.
  baspacho = baspacho_create(BASPACHO_BACKEND_METAL);
  if (!baspacho) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "failed to create BaSpaCho Metal context");
    goto cleanup;
  }

  // Symbolic analysis (CPU work, reads sparsity pattern from unified memory).
  {
    int result = baspacho_analyze(baspacho, n, nnz, row_ptr_i64, col_idx_i64);
    if (result != 0) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "BaSpaCho symbolic analysis failed: %d",
                                result);
      goto cleanup;
    }
  }

  // Register IREE buffers with BaSpaCho's buffer registry for zero-copy.
  iree_sparse_solver_metal_register_iree_buffer(data_buf);
  iree_sparse_solver_metal_register_iree_buffer(rhs_buf);
  iree_sparse_solver_metal_register_iree_buffer(solution_buf);

  // End IREE's current compute encoder before creating ours.
  iree_hal_metal_direct_command_buffer_end_compute_encoder(cmd_buf);

  // Get the underlying MTLCommandBuffer and create encoder for BaSpaCho.
  {
    id<MTLCommandBuffer> mtl_cmd_buf =
        iree_hal_metal_direct_command_buffer_handle(cmd_buf);
    id<MTLComputeCommandEncoder> encoder =
        [mtl_cmd_buf computeCommandEncoder];

    // Set BaSpaCho to use this encoder for all dispatches.
    baspacho_set_external_metal_encoder(
        baspacho, (__bridge void*)mtl_cmd_buf, (__bridge void*)encoder);

    // Allocate pivots for LU factorization.
    status = iree_allocator_malloc(host_allocator,
                                   n * sizeof(int64_t),
                                   (void**)&pivots);
    if (!iree_status_is_ok(status)) {
      [encoder endEncoding];
      baspacho_clear_external_encoder(baspacho);
      goto cleanup;
    }

    // LU factorization — records dispatches into the encoder.
    int result = baspacho_factor_lu_f32_device(baspacho, data_ptr, pivots);
    if (result != 0) {
      [encoder endEncoding];
      baspacho_clear_external_encoder(baspacho);
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "BaSpaCho LU factorization failed: %d",
                                result);
      goto cleanup;
    }

    // Solve — records dispatches into the encoder.
    baspacho_solve_lu_f32_device(baspacho, pivots, rhs_ptr, solution_ptr);

    // End the encoder. IREE will lazily create a new one for subsequent
    // dispatches via iree_hal_metal_get_or_begin_compute_encoder.
    [encoder endEncoding];
    baspacho_clear_external_encoder(baspacho);
  }

cleanup:
  // Unregister buffers.
  iree_sparse_solver_metal_unregister_iree_buffer(data_buf);
  iree_sparse_solver_metal_unregister_iree_buffer(rhs_buf);
  iree_sparse_solver_metal_unregister_iree_buffer(solution_buf);

  if (pivots) iree_allocator_free(host_allocator, pivots);
  if (col_idx_i64) iree_allocator_free(host_allocator, col_idx_i64);
  if (row_ptr_i64) iree_allocator_free(host_allocator, row_ptr_i64);
  if (baspacho) baspacho_destroy(baspacho);
  return status;
}

iree_status_t iree_sparse_solver_metal_dense_solve_gpu(
    iree_hal_command_buffer_t* cmd_buf,
    iree_hal_buffer_t* matrix_buf, int64_t matrix_off, int64_t matrix_len,
    iree_hal_buffer_t* rhs_buf, int64_t rhs_off, int64_t rhs_len,
    iree_hal_buffer_t* solution_buf, int64_t solution_off,
    int64_t solution_len, iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  baspacho_handle_t baspacho = NULL;
  int64_t* pivots = NULL;

  // Get buffer content pointers (unified memory, offset-adjusted).
  float* matrix_ptr =
      (float*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                   matrix_buf) +
               matrix_off);
  float* rhs_ptr = (float*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                                rhs_buf) +
                            rhs_off);
  float* solution_ptr =
      (float*)((uint8_t*)iree_sparse_solver_metal_buffer_contents(
                   solution_buf) +
               solution_off);

  if (!matrix_ptr || !rhs_ptr || !solution_ptr) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "failed to get Metal buffer contents for dense solve");
  }

  // Dense solve via BaSpaCho GPU LU on Metal.
  // Uses a fully-dense CSR pattern so BaSpaCho processes it as a single
  // supernode — MPS MPSMatrixDecompositionLU for factorization, GPU kernels
  // for forward/backward substitution. All dispatches are recorded into
  // IREE's command buffer via the external encoder API.
  int64_t n = rhs_len / (int64_t)sizeof(float);

  // Create BaSpaCho context with Metal backend.
  baspacho = baspacho_create(BASPACHO_BACKEND_METAL);
  if (!baspacho) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to create BaSpaCho Metal context");
  }

  // Dense symbolic analysis (CPU work, builds fully-dense CSR pattern).
  {
    int result = baspacho_dense_analyze(baspacho, n);
    if (result != 0) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "BaSpaCho dense analysis failed: %d", result);
      goto cleanup;
    }
  }

  // Register IREE buffers with BaSpaCho's buffer registry for zero-copy.
  iree_sparse_solver_metal_register_iree_buffer(matrix_buf);
  iree_sparse_solver_metal_register_iree_buffer(rhs_buf);
  iree_sparse_solver_metal_register_iree_buffer(solution_buf);

  // End IREE's current compute encoder before creating ours.
  iree_hal_metal_direct_command_buffer_end_compute_encoder(cmd_buf);

  // Get the underlying MTLCommandBuffer and create encoder for BaSpaCho.
  {
    id<MTLCommandBuffer> mtl_cmd_buf =
        iree_hal_metal_direct_command_buffer_handle(cmd_buf);
    id<MTLComputeCommandEncoder> encoder =
        [mtl_cmd_buf computeCommandEncoder];

    // Set BaSpaCho to use this encoder for all dispatches.
    baspacho_set_external_metal_encoder(
        baspacho, (__bridge void*)mtl_cmd_buf, (__bridge void*)encoder);

    // Allocate pivots for LU factorization.
    status = iree_allocator_malloc(host_allocator,
                                   n * sizeof(int64_t),
                                   (void**)&pivots);
    if (!iree_status_is_ok(status)) {
      [encoder endEncoding];
      baspacho_clear_external_encoder(baspacho);
      goto cleanup_buffers;
    }

    // LU factorization — records dispatches into the encoder.
    // BaSpaCho's dense path uses MPS MPSMatrixDecompositionLU for the
    // single NxN block, which records into the provided encoder.
    int result = baspacho_factor_lu_f32_device(baspacho, matrix_ptr, pivots);
    if (result != 0) {
      [encoder endEncoding];
      baspacho_clear_external_encoder(baspacho);
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "BaSpaCho dense LU factorization failed: %d",
                                result);
      goto cleanup_buffers;
    }

    // Solve — records dispatches into the encoder.
    baspacho_solve_lu_f32_device(baspacho, pivots, rhs_ptr, solution_ptr);

    // End the encoder. IREE will lazily create a new one for subsequent
    // dispatches via iree_hal_metal_get_or_begin_compute_encoder.
    [encoder endEncoding];
    baspacho_clear_external_encoder(baspacho);
  }

cleanup_buffers:
  // Unregister buffers.
  iree_sparse_solver_metal_unregister_iree_buffer(matrix_buf);
  iree_sparse_solver_metal_unregister_iree_buffer(rhs_buf);
  iree_sparse_solver_metal_unregister_iree_buffer(solution_buf);

cleanup:
  if (pivots) iree_allocator_free(host_allocator, pivots);
  if (baspacho) baspacho_destroy(baspacho);
  return status;
}
