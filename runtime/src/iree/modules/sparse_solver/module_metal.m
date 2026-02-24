// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/sparse_solver/module_metal.h"

#import <Metal/Metal.h>

#include "iree/hal/drivers/metal/metal_device.h"
#include "iree/hal/drivers/metal/metal_buffer.h"

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
