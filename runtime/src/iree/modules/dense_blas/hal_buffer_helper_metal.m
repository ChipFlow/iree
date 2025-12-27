// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/dense_blas/hal_buffer_helper.h"

#import <Metal/Metal.h>
#include "iree/hal/drivers/metal/metal_buffer.h"

//===----------------------------------------------------------------------===//
// Metal Buffer Access Implementation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_metal_get_device_ptr(
    iree_hal_buffer_t* buffer,
    iree_hal_device_ptr_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_info);

  // Get the underlying allocated buffer (handles sub-spans).
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  if (!allocated) {
    allocated = buffer;
  }

  // Get the Metal buffer handle.
  id<MTLBuffer> metal_buffer = iree_hal_metal_buffer_handle(allocated);
  if (!metal_buffer) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "buffer is not a Metal buffer");
  }

  out_info->backend = IREE_HAL_BUFFER_BACKEND_METAL;
  out_info->ptr.metal_buffer = (__bridge void*)metal_buffer;
  out_info->offset = iree_hal_buffer_byte_offset(buffer);
  out_info->size = iree_hal_buffer_byte_length(buffer);

  // Check if the buffer is host-visible (shared or managed storage mode).
  MTLResourceOptions options = metal_buffer.resourceOptions;
  MTLStorageMode storageMode = (options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift;
  out_info->is_host_visible = (storageMode == MTLStorageModeShared ||
                               storageMode == MTLStorageModeManaged);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Metal Command Queue Access
//===----------------------------------------------------------------------===//

// Forward declaration - implemented in metal device.
// We need to access the command queue from the device.
// For now, return unimplemented and require the caller to manage sync.

iree_status_t iree_hal_metal_get_command_queue_impl(
    iree_hal_device_t* device,
    id<MTLCommandQueue>* out_queue) {
  // TODO: Expose command queue from iree_hal_metal_device.
  // The Metal HAL device has a command queue but doesn't expose it publicly.
  // For now, return unimplemented - caller should create their own queue
  // and use fences for synchronization.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Metal command queue access not yet exposed");
}

iree_status_t iree_hal_metal_create_command_buffer_impl(
    iree_hal_device_t* device,
    id<MTLCommandBuffer>* out_command_buffer) {
  // TODO: Create command buffer from device's queue.
  // This requires exposing the command queue first.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Metal command buffer creation not yet implemented");
}
