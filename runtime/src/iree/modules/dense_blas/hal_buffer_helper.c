// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/dense_blas/hal_buffer_helper.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Backend-Specific Implementations (defined in separate files)
//===----------------------------------------------------------------------===//

#if IREE_DENSE_BLAS_HAVE_METAL
// Defined in hal_buffer_helper_metal.m
extern iree_status_t iree_hal_metal_get_device_ptr(
    iree_hal_buffer_t* buffer,
    iree_hal_device_ptr_info_t* out_info);
#endif

//===----------------------------------------------------------------------===//
// Backend Detection
//===----------------------------------------------------------------------===//

iree_hal_buffer_backend_t iree_hal_detect_buffer_backend(
    iree_hal_device_t* device) {
  if (!device) return IREE_HAL_BUFFER_BACKEND_UNKNOWN;

  iree_string_view_t device_id = iree_hal_device_id(device);

  if (iree_string_view_starts_with(device_id, IREE_SV("cuda"))) {
    return IREE_HAL_BUFFER_BACKEND_CUDA;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("hip")) ||
             iree_string_view_starts_with(device_id, IREE_SV("rocm"))) {
    return IREE_HAL_BUFFER_BACKEND_HIP;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("metal"))) {
    return IREE_HAL_BUFFER_BACKEND_METAL;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("vulkan"))) {
    return IREE_HAL_BUFFER_BACKEND_VULKAN;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("opencl"))) {
    return IREE_HAL_BUFFER_BACKEND_OPENCL;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("local")) ||
             iree_string_view_starts_with(device_id, IREE_SV("vmvx"))) {
    return IREE_HAL_BUFFER_BACKEND_CPU;
  }

  return IREE_HAL_BUFFER_BACKEND_UNKNOWN;
}

//===----------------------------------------------------------------------===//
// Device Pointer Access - Stub implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_get_cpu_device_ptr(
    iree_hal_buffer_t* buffer,
    iree_hal_device_ptr_info_t* out_info) {
  // For CPU backend, map the buffer to get host pointer.
  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      0, IREE_HAL_WHOLE_BUFFER, &mapping);

  if (!iree_status_is_ok(status)) {
    return status;
  }

  out_info->backend = IREE_HAL_BUFFER_BACKEND_CPU;
  out_info->ptr.cpu_ptr = mapping.contents.data;
  out_info->offset = 0;  // Mapping already accounts for offset.
  out_info->size = mapping.contents.data_length;
  out_info->is_host_visible = true;

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_get_device_ptr(
    iree_hal_buffer_t* buffer,
    iree_hal_device_t* device,
    iree_hal_device_ptr_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_info);

  memset(out_info, 0, sizeof(*out_info));

  // Get the underlying allocated buffer (handles sub-spans).
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  if (!allocated) {
    allocated = buffer;
  }

  iree_hal_buffer_backend_t backend = iree_hal_detect_buffer_backend(device);

  switch (backend) {
#if IREE_DENSE_BLAS_HAVE_METAL
    case IREE_HAL_BUFFER_BACKEND_METAL:
      return iree_hal_metal_get_device_ptr(buffer, out_info);
#endif

    case IREE_HAL_BUFFER_BACKEND_CUDA:
    case IREE_HAL_BUFFER_BACKEND_HIP:
#if !IREE_DENSE_BLAS_HAVE_METAL
    case IREE_HAL_BUFFER_BACKEND_METAL:
#endif
    case IREE_HAL_BUFFER_BACKEND_VULKAN:
    case IREE_HAL_BUFFER_BACKEND_OPENCL:
      // TODO: Add backend-specific implementations.
      // For now, fall through to CPU mapping.
    case IREE_HAL_BUFFER_BACKEND_CPU:
    default:
      return iree_hal_get_cpu_device_ptr(allocated, out_info);
  }
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_get_device_ptr(
    iree_hal_buffer_view_t* buffer_view,
    iree_hal_device_t* device,
    iree_hal_device_ptr_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(buffer_view);

  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  return iree_hal_get_device_ptr(buffer, device, out_info);
}

//===----------------------------------------------------------------------===//
// Command Queue/Stream Access - Stub implementation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_get_command_queue(
    iree_hal_device_t* device,
    iree_command_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_queue);

  memset(out_queue, 0, sizeof(*out_queue));

  // Stub implementation - backend-specific implementations will be added later.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_metal_create_command_buffer(
    iree_hal_device_t* device,
    void** out_command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Metal command buffer creation not yet implemented");
}
