// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_DENSE_BLAS_HAL_BUFFER_HELPER_H_
#define IREE_MODULES_DENSE_BLAS_HAL_BUFFER_HELPER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// HAL Buffer Helper for Zero-Copy GPU Integration
//===----------------------------------------------------------------------===//
// Provides unified access to native device pointers across backends.
// This enables external BLAS/solver libraries to operate directly on
// IREE's GPU buffers without data copies.
//
// Supported backends:
//   - CUDA: CUdeviceptr via iree_hal_cuda_buffer_device_pointer()
//   - HIP: hipDeviceptr_t via iree_hal_hip_buffer_device_pointer()
//   - Metal: id<MTLBuffer> via iree_hal_metal_buffer_handle()
//   - OpenCL: Host pointer mapped, cl_mem lookup via BaSpaCho's
//             OpenCLBufferRegistry (findBuffer maps ptr -> cl_mem)
//   - Vulkan: via export to DEVICE_ALLOCATION

// Backend type for buffer operations.
typedef enum iree_hal_buffer_backend_e {
  IREE_HAL_BUFFER_BACKEND_UNKNOWN = 0,
  IREE_HAL_BUFFER_BACKEND_CPU = 1,
  IREE_HAL_BUFFER_BACKEND_CUDA = 2,
  IREE_HAL_BUFFER_BACKEND_HIP = 3,
  IREE_HAL_BUFFER_BACKEND_METAL = 4,
  IREE_HAL_BUFFER_BACKEND_VULKAN = 5,
  IREE_HAL_BUFFER_BACKEND_OPENCL = 6,
} iree_hal_buffer_backend_t;

// Device pointer union for different backends.
typedef union iree_device_ptr_u {
  void* cpu_ptr;
  uint64_t cuda_ptr;  // CUdeviceptr
  uint64_t hip_ptr;   // hipDeviceptr_t
  void* metal_buffer; // id<MTLBuffer>
  void* vulkan_memory; // VkDeviceMemory
  void* opencl_mem;   // cl_mem
} iree_device_ptr_t;

// Result of getting a device pointer from a HAL buffer.
typedef struct iree_hal_device_ptr_info_t {
  iree_hal_buffer_backend_t backend;
  iree_device_ptr_t ptr;
  iree_device_size_t offset;  // Byte offset within the allocation
  iree_device_size_t size;    // Size in bytes
  bool is_host_visible;       // Can be mapped to host memory
} iree_hal_device_ptr_info_t;

// Detect the backend type from a HAL device.
iree_hal_buffer_backend_t iree_hal_detect_buffer_backend(
    iree_hal_device_t* device);

// Get the native device pointer from a HAL buffer.
// This is the primary API for zero-copy integration.
//
// For GPU backends (CUDA, HIP, Metal), returns the native device pointer.
// For CPU backend, returns the host pointer.
// For Vulkan/OpenCL, attempts to export the buffer.
//
// Parameters:
//   buffer - The HAL buffer to get the pointer from.
//   device - The HAL device (used for backend detection).
//   out_info - Receives the device pointer information.
//
// Returns:
//   iree_ok_status() on success.
//   IREE_STATUS_UNAVAILABLE if the buffer cannot provide a device pointer.
IREE_API_EXPORT iree_status_t iree_hal_get_device_ptr(
    iree_hal_buffer_t* buffer,
    iree_hal_device_t* device,
    iree_hal_device_ptr_info_t* out_info);

// Get the device pointer for a buffer view (handles offset/size).
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_get_device_ptr(
    iree_hal_buffer_view_t* buffer_view,
    iree_hal_device_t* device,
    iree_hal_device_ptr_info_t* out_info);

//===----------------------------------------------------------------------===//
// Command Queue/Stream Helpers
//===----------------------------------------------------------------------===//
// For async operations, external libraries need to use the same command
// queue/stream as IREE to avoid synchronization issues.

// Command queue union for different backends.
typedef union iree_command_queue_u {
  void* cuda_stream;    // CUstream
  void* hip_stream;     // hipStream_t
  void* metal_queue;    // id<MTLCommandQueue>
  void* metal_buffer;   // id<MTLCommandBuffer> for encoding
  void* opencl_queue;   // cl_command_queue
} iree_command_queue_t;

// Get the command queue/stream from a HAL device for async operations.
// External libraries should use this queue to avoid synchronization issues.
IREE_API_EXPORT iree_status_t iree_hal_get_command_queue(
    iree_hal_device_t* device,
    iree_command_queue_t* out_queue);

// Create a command buffer for encoding Metal operations.
// The returned command buffer can be passed to MPS or BaSpaCho.
// Caller must commit the command buffer after encoding operations.
IREE_API_EXPORT iree_status_t iree_hal_metal_create_command_buffer(
    iree_hal_device_t* device,
    void** out_command_buffer);  // Returns id<MTLCommandBuffer>

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_DENSE_BLAS_HAL_BUFFER_HELPER_H_
