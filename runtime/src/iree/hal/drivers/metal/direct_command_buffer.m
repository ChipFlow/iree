// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/direct_command_buffer.h"

#import <Metal/Metal.h>
#import <Metal/MTL4CommandBuffer.h>
#import <Metal/MTL4ComputeCommandEncoder.h>
#import <Metal/MTL4CommandAllocator.h>
#import <Metal/MTL4CommandQueue.h>
#import <Metal/MTL4ArgumentTable.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/builtin_executables.h"
#include "iree/hal/drivers/metal/executable.h"
#include "iree/hal/drivers/metal/metal_buffer.h"
#include "iree/hal/drivers/metal/metal_device.h"
#include "iree/hal/drivers/metal/staging_buffer.h"
#include "iree/hal/utils/resource_set.h"

//===------------------------------------------------------------------------------------------===//
// Segmented submission management
//===------------------------------------------------------------------------------------------===//

// This file implements IREE HAL command buffers using the Metal 4 API.
//
// Metal 4 uses a unified command encoder (MTL4ComputeCommandEncoder) that handles compute
// dispatches, blit operations, and acceleration structures in a single encoder — no encoder
// switching needed. Command buffers are created from the device, encoders use argument tables
// instead of setBuffer/setBytes, and resource residency is managed explicitly via MTLResidencySet.
//
// We still use a two-phase approach with a linked list of command segments. First we create
// segments (iree_hal_metal_command_buffer_prepare_* and iree_hal_metal_command_segment_create_*)
// to keep track of all IREE HAL commands and their data, and then, when finalizing the command
// buffer, we iterate through all segments and record them (iree_hal_metal_command_segment_record_*)
// into a proper Metal 4 command buffer via the unified encoder.

// Command action kind of a command segment.
typedef enum iree_hal_metal_command_segment_action_e {
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_BARRIER,      // Execution/memory barrier command
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_DISPATCH,     // Dispatch command
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_FILL_BUFFER,  // Fill buffer command
  IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_COPY_BUFFER,  // Copy buffer command
} iree_hal_metal_command_segment_action_t;

// API data for execution/memory barrier command segments.
typedef struct iree_hal_metal_barrier_segment_t {
  iree_host_size_t memory_barrier_count;  // Total number of memory barriers
  iree_host_size_t buffer_barrier_count;  // Total number of buffer barriers
  // The list of buffer barriers, pointing to the end of the segment allocation.
  const iree_hal_buffer_barrier_t* buffer_barriers;
} iree_hal_metal_barrier_segment_t;
// + Additional inline allocation for holding all buffer barriers.

typedef struct iree_hal_metal_descriptor_t {
  uint32_t binding;
  iree_hal_buffer_t* buffer;
  iree_device_size_t offset;
  MTLResourceUsage usage;
} iree_hal_metal_descriptor_t;

// API data for dispatch command segments.
typedef struct iree_hal_metal_dispatch_segment_t {
  // Compute function pipeline with information required to dispatch it.
  const iree_hal_metal_pipeline_t* pipeline;

  // Threadgroup size required during dispatch.
  MTLSize threadgroup_size;

  // Workgroup count information--if |workgroups_buffer| is not nil, then indirect dispatch;
  // otherwise uses |workgroup_count| for direct dispatch.
  id<MTLBuffer> workgroups_buffer;
  iree_device_size_t workgroups_offset;
  MTLSize workgroup_count;

  // The number of descriptors bound for this dispatch.
  iree_host_size_t descriptor_count;
  // The list of bound descriptors, pointing to the end of the segment allocation.
  iree_hal_metal_descriptor_t* descriptors;

  // The number of push constant values.
  iree_host_size_t constant_count;
  // The list of push constants, pointing to the end of the segment allocation.
  int32_t* constants;
} iree_hal_metal_dispatch_segment_t;
// + Additional inline allocation for holding all bound descriptors.
// + Additional inline allocation for holding all push constants.

// API data for fill buffer command segments.
typedef struct iree_hal_metal_fill_buffer_segment_t {
  id<MTLBuffer> target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  // The fill pattern, pointing to the end of the segment allocation.
  const void* pattern;
  iree_host_size_t pattern_length;
} iree_hal_metal_fill_buffer_segment_t;
// + Additional inline allocation for holding the fill pattern.

// API data for copy buffer command segments.
typedef struct iree_hal_metal_copy_buffer_segment_t {
  id<MTLBuffer> source_buffer;
  iree_device_size_t source_offset;
  id<MTLBuffer> target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
} iree_hal_metal_copy_buffer_segment_t;

struct iree_hal_metal_command_segment_t;
typedef struct iree_hal_metal_command_segment_t {
  struct iree_hal_metal_command_segment_t* next_segment;
  iree_hal_metal_command_segment_action_t action;
  union {
    iree_hal_metal_barrier_segment_t barrier;
    iree_hal_metal_dispatch_segment_t dispatch;
    iree_hal_metal_fill_buffer_segment_t fill_buffer;
    iree_hal_metal_copy_buffer_segment_t copy_buffer;
  };
} iree_hal_metal_command_segment_t;

typedef struct iree_hal_metal_command_segment_list_t {
  iree_hal_metal_command_segment_t* head;
  iree_hal_metal_command_segment_t* tail;
} iree_hal_metal_command_segment_list_t;

static void iree_hal_metal_command_segment_list_reset(iree_hal_metal_command_segment_list_t* list) {
  memset(list, 0, sizeof(*list));
}

static void iree_hal_metal_command_segment_list_push_front(
    iree_hal_metal_command_segment_list_t* list, iree_hal_metal_command_segment_t* segment) {
  segment->next_segment = list->head;
  list->head = segment;
  if (!list->tail) list->tail = segment;
}

static void iree_hal_metal_command_segment_list_push_back(
    iree_hal_metal_command_segment_list_t* list, iree_hal_metal_command_segment_t* segment) {
  segment->next_segment = NULL;
  if (list->tail) {
    list->tail->next_segment = segment;
    list->tail = segment;
  } else {
    list->head = list->tail = segment;
  }
}

//===------------------------------------------------------------------------------------------===//
// iree_hal_metal_command_buffer_t
//===------------------------------------------------------------------------------------------===//

typedef struct iree_hal_metal_command_buffer_t {
  iree_hal_command_buffer_t base;

  // The HAL device owning this command buffer. We need to retain it to make sure it outlive this
  // command buffer to allow access to shared resources.
  iree_hal_device_t* device;

  // The Metal 4 command queue owning this command buffer.
  id<MTL4CommandQueue> queue;

  // For polyfilling fill/copy/update buffers that are not directly supported by Metal APIs.
  iree_hal_metal_builtin_executable_t* builtin_executable;

  // Arena used for all allocations; references the shared device block pool.
  iree_arena_allocator_t arena;

  // Per-queue shared uniform staging buffer for uploading parameters to the GPU, including argument
  // buffers and buffer update source buffers.
  iree_hal_metal_staging_buffer_t* staging_buffer;

  // Per-command-buffer dedicated buffer for indirect bindings (PhysicalStorageBuffer).
  // This buffer is owned by this command buffer and not shared, preventing race conditions
  // where the GPU reads stale data while another command buffer overwrites shared staging buffer.
  id<MTLBuffer> indirect_bindings_buffer;
  uint32_t indirect_bindings_capacity;  // Current allocated capacity in bytes
  uint32_t indirect_bindings_offset;    // Current write offset in bytes

  iree_allocator_t host_allocator;

  // Maintains a reference to all resources used within the command buffer. Resets on each begin.
  iree_hal_resource_set_t* resource_set;

  // Linked list of command segments to be recorded into a command buffer.
  iree_hal_metal_command_segment_list_t segments;

  // The Metal 4 command buffer for recording commands.
  id<MTL4CommandBuffer> command_buffer;

  // The command allocator used for command buffer recording.
  id<MTL4CommandAllocator> command_allocator;

  // The unified compute command encoder. MTL4 uses a single encoder for compute dispatches,
  // blit operations (copy/fill), and barriers — no encoder switching needed.
  id<MTL4ComputeCommandEncoder> encoder;

  // Argument table for binding buffer addresses to compute kernels.
  // Created once and reused across dispatches (bindings are updated per-dispatch).
  id<MTL4ArgumentTable> argument_table;

  // Residency set for tracking which allocations must be resident during execution.
  // All buffers used by dispatches are added here instead of per-encoder useResource: calls.
  id<MTLResidencySet> residency_set;
} iree_hal_metal_command_buffer_t;

//===------------------------------------------------------------------------------------------===//
// iree_hal_metal_command_buffer_vtable APIs
//===------------------------------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t iree_hal_metal_command_buffer_vtable;

static iree_hal_metal_command_buffer_t* iree_hal_metal_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_command_buffer_vtable);
  return (iree_hal_metal_command_buffer_t*)base_value;
}

static const iree_hal_metal_command_buffer_t* iree_hal_metal_command_buffer_const_cast(
    const iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_command_buffer_vtable);
  return (const iree_hal_metal_command_buffer_t*)base_value;
}

id<MTL4CommandBuffer> iree_hal_metal_direct_command_buffer_handle(
    const iree_hal_command_buffer_t* base_command_buffer) {
  const iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_const_cast(base_command_buffer);
  return command_buffer->command_buffer;
}

static void iree_hal_metal_end_encoder(iree_hal_metal_command_buffer_t* command_buffer) {
  if (command_buffer->encoder) {
    [command_buffer->encoder endEncoding];
    [command_buffer->encoder release];  // -1
    command_buffer->encoder = nil;
  }
}

void iree_hal_metal_direct_command_buffer_end_encoder(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  iree_hal_metal_end_encoder(command_buffer);
}

static void iree_hal_metal_command_buffer_reset(iree_hal_metal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_metal_end_encoder(command_buffer);
  iree_hal_metal_command_segment_list_reset(&command_buffer->segments);
  iree_arena_reset(&command_buffer->arena);
  // Release the indirect bindings buffer if present. We allocate a fresh one
  // for each recording to avoid race conditions where GPU is still reading
  // from the old buffer while we reset. This is necessary because command
  // buffers may be reused before GPU execution completes.
  if (command_buffer->indirect_bindings_buffer != nil) {
    [command_buffer->indirect_bindings_buffer release];  // -1
    command_buffer->indirect_bindings_buffer = nil;
    command_buffer->indirect_bindings_capacity = 0;
  }
  command_buffer->indirect_bindings_offset = 0;
  IREE_TRACE_ZONE_END(z0);
}

static id<MTL4ComputeCommandEncoder> iree_hal_metal_get_or_begin_encoder(
    iree_hal_metal_command_buffer_t* command_buffer) {
  if (!command_buffer->encoder) {
    @autoreleasepool {
      // MTL4 uses a unified encoder that handles compute dispatches, blit operations,
      // and barriers. No dispatch type parameter — concurrency is managed via barriers.
      command_buffer->encoder =
          [[(id<MTL4CommandBuffer>)command_buffer->command_buffer computeCommandEncoder] retain];  // +1
    }
  }
  return command_buffer->encoder;
}

// Default initial capacity for the indirect bindings buffer (4KB should be plenty for most cases).
#define IREE_HAL_METAL_INDIRECT_BINDINGS_BUFFER_DEFAULT_CAPACITY (4 * 1024)

// Reserves space in the per-command-buffer indirect bindings buffer.
// Allocates or grows the buffer as needed. Returns the host pointer and buffer offset.
static iree_status_t iree_hal_metal_indirect_bindings_buffer_reserve(
    iree_hal_metal_command_buffer_t* command_buffer, iree_host_size_t length,
    iree_host_size_t alignment, uint8_t** out_host_ptr, uint32_t* out_offset) {
  // Calculate aligned offset.
  uint32_t aligned_offset = iree_host_align(command_buffer->indirect_bindings_offset, alignment);
  uint32_t required_capacity = aligned_offset + (uint32_t)length;

  // Allocate or grow the buffer if needed.
  if (required_capacity > command_buffer->indirect_bindings_capacity) {
    // Determine new capacity (start with default, then double as needed).
    uint32_t new_capacity = command_buffer->indirect_bindings_capacity == 0
                                ? IREE_HAL_METAL_INDIRECT_BINDINGS_BUFFER_DEFAULT_CAPACITY
                                : command_buffer->indirect_bindings_capacity * 2;
    while (new_capacity < required_capacity) {
      new_capacity *= 2;
    }

    // Allocate new buffer with shared storage mode and default cache mode.
    // This matches staging_buffer.m settings for CPU-GPU coherency.
    MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
    id<MTLBuffer> new_buffer = [command_buffer->queue.device newBufferWithLength:new_capacity
                                                                         options:options];  // +1
    if (!new_buffer) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to allocate indirect bindings buffer with size = %u bytes",
                              new_capacity);
    }

    // Copy existing data if growing an existing buffer.
    if (command_buffer->indirect_bindings_buffer != nil) {
      memcpy(new_buffer.contents, command_buffer->indirect_bindings_buffer.contents,
             command_buffer->indirect_bindings_offset);
      [command_buffer->indirect_bindings_buffer release];  // -1
    }

    command_buffer->indirect_bindings_buffer = new_buffer;
    command_buffer->indirect_bindings_capacity = new_capacity;
  }

  // Update offset and return reservation.
  command_buffer->indirect_bindings_offset = aligned_offset + (uint32_t)length;
  *out_host_ptr = (uint8_t*)command_buffer->indirect_bindings_buffer.contents + aligned_offset;
  *out_offset = aligned_offset;

  return iree_ok_status();
}

// Destroys the given |base_command_buffer| itself, without decreasing refcount in the shared
// staging buffer yet.
static void iree_hal_metal_command_buffer_destroy_internal(
    iree_hal_command_buffer_t* base_command_buffer);

iree_status_t iree_hal_metal_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_host_size_t binding_capacity,
    iree_hal_metal_command_buffer_resource_reference_mode_t resource_reference_mode,
    id<MTL4CommandQueue> queue, iree_arena_block_pool_t* block_pool,
    iree_hal_metal_staging_buffer_t* staging_buffer,
    iree_hal_metal_builtin_executable_t* builtin_executable, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_ASSERT_TRUE(iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT));
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "indirect command buffer not yet supported");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) + iree_hal_command_buffer_validation_state_size(
                                                          mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(iree_hal_device_allocator(device), mode, command_categories,
                                     IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
                                     (uint8_t*)command_buffer + sizeof(*command_buffer),
                                     &iree_hal_metal_command_buffer_vtable, &command_buffer->base);
  command_buffer->device = device;
  command_buffer->queue = [queue retain];  // +1
  command_buffer->builtin_executable = builtin_executable;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  command_buffer->staging_buffer = staging_buffer;
  command_buffer->host_allocator = host_allocator;
  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_metal_command_segment_list_reset(&command_buffer->segments);
    @autoreleasepool {
      id<MTLDevice> device_handle = queue.device;

      // Create a command allocator for this command buffer.
      NSError* alloc_error = nil;
      MTL4CommandAllocatorDescriptor* alloc_desc = [MTL4CommandAllocatorDescriptor new];  // +1
      command_buffer->command_allocator =
          [device_handle newCommandAllocatorWithDescriptor:alloc_desc error:&alloc_error];  // +1
      [alloc_desc release];                                              // -1
      if (!command_buffer->command_allocator) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "failed to create command allocator: %s",
                                  alloc_error.localizedDescription.UTF8String);
      }

      // Create a Metal 4 command buffer from the device.
      if (iree_status_is_ok(status)) {
        command_buffer->command_buffer = [device_handle newCommandBuffer];  // +1
      }

      // Create the argument table for binding buffer addresses to kernels.
      if (iree_status_is_ok(status)) {
        NSError* arg_error = nil;
        MTL4ArgumentTableDescriptor* arg_desc = [MTL4ArgumentTableDescriptor new];  // +1
        arg_desc.maxBufferBindCount = 31;  // Max supported by Metal 4
        command_buffer->argument_table =
            [device_handle newArgumentTableWithDescriptor:arg_desc error:&arg_error];  // +1
        [arg_desc release];                                            // -1
        if (!command_buffer->argument_table) {
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "failed to create argument table: %s",
                                    arg_error.localizedDescription.UTF8String);
        }
      }

      // Create a residency set for tracking buffer residency.
      MTLResidencySetDescriptor* res_desc = [MTLResidencySetDescriptor new];  // +1
      NSError* error = nil;
      command_buffer->residency_set =
          [device_handle newResidencySetWithDescriptor:res_desc error:&error];  // +1
      [res_desc release];                                                       // -1
      if (!command_buffer->residency_set) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "failed to create residency set: %s",
                                  error.localizedDescription.UTF8String);
      }
    }
    command_buffer->encoder = nil;
    // Initialize per-command-buffer indirect bindings buffer (allocated lazily on first use).
    command_buffer->indirect_bindings_buffer = nil;
    command_buffer->indirect_bindings_capacity = 0;
    command_buffer->indirect_bindings_offset = 0;
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;

    // Increase command buffer refcount in the shared staging buffer. We tie this to the command
    // buffer's lifetime to avoid resource leak.
    iree_hal_metal_staging_buffer_increase_command_buffer_refcount(staging_buffer);
    // Retain the device given that we refer to builtin executables and staging buffers whose
    // lifetime is associated with the device.
    iree_hal_device_retain(device);
  } else {
    iree_hal_metal_command_buffer_destroy_internal(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_command_buffer_destroy_internal(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);

  iree_hal_metal_command_buffer_reset(command_buffer);
  IREE_ASSERT_EQ(command_buffer->encoder, nil);
  if (command_buffer->residency_set) {
    [command_buffer->residency_set release];  // -1
  }
  if (command_buffer->argument_table) {
    [command_buffer->argument_table release];  // -1
  }
  if (command_buffer->command_allocator) {
    [command_buffer->command_allocator release];  // -1
  }
  [command_buffer->command_buffer release];  // -1
  [command_buffer->queue release];           // -1
  // Release per-command-buffer indirect bindings buffer if allocated.
  if (command_buffer->indirect_bindings_buffer != nil) {
    [command_buffer->indirect_bindings_buffer release];  // -1
  }
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(command_buffer->host_allocator, command_buffer);
}

static void iree_hal_metal_command_buffer_destroy(iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  iree_hal_device_t* device = command_buffer->device;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Decrease command buffer refcount in the shared staging buffer, and potentially reclaim
  // resources. We tie this to the command buffer's lifetime to avoid resource leak.
  if (command_buffer->staging_buffer) {
    iree_hal_metal_staging_buffer_decrease_command_buffer_refcount(command_buffer->staging_buffer);
  }

  iree_hal_metal_command_buffer_destroy_internal(base_command_buffer);

  iree_hal_device_release(device);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_metal_command_buffer_isa(iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource, &iree_hal_metal_command_buffer_vtable);
}

static iree_status_t iree_hal_metal_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color, const iree_hal_label_location_t* location) {
  // TODO(antiagainst): implement support for debug group
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(antiagainst): implement support for debug group
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_prepare_barrier(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count, const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count, const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "non-zero barrier flag not yet supported");
  }

  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  iree_host_size_t buffer_barrier_length = buffer_barrier_count * sizeof(iree_hal_buffer_barrier_t);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment) + buffer_barrier_length,
                              (void**)&storage_base));

  // Copy the buffer barriers to the end of the current segments for later access. We don't copy
  // memory barriers because in Metal there is only coarse-grained full memory barrier affecting
  // all buffers, regardless of the fine-grained details from IREE HAL barriers.
  uint8_t* barrier_ptr = storage_base + sizeof(*segment);
  memcpy(barrier_ptr, (const uint8_t*)buffer_barriers, buffer_barrier_length);

  // Compose and push the barrier segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_BARRIER;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->barrier.memory_barrier_count = memory_barrier_count;
  segment->barrier.buffer_barrier_count = buffer_barrier_count;
  segment->barrier.buffer_barriers = (const iree_hal_buffer_barrier_t*)barrier_ptr;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record_barrier(
    iree_hal_metal_command_buffer_t* command_buffer, iree_hal_metal_barrier_segment_t* segment) {
  // MTL4 uses unified barriers on the compute encoder. The barrier operates on encoder stages
  // rather than requiring encoder switching.
  id<MTL4ComputeCommandEncoder> encoder = iree_hal_metal_get_or_begin_encoder(command_buffer);

  // All IREE compute dispatches and blit operations go through the unified encoder.
  // Use MTLStageDispatch | MTLStageBlit to cover both.
  MTLStages stages = MTLStageDispatch | MTLStageBlit;

  if (segment->memory_barrier_count == 0 && segment->buffer_barrier_count == 0) {
    // Execution-only barrier (no memory visibility needed).
    [encoder barrierAfterEncoderStages:stages
                   beforeEncoderStages:stages
                     visibilityOptions:MTL4VisibilityOptionNone];
    return iree_ok_status();
  }

  // Memory barrier — MTL4 does not support per-resource barriers, so we use a full device barrier.
  // This covers both the memory_barrier and buffer_barrier cases.
  [encoder barrierAfterEncoderStages:stages
                 beforeEncoderStages:stages
                   visibilityOptions:MTL4VisibilityOptionDevice];
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer, iree_host_size_t event_count,
    const iree_hal_event_t** events, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers, iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_ref_t buffer_ref,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  // This is a hint to the device and we have nothing to do for Metal.
  return iree_ok_status();
}

// Fills |value| with the duplicated single byte value and return true if the given |pattern| has
// duplicated values for each of its |pattern_length| bytes.
static bool iree_hal_metal_get_duplicated_single_byte_value(const void* pattern,
                                                            size_t pattern_length, uint8_t* value) {
  switch (pattern_length) {
    case 1: {
      *value = *(uint8_t*)pattern;
      return true;
    }
    case 2: {
      uint16_t two_bytes = *(uint16_t*)pattern;
      uint16_t byte0 = two_bytes & 0xffu;
      uint16_t byte1 = two_bytes >> 8u;
      if (byte0 == byte1) {
        *value = (int8_t)byte0;
        return true;
      }
      break;
    }
    case 4: {
      uint32_t four_bytes = *(uint32_t*)pattern;
      uint32_t byte0 = four_bytes & 0xffu;
      uint32_t byte1 = (four_bytes >> 8u) & 0xffu;
      uint32_t byte2 = (four_bytes >> 16u) & 0xffu;
      uint32_t byte3 = four_bytes >> 24u;
      if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) {
        *value = (int8_t)byte0;
        return true;
      }
      break;
    }
    default:
      break;
  }
  return false;
}

// Duplicates the given |pattern| into 4-bytes and returns the value.
static uint32_t iree_hal_metal_duplicate_to_four_byte_value(const void* pattern,
                                                            size_t pattern_length) {
  if (pattern_length == 1) {
    uint8_t single_byte = *(uint8_t*)pattern;
    uint32_t value = (uint32_t)single_byte;
    value |= (value << 8u);
    value |= (value << 16u);
    return value;
  }

  if (pattern_length == 2) {
    uint16_t two_bytes = *(uint16_t*)pattern;
    uint32_t value = (uint32_t)two_bytes;
    value |= (value << 16u);
    return value;
  }

  IREE_ASSERT(pattern_length == 4);
  return *(uint32_t*)pattern;
}

static iree_status_t iree_hal_metal_command_buffer_prepare_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_ref_t target_ref,
    const void* pattern, iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment) + pattern_length,
                              (void**)&storage_base));

  // Copy the patttern to the end of the segment for later access.
  uint8_t* pattern_ptr = storage_base + sizeof(*segment);
  memcpy(pattern_ptr, (const uint8_t*)pattern, pattern_length);

  // Compose and push the fill buffer segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_FILL_BUFFER;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->fill_buffer.target_buffer = target_device_buffer;
  segment->fill_buffer.target_offset = target_offset;
  segment->fill_buffer.length = target_ref.length;
  segment->fill_buffer.pattern = (const void*)pattern_ptr;
  segment->fill_buffer.pattern_length = pattern_length;

  iree_status_t status =
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &target_ref.buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_segment_record_fill_buffer(
    iree_hal_metal_command_buffer_t* command_buffer,
    iree_hal_metal_fill_buffer_segment_t* segment) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Note that fillBuffer:range:value: only accepts a single byte as the pattern but FillBuffer
  // can accept 1/2/4 bytes. If the pattern itself contains repeated bytes, we can call into
  // fillBuffer:range:value:. Otherwise we need to emulate the support.
  uint8_t pattern_1byte = 0u;

  // Per the spec for fillBuffer:range:value: "The alignment and length of the range must both be a
  // multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS."
#if defined(IREE_PLATFORM_MACOS)
  const bool can_use_metal_api = segment->target_offset % 4 == 0 && segment->length % 4 == 0 &&
                                 iree_hal_metal_get_duplicated_single_byte_value(
                                     segment->pattern, segment->pattern_length, &pattern_1byte);
#else
  const bool can_use_metal_api = iree_hal_metal_get_duplicated_single_byte_value(
      segment->pattern, segment->pattern_length, &pattern_1byte);
#endif

  if (can_use_metal_api) {
    // MTL4 unified encoder supports fillBuffer:range:value: directly.
    id<MTL4ComputeCommandEncoder> encoder = iree_hal_metal_get_or_begin_encoder(command_buffer);
    [encoder fillBuffer:segment->target_buffer
                  range:NSMakeRange(segment->target_offset, segment->length)
                  value:pattern_1byte];
    // Track residency for the target buffer.
    [command_buffer->residency_set addAllocation:segment->target_buffer];
    [command_buffer->residency_set commit];
    [command_buffer->residency_set requestResidency];
    [command_buffer->command_buffer useResidencySet:command_buffer->residency_set];
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  id<MTL4ComputeCommandEncoder> encoder = iree_hal_metal_get_or_begin_encoder(command_buffer);
  uint32_t pattern_4byte =
      iree_hal_metal_duplicate_to_four_byte_value(segment->pattern, segment->pattern_length);
  iree_status_t status = iree_hal_metal_builtin_executable_fill_buffer(
      command_buffer->builtin_executable, encoder, segment->target_buffer,
      segment->target_offset, segment->length, pattern_4byte,
      command_buffer->argument_table, command_buffer->staging_buffer,
      command_buffer->residency_set);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_segment_create_copy_buffer(
    iree_hal_metal_command_buffer_t* command_buffer, id<MTLBuffer> source_device_buffer,
    iree_device_size_t source_offset, id<MTLBuffer> target_device_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, sizeof(*segment), (void**)&storage_base));

  // Compose and push the barrier segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_COPY_BUFFER;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->copy_buffer.source_buffer = source_device_buffer;
  segment->copy_buffer.source_offset = source_offset;
  segment->copy_buffer.target_buffer = target_device_buffer;
  segment->copy_buffer.target_offset = target_offset;
  segment->copy_buffer.length = length;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record_copy_buffer(
    iree_hal_metal_command_buffer_t* command_buffer,
    iree_hal_metal_copy_buffer_segment_t* segment) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Per the spec for copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size, the source/target
  // offset and length must be a multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS.
#if defined(IREE_PLATFORM_MACOS)
  bool can_use_metal_api = segment->source_offset % 4 == 0 && segment->target_offset % 4 == 0 &&
                           segment->length % 4 == 0;
#else
  bool can_use_metal_api = true;
#endif

  iree_status_t status = iree_ok_status();
  if (can_use_metal_api) {
    // MTL4 unified encoder supports copyFromBuffer: directly.
    id<MTL4ComputeCommandEncoder> encoder = iree_hal_metal_get_or_begin_encoder(command_buffer);
    [encoder copyFromBuffer:segment->source_buffer
               sourceOffset:segment->source_offset
                   toBuffer:segment->target_buffer
          destinationOffset:segment->target_offset
                       size:segment->length];
    // Track residency for source and target buffers.
    [command_buffer->residency_set addAllocation:segment->source_buffer];
    [command_buffer->residency_set addAllocation:segment->target_buffer];
    [command_buffer->residency_set commit];
    [command_buffer->residency_set requestResidency];
    [command_buffer->command_buffer useResidencySet:command_buffer->residency_set];
  } else {
    id<MTL4ComputeCommandEncoder> encoder = iree_hal_metal_get_or_begin_encoder(command_buffer);
    status = iree_hal_metal_builtin_executable_copy_buffer(
        command_buffer->builtin_executable, encoder, segment->source_buffer, segment->source_offset,
        segment->target_buffer, segment->target_offset, segment->length,
        command_buffer->argument_table, command_buffer->staging_buffer,
        command_buffer->residency_set);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_prepare_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // There are no direct corresponding APIs in Metal. We update the source buffer data to the
  // staging buffer and then copy over.

  iree_const_byte_span_t source_data_span =
      iree_make_const_byte_span((uint8_t*)source_buffer + source_offset, target_ref.length);
  uint32_t offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_staging_buffer_append(command_buffer->staging_buffer, source_data_span,
                                               /*alignment=*/4, &offset));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &target_ref.buffer));

  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  iree_status_t status = iree_hal_metal_command_segment_create_copy_buffer(
      command_buffer, command_buffer->staging_buffer->metal_buffer, offset, target_device_buffer,
      target_offset, target_ref.length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_prepare_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_ref_t source_ref,
    iree_hal_buffer_ref_t target_ref, iree_hal_copy_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_buffer_t* resources[2] = {source_ref.buffer, target_ref.buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, IREE_ARRAYSIZE(resources),
                                       resources));

  id<MTLBuffer> source_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(source_ref.buffer));
  id<MTLBuffer> target_device_buffer =
      iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(target_ref.buffer));

  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  iree_status_t status = iree_hal_metal_command_segment_create_copy_buffer(
      command_buffer, source_device_buffer, source_offset, target_device_buffer, target_offset,
      target_ref.length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "collectives not yet supported");
}

// Prepares kernels and argument buffers needed for kernel dispatches.
static iree_status_t iree_hal_metal_command_buffer_prepare_dispatch(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal, const iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO: support custom args (should be easy) and indirect arguments (via
  // setKernelBuffer:offset:atIndex:).
  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "direct/indirect arguments are not supported on Metal");
  }

  iree_host_size_t resource_count = 1;
  const void* resources[2] = {executable, NULL};
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    resources[resource_count++] = config.workgroup_count_ref.buffer;
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, resource_count, &executable));

  const iree_hal_metal_pipeline_t* pipeline = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_executable_lookup_pipeline(executable, export_ordinal, &pipeline));

  // Allocate the command segment and keep track of all necessary API data.
  uint8_t* storage_base = NULL;
  iree_hal_metal_command_segment_t* segment = NULL;
  iree_host_size_t descriptor_length = bindings.count * sizeof(iree_hal_metal_descriptor_t);
  iree_host_size_t total_size = sizeof(*segment) + descriptor_length + constants.data_length;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size, (void**)&storage_base));

  // Compose and push the dispatch segment.
  segment = (iree_hal_metal_command_segment_t*)storage_base;
  memset(segment, 0, sizeof(*segment));
  segment->action = IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_DISPATCH;
  iree_hal_metal_command_segment_list_push_back(&command_buffer->segments, segment);

  segment->dispatch.pipeline = pipeline;
  segment->dispatch.threadgroup_size =
      config.workgroup_size[0] ? MTLSizeMake(config.workgroup_size[0], config.workgroup_size[1],
                                             config.workgroup_size[2])
                               : pipeline->threadgroup_size;

  // Copy descriptors from all sets to the end of the current segment for later access.
  segment->dispatch.descriptor_count = bindings.count;
  segment->dispatch.descriptors = (iree_hal_metal_descriptor_t*)(storage_base + sizeof(*segment));
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    iree_hal_metal_descriptor_t* descriptor = &segment->dispatch.descriptors[i];

    descriptor->binding = i;
    descriptor->buffer = bindings.values[i].buffer;
    descriptor->offset = bindings.values[i].offset;

    MTLResourceUsage usage = MTLResourceUsageRead;
    uint64_t binding_bit = 1ull << i;
    if (iree_any_bit_set(pipeline->binding_read_only_bits, binding_bit)) {
      usage |= MTLResourceUsageWrite;
    }
    descriptor->usage = usage;

    if (descriptor->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1, &descriptor->buffer));
    }
  }

  // Copy push constants to the end of the current segment for later access.
  segment->dispatch.constant_count = constants.data_length / sizeof(int32_t);
  uint8_t* constant_ptr = storage_base + sizeof(*segment) + descriptor_length;
  segment->dispatch.constants = (int32_t*)constant_ptr;
  memcpy(constant_ptr, constants.data, constants.data_length);

  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    segment->dispatch.workgroups_buffer = iree_hal_metal_buffer_handle(
        iree_hal_buffer_allocated_buffer(config.workgroup_count_ref.buffer));
    segment->dispatch.workgroups_offset = config.workgroup_count_ref.offset;
  } else {
    segment->dispatch.workgroup_count = MTLSizeMake(
        config.workgroup_count[0], config.workgroup_count[1], config.workgroup_count[2]);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record_dispatch(
    iree_hal_metal_command_buffer_t* command_buffer, iree_hal_metal_dispatch_segment_t* segment) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get or create the unified encoder.
  id<MTL4ComputeCommandEncoder> encoder = iree_hal_metal_get_or_begin_encoder(command_buffer);

  // Set the compute kernel to dispatch.
  [encoder setComputePipelineState:segment->pipeline->pipeline_state];

  // Record argument buffers for all descriptors via the argument table.
  iree_hal_metal_descriptor_t* descriptors = segment->descriptors;
  id<MTL4ArgumentTable> arg_table = command_buffer->argument_table;

  if (segment->pipeline->uses_indirect_bindings) {
    // For indirect bindings (PhysicalStorageBuffer), build a two-level pointer structure.
    // The MSL structure is:
    //   struct _6 { device T* _m0; device T* _m1; ... };  // inner: actual buffer pointers
    //   struct spvDescriptorSetBuffer3 { device _6* _resource_var_indirect_0_; };  // outer
    // Buffer(3) must contain spvDescriptorSetBuffer3, which points to _6.

    // Determine the maximum binding index to size the inner struct.
    uint32_t max_binding = 0;
    for (iree_host_size_t i = 0; i < segment->descriptor_count; ++i) {
      if (descriptors[i].binding > max_binding) {
        max_binding = descriptors[i].binding;
      }
    }
    size_t inner_struct_size = (max_binding + 1) * sizeof(uint64_t);

    // Reserve space for inner struct (contains GPU addresses for each binding).
    uint8_t* inner_host_ptr = NULL;
    uint32_t inner_offset = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_metal_indirect_bindings_buffer_reserve(
                command_buffer, inner_struct_size,
                /*alignment=*/sizeof(uint64_t), &inner_host_ptr, &inner_offset));

    // Fill the inner struct with GPU addresses and add buffers to residency set.
    uint64_t* address_table = (uint64_t*)inner_host_ptr;
    memset(address_table, 0, inner_struct_size);
    for (iree_host_size_t i = 0; i < segment->descriptor_count; ++i) {
      uint32_t current_binding = descriptors[i].binding;
      id<MTLBuffer> current_buffer =
          iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(descriptors[i].buffer));
      iree_host_size_t offset =
          iree_hal_buffer_byte_offset(descriptors[i].buffer) + descriptors[i].offset;

      address_table[current_binding] = current_buffer.gpuAddress + offset;
      [command_buffer->residency_set addAllocation:current_buffer];
    }

    // Reserve space for outer struct (contains pointer to inner struct).
    uint8_t* outer_host_ptr = NULL;
    uint32_t outer_offset = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_metal_indirect_bindings_buffer_reserve(
                command_buffer, sizeof(uint64_t),
                /*alignment=*/sizeof(uint64_t), &outer_host_ptr, &outer_offset));

    // Fill the outer struct with pointer to inner struct.
    id<MTLBuffer> indirect_buffer = command_buffer->indirect_bindings_buffer;
    uint64_t* outer_ptr = (uint64_t*)outer_host_ptr;
    *outer_ptr = indirect_buffer.gpuAddress + inner_offset;

    // Ensure CPU writes to the dedicated buffer are visible to GPU.
#if defined(__aarch64__)
    __asm__ __volatile__("dmb ishst" ::: "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif

    // Bind the outer struct address via the argument table at index 3.
    [arg_table setAddress:(indirect_buffer.gpuAddress + outer_offset) atIndex:3];
    [command_buffer->residency_set addAllocation:indirect_buffer];
  } else {
    // Standard path: build an argument buffer at [[buffer(0)]] containing GPU addresses.
    // SPIRV-Cross generates MSL that uses argument buffers:
    //   constant spvDescriptorSetBuffer0& descriptorSet [[buffer(0)]]
    //   with members: device T* [[id(N)]] at position N * sizeof(uint64_t).
    // In Metal 3, MTLArgumentEncoder handled this layout. In Metal 4, we build
    // the argument buffer manually and bind its address at argument table index 0.

    // Determine the maximum binding index to size the argument buffer.
    uint32_t max_binding = 0;
    for (iree_host_size_t i = 0; i < segment->descriptor_count; ++i) {
      if (descriptors[i].binding > max_binding) {
        max_binding = descriptors[i].binding;
      }
    }
    size_t arg_buffer_size = (max_binding + 1) * sizeof(uint64_t);

    // Reserve space for the argument buffer (array of GPU addresses).
    uint8_t* arg_host_ptr = NULL;
    uint32_t arg_offset = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_metal_indirect_bindings_buffer_reserve(
                command_buffer, arg_buffer_size,
                /*alignment=*/sizeof(uint64_t), &arg_host_ptr, &arg_offset));

    // Fill the argument buffer with GPU addresses for each binding.
    uint64_t* address_table = (uint64_t*)arg_host_ptr;
    memset(address_table, 0, arg_buffer_size);
    for (iree_host_size_t i = 0; i < segment->descriptor_count; ++i) {
      uint32_t current_binding = descriptors[i].binding;
      id<MTLBuffer> current_buffer =
          iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(descriptors[i].buffer));
      iree_host_size_t offset =
          iree_hal_buffer_byte_offset(descriptors[i].buffer) + descriptors[i].offset;

      address_table[current_binding] = current_buffer.gpuAddress + offset;
      [command_buffer->residency_set addAllocation:current_buffer];
    }

    // Ensure CPU writes to the argument buffer are visible to GPU.
#if defined(__aarch64__)
    __asm__ __volatile__("dmb ishst" ::: "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif

    // Bind the argument buffer at index 0 (matching [[buffer(0)]] in the shader).
    id<MTLBuffer> indirect_buffer = command_buffer->indirect_bindings_buffer;
    [arg_table setAddress:(indirect_buffer.gpuAddress + arg_offset) atIndex:0];
    [command_buffer->residency_set addAllocation:indirect_buffer];
  }

  // Record push constants — MTL4 has no inline setBytes, so write to staging buffer
  // and bind via argument table.
  if (segment->constant_count != 0) {
    iree_const_byte_span_t constants_span = iree_make_const_byte_span(
        (const uint8_t*)segment->constants, segment->constant_count * sizeof(int32_t));
    uint32_t constants_offset = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_metal_staging_buffer_append(
                command_buffer->staging_buffer, constants_span,
                /*alignment=*/sizeof(int32_t), &constants_offset));
    id<MTLBuffer> staging_metal = command_buffer->staging_buffer->metal_buffer;
    [arg_table setAddress:(staging_metal.gpuAddress + constants_offset)
                  atIndex:IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX];
    [command_buffer->residency_set addAllocation:staging_metal];
  }

  // Commit residency changes and attach to command buffer.
  [command_buffer->residency_set commit];
  [command_buffer->residency_set requestResidency];
  [command_buffer->command_buffer useResidencySet:command_buffer->residency_set];

  // Set the argument table on the encoder.
  [encoder setArgumentTable:arg_table];

  // Record the dispatch, either direct or indirect.
  if (segment->workgroups_buffer == nil) {
    // Direct dispatch of a fixed workgroup count.
    [encoder dispatchThreadgroups:segment->workgroup_count
            threadsPerThreadgroup:segment->threadgroup_size];
  } else {
    // Indirect dispatch using a GPU address for the workgroup count buffer.
    MTLGPUAddress indirect_addr =
        segment->workgroups_buffer.gpuAddress + segment->workgroups_offset;
    [encoder dispatchThreadgroupsWithIndirectBuffer:indirect_addr
                              threadsPerThreadgroup:segment->threadgroup_size];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_segment_record(
    iree_hal_metal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_hal_metal_command_segment_t* segment = command_buffer->segments.head; segment;
       segment = segment->next_segment) {
    switch (segment->action) {
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_BARRIER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_metal_command_segment_record_barrier(command_buffer, &segment->barrier));
      } break;
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_DISPATCH: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_metal_command_segment_record_dispatch(command_buffer, &segment->dispatch));
      } break;
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_FILL_BUFFER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_command_segment_record_fill_buffer(
                                                  command_buffer, &segment->fill_buffer));
      } break;
      case IREE_HAL_METAL_COMMAND_SEGMENT_ACTION_COPY_BUFFER: {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_command_segment_record_copy_buffer(
                                                  command_buffer, &segment->copy_buffer));
      } break;
      default:
        IREE_ASSERT(false, "unhandled command segment kind");
        break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  iree_hal_metal_command_buffer_reset(command_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_metal_command_buffer_t* command_buffer =
      iree_hal_metal_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Begin the command buffer with the allocator before recording.
  [command_buffer->command_buffer beginCommandBufferWithAllocator:command_buffer->command_allocator];

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_metal_command_segment_record(command_buffer));
  iree_hal_metal_end_encoder(command_buffer);

  // End the command buffer (required before queue commit).
  [command_buffer->command_buffer endCommandBuffer];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t iree_hal_metal_command_buffer_vtable = {
    .destroy = iree_hal_metal_command_buffer_destroy,
    .begin = iree_hal_metal_command_buffer_begin,
    .end = iree_hal_metal_command_buffer_end,
    .begin_debug_group = iree_hal_metal_command_buffer_begin_debug_group,
    .end_debug_group = iree_hal_metal_command_buffer_end_debug_group,
    .execution_barrier = iree_hal_metal_command_buffer_prepare_barrier,
    .signal_event = iree_hal_metal_command_buffer_signal_event,
    .reset_event = iree_hal_metal_command_buffer_reset_event,
    .wait_events = iree_hal_metal_command_buffer_wait_events,
    .advise_buffer = iree_hal_metal_command_buffer_advise_buffer,
    .fill_buffer = iree_hal_metal_command_buffer_prepare_fill_buffer,
    .update_buffer = iree_hal_metal_command_buffer_prepare_update_buffer,
    .copy_buffer = iree_hal_metal_command_buffer_prepare_copy_buffer,
    .collective = iree_hal_metal_command_buffer_collective,
    .dispatch = iree_hal_metal_command_buffer_prepare_dispatch,
};
