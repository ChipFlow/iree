// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/sparse_solver/module.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/modules/sparse_solver/baspacho_wrapper.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module.h"

//===----------------------------------------------------------------------===//
// Module Version
//===----------------------------------------------------------------------===//

#define IREE_SPARSE_SOLVER_MODULE_VERSION_0_1 0x00000001u
#define IREE_SPARSE_SOLVER_MODULE_VERSION_LATEST \
  IREE_SPARSE_SOLVER_MODULE_VERSION_0_1

//===----------------------------------------------------------------------===//
// Backend Detection
//===----------------------------------------------------------------------===//

static baspacho_backend_t iree_sparse_solver_detect_backend(
    iree_hal_device_t* device) {
  iree_string_view_t device_id = iree_hal_device_id(device);

  if (iree_string_view_starts_with(device_id, IREE_SV("cuda"))) {
    return BASPACHO_BACKEND_CUDA;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("metal"))) {
    return BASPACHO_BACKEND_METAL;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("vulkan")) ||
             iree_string_view_starts_with(device_id, IREE_SV("opencl"))) {
    return BASPACHO_BACKEND_OPENCL;
  }
  return BASPACHO_BACKEND_CPU;
}

//===----------------------------------------------------------------------===//
// Module State
//===----------------------------------------------------------------------===//

// Cast from base module to our module type.
#define IREE_SPARSE_SOLVER_MODULE_CAST(base_module)  \
  ((iree_sparse_solver_module_t*)((uint8_t*)(base_module) + \
                                  iree_vm_native_module_size()))

typedef struct iree_sparse_solver_module_t {
  iree_allocator_t host_allocator;
  iree_hal_device_t* device;
  baspacho_backend_t backend;
  iree_sparse_solver_module_flags_t flags;
} iree_sparse_solver_module_t;

typedef struct iree_sparse_solver_module_state_t {
  iree_allocator_t host_allocator;
  iree_sparse_solver_module_t* module;
} iree_sparse_solver_module_state_t;

static void IREE_API_PTR iree_sparse_solver_module_destroy(void* self) {
  iree_vm_module_t* base_module = (iree_vm_module_t*)self;
  iree_sparse_solver_module_t* module =
      IREE_SPARSE_SOLVER_MODULE_CAST(base_module);
  iree_hal_device_release(module->device);
  // NOTE: Native module framework handles freeing base_module memory.
}

static iree_status_t IREE_API_PTR iree_sparse_solver_module_alloc_state(
    void* self, iree_allocator_t host_allocator,
    iree_vm_module_state_t** out_module_state) {
  iree_vm_module_t* base_module = (iree_vm_module_t*)self;
  iree_sparse_solver_module_t* module =
      IREE_SPARSE_SOLVER_MODULE_CAST(base_module);
  iree_sparse_solver_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->module = module;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR iree_sparse_solver_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_sparse_solver_module_state_t* state =
      (iree_sparse_solver_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);
}

//===----------------------------------------------------------------------===//
// BaSpaCho Handle Reference Type
//===----------------------------------------------------------------------===//

// Ref type for wrapping baspacho_handle_t in VM references.
typedef struct iree_sparse_solver_handle_t {
  iree_vm_ref_object_t ref_object;
  baspacho_handle_t baspacho_handle;
  iree_allocator_t host_allocator;
} iree_sparse_solver_handle_t;

// Forward declaration of registration variable for type() function.
static iree_vm_ref_type_t iree_sparse_solver_handle_registration_;

// Returns the registered type for sparse solver handles.
static inline iree_vm_ref_type_t iree_sparse_solver_handle_type(void) {
  return iree_sparse_solver_handle_registration_;
}

// Type descriptor storage for registration.
static iree_vm_ref_type_descriptor_t iree_sparse_solver_handle_descriptor_;

static void IREE_API_PTR iree_sparse_solver_handle_destroy(void* ptr) {
  iree_sparse_solver_handle_t* handle = (iree_sparse_solver_handle_t*)ptr;
  if (handle->baspacho_handle) {
    baspacho_destroy(handle->baspacho_handle);
  }
  iree_allocator_free(handle->host_allocator, handle);
}

static iree_status_t iree_sparse_solver_handle_create(
    baspacho_handle_t baspacho_handle,
    iree_allocator_t host_allocator,
    iree_sparse_solver_handle_t** out_handle) {
  iree_sparse_solver_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*handle), (void**)&handle));
  // Initialize ref count to 1.
  iree_atomic_ref_count_init(&handle->ref_object.counter);
  handle->baspacho_handle = baspacho_handle;
  handle->host_allocator = host_allocator;
  *out_handle = handle;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Sparse Solver Exports
//===----------------------------------------------------------------------===//

// analyze: rIIrr -> r
// Note: First argument is device ref, which we ignore since module already
// has the device. This allows the MLIR interface to be device-parametric
// while we use the device from module creation.
static iree_status_t iree_sparse_solver_analyze_impl(
    iree_vm_stack_t* IREE_RESTRICT stack,
    iree_sparse_solver_module_t* module,
    iree_sparse_solver_module_state_t* state,
    int64_t n, int64_t nnz,
    iree_hal_buffer_view_t* row_ptr_view,
    iree_hal_buffer_view_t* col_idx_view,
    iree_vm_ref_t* out_handle) {
  // Create BaSpaCho context with detected backend.
  baspacho_handle_t baspacho = baspacho_create(module->backend);
  if (!baspacho) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to create BaSpaCho context");
  }

  // Map row_ptr buffer to get data.
  iree_hal_buffer_t* row_ptr_buffer = iree_hal_buffer_view_buffer(row_ptr_view);
  iree_hal_buffer_mapping_t row_ptr_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      row_ptr_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &row_ptr_mapping));

  // Map col_idx buffer to get data.
  iree_hal_buffer_t* col_idx_buffer = iree_hal_buffer_view_buffer(col_idx_view);
  iree_hal_buffer_mapping_t col_idx_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      col_idx_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &col_idx_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_unmap_range(&row_ptr_mapping);
    baspacho_destroy(baspacho);
    return status;
  }

  // Perform symbolic analysis.
  int result = baspacho_analyze(baspacho, n, nnz,
                                 (const int64_t*)row_ptr_mapping.contents.data,
                                 (const int64_t*)col_idx_mapping.contents.data);

  // Unmap buffers.
  iree_hal_buffer_unmap_range(&col_idx_mapping);
  iree_hal_buffer_unmap_range(&row_ptr_mapping);

  if (result != 0) {
    baspacho_destroy(baspacho);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "BaSpaCho symbolic analysis failed with code %d",
                            result);
  }

  // Wrap in VM ref.
  iree_sparse_solver_handle_t* handle = NULL;
  status = iree_sparse_solver_handle_create(
      baspacho, module->host_allocator, &handle);
  if (!iree_status_is_ok(status)) {
    baspacho_destroy(baspacho);
    return status;
  }

  iree_vm_ref_wrap_assign(handle, iree_sparse_solver_handle_type(),
                           out_handle);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_sparse_solver_analyze,
                   iree_sparse_solver_module_state_t, rIIrr, r) {
  iree_sparse_solver_module_t* sparse_module = state->module;

  // args->r0 is the device ref - we ignore it and use module->device instead.
  // This is intentional: the MLIR interface takes device for consistency,
  // but we use the device that was provided at module creation.
  (void)args->r0;

  int64_t n = args->i1;
  int64_t nnz = args->i2;
  iree_hal_buffer_view_t* row_ptr_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r3, &row_ptr_view));
  iree_hal_buffer_view_t* col_idx_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r4, &col_idx_view));

  // Copy ret ref to local variable to avoid packed member address warning.
  iree_vm_ref_t out_handle = {0};
  iree_status_t status = iree_sparse_solver_analyze_impl(
      stack, sparse_module, state, n, nnz, row_ptr_view,
      col_idx_view, &out_handle);
  rets->r0 = out_handle;
  return status;
}

// release: r -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_release,
                   iree_sparse_solver_module_state_t, r, v) {
  // Copy ref to local variable to avoid packed member address warning.
  iree_vm_ref_t ref = args->r0;
  iree_vm_ref_release(&ref);
  return iree_ok_status();
}

// Helper to extract BaSpaCho handle from VM ref.
static iree_sparse_solver_handle_t* iree_sparse_solver_get_handle(
    iree_vm_ref_t* ref) {
  if (!ref || iree_vm_ref_is_null(ref)) return NULL;
  // Check type matches and return pointer.
  if (ref->type == iree_sparse_solver_handle_type()) {
    return (iree_sparse_solver_handle_t*)ref->ptr;
  }
  return NULL;
}

// factor: rr -> i
IREE_VM_ABI_EXPORT(iree_sparse_solver_factor,
                   iree_sparse_solver_module_state_t, rr, i) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    rets->i0 = -1;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }

  iree_hal_buffer_view_t* values_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &values_view));

  // Map values buffer.
  iree_hal_buffer_t* values_buffer = iree_hal_buffer_view_buffer(values_view);
  iree_hal_buffer_mapping_t values_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      values_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &values_mapping));

  // Perform numeric factorization.
  int result = baspacho_factor_f32(handle->baspacho_handle,
                                   (const float*)values_mapping.contents.data);

  iree_hal_buffer_unmap_range(&values_mapping);
  rets->i0 = result;
  return iree_ok_status();
}

// factor.f64: rr -> i
IREE_VM_ABI_EXPORT(iree_sparse_solver_factor_f64,
                   iree_sparse_solver_module_state_t, rr, i) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    rets->i0 = -1;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }

  iree_hal_buffer_view_t* values_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &values_view));

  iree_hal_buffer_t* values_buffer = iree_hal_buffer_view_buffer(values_view);
  iree_hal_buffer_mapping_t values_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      values_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &values_mapping));

  int result = baspacho_factor_f64(handle->baspacho_handle,
                                   (const double*)values_mapping.contents.data);

  iree_hal_buffer_unmap_range(&values_mapping);
  rets->i0 = result;
  return iree_ok_status();
}

// solve: rrr -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve,
                   iree_sparse_solver_module_state_t, rrr, v) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }

  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &rhs_view));
  iree_hal_buffer_view_t* solution_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r2, &solution_view));

  // Map RHS buffer (read-only).
  iree_hal_buffer_t* rhs_buffer = iree_hal_buffer_view_buffer(rhs_view);
  iree_hal_buffer_mapping_t rhs_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      rhs_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &rhs_mapping));

  // Map solution buffer (read-write).
  iree_hal_buffer_t* solution_buffer = iree_hal_buffer_view_buffer(solution_view);
  iree_hal_buffer_mapping_t solution_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      solution_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE, 0, IREE_HAL_WHOLE_BUFFER, &solution_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_unmap_range(&rhs_mapping);
    return status;
  }

  // Solve in-place.
  baspacho_solve_f32(handle->baspacho_handle,
                     (const float*)rhs_mapping.contents.data,
                     (float*)solution_mapping.contents.data);

  iree_hal_buffer_unmap_range(&solution_mapping);
  iree_hal_buffer_unmap_range(&rhs_mapping);
  return iree_ok_status();
}

// solve.f64: rrr -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve_f64,
                   iree_sparse_solver_module_state_t, rrr, v) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }

  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &rhs_view));
  iree_hal_buffer_view_t* solution_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r2, &solution_view));

  iree_hal_buffer_t* rhs_buffer = iree_hal_buffer_view_buffer(rhs_view);
  iree_hal_buffer_mapping_t rhs_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      rhs_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &rhs_mapping));

  iree_hal_buffer_t* solution_buffer = iree_hal_buffer_view_buffer(solution_view);
  iree_hal_buffer_mapping_t solution_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      solution_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE, 0, IREE_HAL_WHOLE_BUFFER, &solution_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_unmap_range(&rhs_mapping);
    return status;
  }

  baspacho_solve_f64(handle->baspacho_handle,
                     (const double*)rhs_mapping.contents.data,
                     (double*)solution_mapping.contents.data);

  iree_hal_buffer_unmap_range(&solution_mapping);
  iree_hal_buffer_unmap_range(&rhs_mapping);
  return iree_ok_status();
}

// solve.batched: rrrI -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve_batched,
                   iree_sparse_solver_module_state_t, rrrI, v) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }

  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &rhs_view));
  iree_hal_buffer_view_t* solution_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r2, &solution_view));
  int64_t num_rhs = args->i3;

  iree_hal_buffer_t* rhs_buffer = iree_hal_buffer_view_buffer(rhs_view);
  iree_hal_buffer_mapping_t rhs_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      rhs_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &rhs_mapping));

  iree_hal_buffer_t* solution_buffer = iree_hal_buffer_view_buffer(solution_view);
  iree_hal_buffer_mapping_t solution_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      solution_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE, 0, IREE_HAL_WHOLE_BUFFER, &solution_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_unmap_range(&rhs_mapping);
    return status;
  }

  baspacho_solve_batched_f32(handle->baspacho_handle,
                              (const float*)rhs_mapping.contents.data,
                              (float*)solution_mapping.contents.data,
                              num_rhs);

  iree_hal_buffer_unmap_range(&solution_mapping);
  iree_hal_buffer_unmap_range(&rhs_mapping);
  return iree_ok_status();
}

// solve.batched.f64: rrrI -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve_batched_f64,
                   iree_sparse_solver_module_state_t, rrrI, v) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }

  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &rhs_view));
  iree_hal_buffer_view_t* solution_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r2, &solution_view));
  int64_t num_rhs = args->i3;

  iree_hal_buffer_t* rhs_buffer = iree_hal_buffer_view_buffer(rhs_view);
  iree_hal_buffer_mapping_t rhs_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      rhs_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &rhs_mapping));

  iree_hal_buffer_t* solution_buffer = iree_hal_buffer_view_buffer(solution_view);
  iree_hal_buffer_mapping_t solution_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      solution_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE, 0, IREE_HAL_WHOLE_BUFFER, &solution_mapping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_unmap_range(&rhs_mapping);
    return status;
  }

  baspacho_solve_batched_f64(handle->baspacho_handle,
                              (const double*)rhs_mapping.contents.data,
                              (double*)solution_mapping.contents.data,
                              num_rhs);

  iree_hal_buffer_unmap_range(&solution_mapping);
  iree_hal_buffer_unmap_range(&rhs_mapping);
  return iree_ok_status();
}

// get_factor_nnz: r -> I
IREE_VM_ABI_EXPORT(iree_sparse_solver_get_factor_nnz,
                   iree_sparse_solver_module_state_t, r, I) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    rets->i0 = 0;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }
  rets->i0 = baspacho_get_factor_nnz(handle->baspacho_handle);
  return iree_ok_status();
}

// get_num_supernodes: r -> I
IREE_VM_ABI_EXPORT(iree_sparse_solver_get_num_supernodes,
                   iree_sparse_solver_module_state_t, r, I) {
  iree_vm_ref_t handle_ref = args->r0;
  iree_sparse_solver_handle_t* handle = iree_sparse_solver_get_handle(&handle_ref);
  if (!handle || !handle->baspacho_handle) {
    rets->i0 = 0;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid handle");
  }
  rets->i0 = baspacho_get_num_supernodes(handle->baspacho_handle);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VM Module Interface
//===----------------------------------------------------------------------===//

// Define shims for NEW calling conventions only.
// Standard shims (r_v, r_I, rr_i, rrr_v) are already defined in shims.c.
IREE_VM_ABI_DEFINE_SHIM(rrrI, v);   // solve.batched, solve.batched.f64
IREE_VM_ABI_DEFINE_SHIM(rIIrr, r);  // analyze

// Module function table.
static const iree_vm_native_function_ptr_t iree_sparse_solver_module_funcs_[] =
    {
        // analyze
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rIIrr_r,
            .target =
                (iree_vm_native_function_target_t)iree_sparse_solver_analyze,
        },
        // release
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_r_v,
            .target =
                (iree_vm_native_function_target_t)iree_sparse_solver_release,
        },
        // factor
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rr_i,
            .target =
                (iree_vm_native_function_target_t)iree_sparse_solver_factor,
        },
        // factor.f64
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rr_i,
            .target =
                (iree_vm_native_function_target_t)iree_sparse_solver_factor_f64,
        },
        // solve
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rrr_v,
            .target =
                (iree_vm_native_function_target_t)iree_sparse_solver_solve,
        },
        // solve.f64
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rrr_v,
            .target =
                (iree_vm_native_function_target_t)iree_sparse_solver_solve_f64,
        },
        // solve.batched
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rrrI_v,
            .target = (iree_vm_native_function_target_t)
                iree_sparse_solver_solve_batched,
        },
        // solve.batched.f64
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rrrI_v,
            .target = (iree_vm_native_function_target_t)
                iree_sparse_solver_solve_batched_f64,
        },
        // get_factor_nnz
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_r_I,
            .target = (iree_vm_native_function_target_t)
                iree_sparse_solver_get_factor_nnz,
        },
        // get_num_supernodes
        {
            .shim = (iree_vm_native_function_shim_t)iree_vm_shim_r_I,
            .target = (iree_vm_native_function_target_t)
                iree_sparse_solver_get_num_supernodes,
        },
};

// Module exports.
static const iree_vm_native_export_descriptor_t
    iree_sparse_solver_module_exports_[] = {
        {
            .local_name = iree_string_view_literal("analyze"),
            .calling_convention = iree_string_view_literal("0rIIrr_r"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("release"),
            .calling_convention = iree_string_view_literal("0r_v"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("factor"),
            .calling_convention = iree_string_view_literal("0rr_i"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("factor.f64"),
            .calling_convention = iree_string_view_literal("0rr_i"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("solve"),
            .calling_convention = iree_string_view_literal("0rrr_v"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("solve.f64"),
            .calling_convention = iree_string_view_literal("0rrr_v"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("solve.batched"),
            .calling_convention = iree_string_view_literal("0rrrI_v"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("solve.batched.f64"),
            .calling_convention = iree_string_view_literal("0rrrI_v"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("get_factor_nnz"),
            .calling_convention = iree_string_view_literal("0r_I"),
            .attr_count = 0,
            .attrs = NULL,
        },
        {
            .local_name = iree_string_view_literal("get_num_supernodes"),
            .calling_convention = iree_string_view_literal("0r_I"),
            .attr_count = 0,
            .attrs = NULL,
        },
};

static_assert(IREE_ARRAYSIZE(iree_sparse_solver_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_sparse_solver_module_exports_),
              "function pointer table must be 1:1 with exports");

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t
    iree_sparse_solver_module_imports_[1];

static const iree_vm_native_module_descriptor_t
    iree_sparse_solver_module_descriptor_ = {
        .name = iree_string_view_literal("sparse_solver"),
        .version = IREE_SPARSE_SOLVER_MODULE_VERSION_LATEST,
        .attr_count = 0,
        .attrs = NULL,
        .dependency_count = 0,
        .dependencies = NULL,
        .import_count = 0,
        .imports = iree_sparse_solver_module_imports_,
        .export_count = IREE_ARRAYSIZE(iree_sparse_solver_module_exports_),
        .exports = iree_sparse_solver_module_exports_,
        .function_count = IREE_ARRAYSIZE(iree_sparse_solver_module_funcs_),
        .functions = iree_sparse_solver_module_funcs_,
};

// Register the sparse solver handle ref type with the VM.
// This must be called once before creating any modules.
static iree_status_t iree_sparse_solver_register_types(
    iree_vm_instance_t* instance) {
  // Only register once.
  if (iree_sparse_solver_handle_registration_ != 0) {
    return iree_ok_status();
  }

  iree_sparse_solver_handle_descriptor_.type_name =
      iree_make_cstring_view("sparse_solver.handle");
  iree_sparse_solver_handle_descriptor_.offsetof_counter =
      offsetof(iree_sparse_solver_handle_t, ref_object.counter) /
      IREE_VM_REF_COUNTER_ALIGNMENT;
  iree_sparse_solver_handle_descriptor_.destroy =
      iree_sparse_solver_handle_destroy;

  return iree_vm_instance_register_type(
      instance, &iree_sparse_solver_handle_descriptor_,
      &iree_sparse_solver_handle_registration_);
}

IREE_API_EXPORT iree_status_t iree_sparse_solver_module_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_sparse_solver_module_flags_t flags, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Register our custom ref type with the VM.
  IREE_RETURN_IF_ERROR(iree_sparse_solver_register_types(instance));

  // Setup the interface with the functions we implement ourselves.
  static const iree_vm_module_t interface = {
      .destroy = iree_sparse_solver_module_destroy,
      .alloc_state = iree_sparse_solver_module_alloc_state,
      .free_state = iree_sparse_solver_module_free_state,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_sparse_solver_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);

  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_sparse_solver_module_descriptor_, instance,
      host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_sparse_solver_module_t* module =
      IREE_SPARSE_SOLVER_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  module->device = device;
  module->flags = flags;
  iree_hal_device_retain(device);

  // Detect backend based on device type.
  module->backend = iree_sparse_solver_detect_backend(device);

  *out_module = base_module;
  return iree_ok_status();
}
