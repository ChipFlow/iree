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
// Sparse Solver Exports (Stub Implementations)
//===----------------------------------------------------------------------===//

// NOTE: These are stub implementations that return UNIMPLEMENTED.
// Full implementations require integrating with the BaSpaCho wrapper.

// analyze: rIIrr -> r
static iree_status_t iree_sparse_solver_analyze_impl(
    iree_vm_stack_t* IREE_RESTRICT stack,
    iree_sparse_solver_module_t* module,
    iree_sparse_solver_module_state_t* state,
    iree_hal_buffer_view_t* device_buffer_view,
    int64_t n, int64_t nnz,
    iree_hal_buffer_view_t* row_ptr_view,
    iree_hal_buffer_view_t* col_idx_view,
    iree_vm_ref_t* out_handle) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.analyze not yet implemented");
}

IREE_VM_ABI_EXPORT(iree_sparse_solver_analyze,
                   iree_sparse_solver_module_state_t, rIIrr, r) {
  iree_sparse_solver_module_t* sparse_module = state->module;

  iree_hal_buffer_view_t* device_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &device_view));
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
      stack, sparse_module, state, device_view, n, nnz, row_ptr_view,
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

// factor: rr -> i
IREE_VM_ABI_EXPORT(iree_sparse_solver_factor,
                   iree_sparse_solver_module_state_t, rr, i) {
  rets->i0 = -1;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.factor not yet implemented");
}

// factor.f64: rr -> i
IREE_VM_ABI_EXPORT(iree_sparse_solver_factor_f64,
                   iree_sparse_solver_module_state_t, rr, i) {
  rets->i0 = -1;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.factor.f64 not yet implemented");
}

// solve: rrr -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve,
                   iree_sparse_solver_module_state_t, rrr, v) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.solve not yet implemented");
}

// solve.f64: rrr -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve_f64,
                   iree_sparse_solver_module_state_t, rrr, v) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.solve.f64 not yet implemented");
}

// solve.batched: rrrI -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve_batched,
                   iree_sparse_solver_module_state_t, rrrI, v) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.solve.batched not yet implemented");
}

// solve.batched.f64: rrrI -> v
IREE_VM_ABI_EXPORT(iree_sparse_solver_solve_batched_f64,
                   iree_sparse_solver_module_state_t, rrrI, v) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.solve.batched.f64 not yet implemented");
}

// get_factor_nnz: r -> I
IREE_VM_ABI_EXPORT(iree_sparse_solver_get_factor_nnz,
                   iree_sparse_solver_module_state_t, r, I) {
  rets->i0 = 0;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.get_factor_nnz not yet implemented");
}

// get_num_supernodes: r -> I
IREE_VM_ABI_EXPORT(iree_sparse_solver_get_num_supernodes,
                   iree_sparse_solver_module_state_t, r, I) {
  rets->i0 = 0;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sparse_solver.get_num_supernodes not yet implemented");
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

IREE_API_EXPORT iree_status_t iree_sparse_solver_module_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_sparse_solver_module_flags_t flags, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

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
