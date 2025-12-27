// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/dense_blas/module.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module.h"

#if IREE_DENSE_BLAS_HAVE_METAL
#include "iree/modules/dense_blas/gemm_metal.h"
#endif

//===----------------------------------------------------------------------===//
// Module Version
//===----------------------------------------------------------------------===//

// Version 0.1: Initial GEMM export.
#define IREE_DENSE_BLAS_MODULE_VERSION_0_1 0x00000001u
#define IREE_DENSE_BLAS_MODULE_VERSION_LATEST IREE_DENSE_BLAS_MODULE_VERSION_0_1

//===----------------------------------------------------------------------===//
// Backend Detection
//===----------------------------------------------------------------------===//

typedef enum iree_dense_blas_backend_e {
  IREE_DENSE_BLAS_BACKEND_CPU = 0,      // Accelerate/OpenBLAS
  IREE_DENSE_BLAS_BACKEND_CUDA = 1,     // cuBLAS
  IREE_DENSE_BLAS_BACKEND_HIP = 2,      // rocBLAS
  IREE_DENSE_BLAS_BACKEND_METAL = 3,    // MPS
  IREE_DENSE_BLAS_BACKEND_OPENCL = 4,   // CLBlast
} iree_dense_blas_backend_t;

static iree_dense_blas_backend_t iree_dense_blas_detect_backend(
    iree_hal_device_t* device) {
  iree_string_view_t device_id = iree_hal_device_id(device);

  if (iree_string_view_starts_with(device_id, IREE_SV("cuda"))) {
    return IREE_DENSE_BLAS_BACKEND_CUDA;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("hip"))) {
    return IREE_DENSE_BLAS_BACKEND_HIP;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("metal"))) {
    return IREE_DENSE_BLAS_BACKEND_METAL;
  } else if (iree_string_view_starts_with(device_id, IREE_SV("vulkan")) ||
             iree_string_view_starts_with(device_id, IREE_SV("opencl"))) {
    return IREE_DENSE_BLAS_BACKEND_OPENCL;
  }
  return IREE_DENSE_BLAS_BACKEND_CPU;
}

//===----------------------------------------------------------------------===//
// Module State
//===----------------------------------------------------------------------===//

// Cast from base module to our module type.
#define IREE_DENSE_BLAS_MODULE_CAST(base_module) \
  ((iree_dense_blas_module_t*)((uint8_t*)(base_module) + \
                               iree_vm_native_module_size()))

typedef struct iree_dense_blas_module_t {
  iree_allocator_t host_allocator;
  iree_hal_device_t* device;
  iree_dense_blas_backend_t backend;
  iree_dense_blas_module_flags_t flags;
} iree_dense_blas_module_t;

typedef struct iree_dense_blas_module_state_t {
  iree_allocator_t host_allocator;
  iree_dense_blas_module_t* module;
} iree_dense_blas_module_state_t;

static void IREE_API_PTR iree_dense_blas_module_destroy(void* self) {
  iree_vm_module_t* base_module = (iree_vm_module_t*)self;
  iree_dense_blas_module_t* module = IREE_DENSE_BLAS_MODULE_CAST(base_module);
  iree_hal_device_release(module->device);
  // NOTE: The native module framework handles freeing the base_module memory.
  // We only release our own resources here.
}

static iree_status_t IREE_API_PTR iree_dense_blas_module_alloc_state(
    void* self, iree_allocator_t host_allocator,
    iree_vm_module_state_t** out_module_state) {
  iree_vm_module_t* base_module = (iree_vm_module_t*)self;
  iree_dense_blas_module_t* module = IREE_DENSE_BLAS_MODULE_CAST(base_module);
  iree_dense_blas_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->module = module;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR iree_dense_blas_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_dense_blas_module_state_t* state =
      (iree_dense_blas_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);
}

//===----------------------------------------------------------------------===//
// GEMM Export
//===----------------------------------------------------------------------===//
// Exported as: dense_blas.matmul
// Calling convention: rrr_v (3 buffer_view refs in, void out)
// Performs: C = A @ B (standard matrix multiplication)

// Implementation function called by the shim.
static iree_status_t iree_dense_blas_module_matmul_impl(
    iree_vm_stack_t* IREE_RESTRICT stack,
    iree_dense_blas_module_t* module,
    iree_dense_blas_module_state_t* state,
    iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs,
    iree_hal_buffer_view_t* out) {
  // Dispatch to backend.
  switch (module->backend) {
#if IREE_DENSE_BLAS_HAVE_METAL
    case IREE_DENSE_BLAS_BACKEND_METAL:
      return iree_dense_blas_gemm_metal(
          module->device, lhs, rhs, out,
          /*alpha=*/1.0f, /*beta=*/0.0f,
          /*transpose_lhs=*/false, /*transpose_rhs=*/false,
          module->host_allocator);
#endif

    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "GEMM not implemented for backend %d",
                              module->backend);
  }
}

// Target function for VM ABI.
// Using the IREE_VM_ABI_EXPORT pattern.
IREE_VM_ABI_EXPORT(iree_dense_blas_module_matmul,
                   iree_dense_blas_module_state_t, rrr, v) {
  // Cast module from void* to our module type.
  iree_dense_blas_module_t* dense_blas_module = state->module;

  // Dereference buffer view refs.
  iree_hal_buffer_view_t* lhs = NULL;
  iree_hal_buffer_view_t* rhs = NULL;
  iree_hal_buffer_view_t* out = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r0, &lhs));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &rhs));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r2, &out));

  return iree_dense_blas_module_matmul_impl(stack, dense_blas_module, state,
                                            lhs, rhs, out);
}

//===----------------------------------------------------------------------===//
// VM Module Interface
//===----------------------------------------------------------------------===//

// rrr_v shim is now defined in shims.c (shared with sparse_solver)

// Module function table.
static const iree_vm_native_function_ptr_t iree_dense_blas_module_funcs_[] = {
    {
        .shim = (iree_vm_native_function_shim_t)iree_vm_shim_rrr_v,
        .target = (iree_vm_native_function_target_t)iree_dense_blas_module_matmul,
    },
};

// Module exports.
static const iree_vm_native_export_descriptor_t
    iree_dense_blas_module_exports_[] = {
        {
            .local_name = iree_string_view_literal("matmul"),
            .calling_convention = iree_string_view_literal("0rrr_v"),
            .attr_count = 0,
            .attrs = NULL,
        },
};

static_assert(IREE_ARRAYSIZE(iree_dense_blas_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_dense_blas_module_exports_),
              "function pointer table must be 1:1 with exports");

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t
    iree_dense_blas_module_imports_[1];

static const iree_vm_native_module_descriptor_t
    iree_dense_blas_module_descriptor_ = {
        .name = iree_string_view_literal("dense_blas"),
        .version = IREE_DENSE_BLAS_MODULE_VERSION_LATEST,
        .attr_count = 0,
        .attrs = NULL,
        .dependency_count = 0,
        .dependencies = NULL,
        .import_count = 0,
        .imports = iree_dense_blas_module_imports_,
        .export_count = IREE_ARRAYSIZE(iree_dense_blas_module_exports_),
        .exports = iree_dense_blas_module_exports_,
        .function_count = IREE_ARRAYSIZE(iree_dense_blas_module_funcs_),
        .functions = iree_dense_blas_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_dense_blas_module_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_dense_blas_module_flags_t flags, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves.
  static const iree_vm_module_t interface = {
      .destroy = iree_dense_blas_module_destroy,
      .alloc_state = iree_dense_blas_module_alloc_state,
      .free_state = iree_dense_blas_module_free_state,
  };

  // Allocate shared module state.
  iree_host_size_t total_size = iree_vm_native_module_size() +
                                sizeof(iree_dense_blas_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);

  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_dense_blas_module_descriptor_,
      instance, host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_dense_blas_module_t* module = IREE_DENSE_BLAS_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  module->device = device;
  module->flags = flags;
  iree_hal_device_retain(device);

  // Detect backend based on device type.
  module->backend = iree_dense_blas_detect_backend(device);

  *out_module = base_module;
  return iree_ok_status();
}
