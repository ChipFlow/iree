// Copyright 2024 IREE Metal PJRT Plugin Contributors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/metal/client.h"

#include "iree/hal/drivers/metal/registration/driver_module.h"
#include "iree/modules/hal/module.h"

#if IREE_DENSE_BLAS_HAVE_METAL
#include "iree/modules/dense_blas/module.h"
#endif

#if IREE_SPARSE_SOLVER_HAVE_METAL
#include "iree/modules/sparse_solver/module.h"
#endif

namespace iree::pjrt::metal {

MetalClientInstance::MetalClientInstance(std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  // Platform name must match how it's registered
  cached_platform_name_ = "iree_metal";
  IREE_CHECK_OK(iree_hal_metal_driver_module_register(driver_registry_));
}

MetalClientInstance::~MetalClientInstance() {}

iree_status_t MetalClientInstance::CreateDriver(
    iree_hal_driver_t** out_driver) {
  iree_string_view_t driver_name = iree_make_cstring_view("metal");
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
      driver_registry_, driver_name, host_allocator_, out_driver));
  logger().debug("Metal driver created");
  return iree_ok_status();
}

bool MetalClientInstance::SetDefaultCompilerFlags(CompilerJob* compiler_job) {
  // Use metal target for Apple Metal GPU
  return compiler_job->SetFlag("--iree-hal-target-device=metal");
}

iree_status_t MetalClientInstance::PopulateVMModules(
    std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
    iree_hal_device_t* hal_device,
    iree::vm::ref<iree_vm_module_t>& main_module) {
  // HAL module (required).
  modules.push_back({});
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      vm_instance(), iree_hal_module_device_policy_default(),
      /*device_count=*/1, &hal_device, IREE_HAL_MODULE_FLAG_NONE,
      iree_hal_module_debug_sink_stdio(stderr), host_allocator(),
      &modules.back()));

#if IREE_DENSE_BLAS_HAVE_METAL
  // Dense BLAS module for GPU-accelerated matrix operations.
  modules.push_back({});
  IREE_RETURN_IF_ERROR(iree_dense_blas_module_create(
      vm_instance(), hal_device, IREE_DENSE_BLAS_MODULE_FLAG_NONE,
      host_allocator(), &modules.back()));
  logger().debug("Dense BLAS module created for Metal backend");
#endif

#if IREE_SPARSE_SOLVER_HAVE_METAL
  // Sparse solver module for GPU-accelerated sparse Cholesky (BaSpaCho).
  modules.push_back({});
  IREE_RETURN_IF_ERROR(iree_sparse_solver_module_create(
      vm_instance(), hal_device, IREE_SPARSE_SOLVER_MODULE_FLAG_NONE,
      host_allocator(), &modules.back()));
  logger().debug("Sparse solver module created for Metal backend");
#endif

  // Main module (the user's compiled program).
  modules.push_back(main_module);
  return iree_ok_status();
}

}  // namespace iree::pjrt::metal
