// Copyright 2024 IREE Metal PJRT Plugin Contributors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/metal/client.h"

#include "iree/hal/drivers/metal/registration/driver_module.h"

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

}  // namespace iree::pjrt::metal
