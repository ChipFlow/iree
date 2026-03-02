// Copyright 2024 IREE Metal PJRT Plugin Contributors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_METAL_CLIENT_H_
#define IREE_PJRT_PLUGIN_PJRT_METAL_CLIENT_H_

#include "iree/hal/drivers/metal/api.h"
#include "iree_pjrt/common/api_impl.h"

namespace iree::pjrt::metal {

class MetalClientInstance final : public ClientInstance {
 public:
  MetalClientInstance(std::unique_ptr<Platform> platform);
  ~MetalClientInstance();
  iree_status_t CreateDriver(iree_hal_driver_t** out_driver) override;
  bool SetDefaultCompilerFlags(CompilerJob* compiler_job) override;

  // Uses MTLSharedEvent notifyListener for efficient GPU event notification
  // instead of spawning a thread per event.
  EventInstance* CreateEvent(iree::vm::ref<iree_hal_fence_t> fence) override;

 protected:
  // Override to add dense_blas module for GPU-accelerated BLAS operations.
  iree_status_t PopulateVMModules(
      std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
      iree_hal_device_t* hal_device,
      iree::vm::ref<iree_vm_module_t>& main_module) override;

 private:
  // Opaque pointers to MTLSharedEventListener* and dispatch_queue_t.
  // Actual types used in client.mm (Objective-C++).
  void* event_listener_ = nullptr;
  void* listener_queue_ = nullptr;
};

}  // namespace iree::pjrt::metal

#endif  // IREE_PJRT_PLUGIN_PJRT_METAL_CLIENT_H_
