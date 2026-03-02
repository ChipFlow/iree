// Copyright 2024 IREE Metal PJRT Plugin Contributors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/metal/client.h"

#import <Metal/Metal.h>

#include "iree/hal/drivers/metal/registration/driver_module.h"
#include "iree/hal/drivers/metal/shared_event.h"
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

  // Create a dedicated dispatch queue and listener for MTLSharedEvent
  // notifications. This avoids spawning a thread per event.
  listener_queue_ = dispatch_queue_create(
      "com.iree.pjrt.metal.event_listener", DISPATCH_QUEUE_SERIAL);
  event_listener_ =
      [[MTLSharedEventListener alloc] initWithDispatchQueue:listener_queue_];
}

MetalClientInstance::~MetalClientInstance() {
  if (event_listener_) {
    [event_listener_ release];
    event_listener_ = nil;
  }
  if (listener_queue_) {
    dispatch_release(listener_queue_);
    listener_queue_ = nil;
  }
}

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

EventInstance* MetalClientInstance::CreateEvent(
    iree::vm::ref<iree_hal_fence_t> fence) {
  if (!fence) {
    return new EventInstance(/*fence=*/nullptr);
  }

  // Create event without starting a wait thread — we'll use notifyListener.
  auto* event = new EventInstance(std::move(fence), /*start_thread=*/false);

  iree_hal_semaphore_list_t sems =
      iree_hal_fence_semaphore_list(event->fence());

  if (sems.count == 0) {
    // No semaphores to wait on — signal immediately.
    event->SignalReady(iree_ok_status());
    return event;
  }

  if (sems.count == 1) {
    // Fast path: single semaphore, register notifyListener directly.
    id<MTLSharedEvent> shared_event =
        iree_hal_metal_shared_event_handle(sems.semaphores[0]);
    uint64_t target_value = sems.payload_values[0];

    // Prevent event from being freed before the notification fires.
    // The block retains 'event' as a raw pointer; the caller retains
    // ownership. The PJRT_Event_Destroy handler uses OnReady to ensure
    // the event isn't deleted until it's signaled, so 'event' will be
    // alive when this block fires.
    [shared_event notifyListener:event_listener_
                         atValue:target_value
                           block:^(id<MTLSharedEvent> se, uint64_t v) {
                             if (v >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
                               event->SignalReady(iree_make_status(
                                   IREE_STATUS_ABORTED,
                                   "Metal shared event failed"));
                             } else {
                               event->SignalReady(iree_ok_status());
                             }
                           }];
    return event;
  }

  // Multi-semaphore path: use an atomic counter to track completions.
  // Shared state allocated on the heap; freed by the last block to fire.
  struct MultiWaitState {
    std::atomic<int32_t> remaining;
    std::atomic<bool> did_fail;
    EventInstance* event;
  };
  auto* state = new MultiWaitState{
      .remaining = static_cast<int32_t>(sems.count),
      .did_fail = false,
      .event = event,
  };

  for (iree_host_size_t i = 0; i < sems.count; ++i) {
    id<MTLSharedEvent> shared_event =
        iree_hal_metal_shared_event_handle(sems.semaphores[i]);
    uint64_t target_value = sems.payload_values[i];

    [shared_event notifyListener:event_listener_
                         atValue:target_value
                           block:^(id<MTLSharedEvent> se, uint64_t v) {
                             if (v >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
                               state->did_fail.store(true,
                                                     std::memory_order_relaxed);
                             }
                             int32_t prev =
                                 state->remaining.fetch_sub(
                                     1, std::memory_order_acq_rel);
                             if (prev == 1) {
                               // Last semaphore completed.
                               if (state->did_fail.load(
                                       std::memory_order_relaxed)) {
                                 state->event->SignalReady(iree_make_status(
                                     IREE_STATUS_ABORTED,
                                     "Metal shared event failed"));
                               } else {
                                 state->event->SignalReady(iree_ok_status());
                               }
                               delete state;
                             }
                           }];
  }

  return event;
}

}  // namespace iree::pjrt::metal
