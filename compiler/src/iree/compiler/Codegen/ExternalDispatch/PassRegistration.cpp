//===-- PassRegistration.cpp - External Dispatch Pass Registration --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalDispatch/Passes.h"

namespace mlir::iree_compiler {

void registerExternalDispatchPasses() {
  // Register all passes generated from tablegen.
  registerPasses();
}

}  // namespace mlir::iree_compiler
