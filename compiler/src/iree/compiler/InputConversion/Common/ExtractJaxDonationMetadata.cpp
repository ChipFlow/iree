// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::InputConversion {

#define GEN_PASS_DEF_EXTRACTJAXDONATIONMETADATAPASS
#include "iree/compiler/InputConversion/Common/Passes.h.inc"

namespace {

class ExtractJaxDonationMetadataPass final
    : public impl::ExtractJaxDonationMetadataPassBase<
          ExtractJaxDonationMetadataPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    // Walk all functions in the module
    moduleOp.walk([&](FunctionOpInterface funcOp) {
      // Collect existing reflection attributes
      SmallVector<NamedAttribute> reflectionAttrs;
      if (auto existing =
              funcOp->getAttrOfType<DictionaryAttr>("iree.reflection")) {
        llvm::append_range(reflectionAttrs, existing.getValue());
      }

      // Scan function arguments for donation markers.
      // JAX uses tf.aliasing_output to indicate an argument's buffer can be
      // donated (reused for an output). We record these indices for runtime
      // validation.
      std::string donationIndices;
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        // Check for tf.aliasing_output (JAX's donation marker)
        if (funcOp.getArgAttr(i, "tf.aliasing_output")) {
          if (!donationIndices.empty())
            donationIndices += ",";
          donationIndices += std::to_string(i);
        }
        // Also check for jax.buffer_donor for compatibility
        else if (funcOp.getArgAttr(i, "jax.buffer_donor")) {
          if (!donationIndices.empty())
            donationIndices += ",";
          donationIndices += std::to_string(i);
        }
      }

      // If we found donated args, add to reflection metadata
      if (!donationIndices.empty()) {
        reflectionAttrs.push_back(NamedAttribute(
            StringAttr::get(ctx, "jax.donated_args"),
            StringAttr::get(ctx, donationIndices)));

        funcOp->setAttr("iree.reflection",
                        DictionaryAttr::get(ctx, reflectionAttrs));
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::InputConversion
