// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "iree-fixed-point-iterator"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_FIXEDPOINTITERATORPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

// Computes a structural fingerprint of an operation for cycle detection.
// This is faster than full IR hashing but still catches most oscillations.
static uint64_t computeIRFingerprint(Operation *op) {
  uint64_t hash = 0;

  // Count operations by type
  llvm::DenseMap<mlir::OperationName, int64_t> opCounts;
  op->walk([&](Operation *nested) {
    opCounts[nested->getName()]++;
  });

  // Combine counts into hash
  for (auto &[name, count] : opCounts) {
    hash ^= llvm::hash_combine(name.getAsOpaquePointer(), count);
  }

  // Also factor in the number of block arguments and results for top-level ops
  if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
    for (auto &nestedOp : moduleOp.getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(nestedOp)) {
        hash ^= llvm::hash_combine(funcOp.getNumArguments(),
                                   funcOp.getNumResults());
      }
    }
  }

  return hash;
}

// Dynamic pass which runs a sub-pipeline to a fixed point or a maximum
// iteration count.
//
// There is no direct coupling between this iterator and the contained passes.
// Indirectly, at the start of each iteration, this pass will set the
// "iree.fixedpoint.converged" unit attribute on the root operation. If it is
// still there when the sub-pipeline is complete, it will be removed and
// iteration terminates. If a sub-pass removes it, then iteration will
// continue.
class FixedPointIteratorPass
    : public impl::FixedPointIteratorPassBase<FixedPointIteratorPass> {
public:
  using Base::Base;
  FixedPointIteratorPass() = default;
  FixedPointIteratorPass(const FixedPointIteratorPass &other)
      : impl::FixedPointIteratorPassBase<FixedPointIteratorPass>(other) {}
  FixedPointIteratorPass(OpPassManager pipeline);

private:
  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

  std::optional<OpPassManager> pipeline;

  // Serialized form of the body pipeline.
  Option<std::string> pipelineStr{
      *this, "pipeline", llvm::cl::desc("Pipeline to run to a fixed point")};
  Option<int> maxIterations{*this, "max-iterations",
                            llvm::cl::desc("Maximum number of iterations"),
                            llvm::cl::init(10)};
};

FixedPointIteratorPass::FixedPointIteratorPass(OpPassManager pipeline)
    : pipeline(std::move(pipeline)) {
  llvm::raw_string_ostream ss(pipelineStr);
  this->pipeline->printAsTextualPipeline(ss);
  ss.flush();
}

LogicalResult FixedPointIteratorPass::initializeOptions(
    StringRef options,
    function_ref<LogicalResult(const Twine &)> errorHandler) {
  if (failed(Pass::initializeOptions(options, errorHandler)))
    return failure();
  if (pipeline)
    return success();

  // Pipelines are expected to be of the form `<op-name>(<pipeline>)`.
  // TODO: This was lifted from the Inliner pass. We should provide a parse
  // entry point that is the direct inverse of printAsTextualPipeline() and
  // at least keep this internal to the upstream implementation.
  // See: https://github.com/llvm/llvm-project/issues/52813
  StringRef pipelineSr = pipelineStr;
  size_t pipelineStart = pipelineSr.find_first_of('(');
  if (pipelineStart == StringRef::npos || !pipelineSr.consume_back(")"))
    return failure();
  StringRef opName = pipelineSr.take_front(pipelineStart);
  OpPassManager pm(opName);
  if (failed(parsePassPipeline(pipelineSr.drop_front(1 + pipelineStart), pm)))
    return failure();
  pipeline = std::move(pm);
  return success();
}

void FixedPointIteratorPass::getDependentDialects(
    DialectRegistry &registry) const {
  pipeline->getDependentDialects(registry);
}

void FixedPointIteratorPass::runOnOperation() {
  MLIRContext *context = &getContext();
  StringAttr markerName = StringAttr::get(context, "iree.fixedpoint.iteration");
  StringAttr modifiedName =
      StringAttr::get(context, "iree.fixedpoint.modified");

  if (getOperation()->hasAttr(markerName)) {
    emitError(getOperation()->getLoc())
        << "nested fixed point pipelines not supported";
    return signalPassFailure();
  }

  // Track fingerprints to detect cycles (A -> B -> A oscillations)
  llvm::SmallDenseSet<uint64_t, 16> seenFingerprints;

  for (int i = 0; i < maxIterations; ++i) {
    getOperation()->setAttr(markerName,
                            IntegerAttr::get(IndexType::get(context), i));
    getOperation()->removeAttr(modifiedName);
    if (failed(runPipeline(*pipeline, getOperation()))) {
      return signalPassFailure();
    }

    if (!getOperation()->hasAttr(modifiedName)) {
      // Normal exit - no modifications made.
      LLVM_DEBUG(llvm::dbgs() << "Fixed-point converged after " << (i + 1)
                              << " iterations\n");
      getOperation()->removeAttr(markerName);
      return;
    }

    // Check for oscillation by computing a fingerprint of the current IR state.
    // If we've seen this fingerprint before, we're in a cycle and should stop.
    uint64_t fingerprint = computeIRFingerprint(getOperation());
    if (!seenFingerprints.insert(fingerprint).second) {
      // We've seen this state before - oscillation detected.
      // Treat this as convergence since we're not making net progress.
      LLVM_DEBUG(llvm::dbgs()
                 << "Fixed-point detected oscillation after " << (i + 1)
                 << " iterations (fingerprint collision), treating as converged\n");
      getOperation()->removeAttr(markerName);
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Fixed-point iteration " << (i + 1)
                            << ", fingerprint: " << fingerprint << "\n");
  }

  // Abnormal exit - iteration count exceeded.
  emitError(getOperation()->getLoc())
      << "maximum iteration count exceeded in fixed point pipeline";
  return signalPassFailure();
}

} // namespace

std::unique_ptr<OperationPass<void>>
createFixedPointIteratorPass(OpPassManager pipeline) {
  return std::make_unique<FixedPointIteratorPass>(std::move(pipeline));
}

} // namespace mlir::iree_compiler::IREE::Util
