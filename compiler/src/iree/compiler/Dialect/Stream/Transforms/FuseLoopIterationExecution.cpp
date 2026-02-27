// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-fuse-loop-iteration-execution"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_FUSELOOPITERATIONEXECUTIONPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// Maximum number of iterations to unroll into a single execute region.
// Beyond this threshold, the IR size becomes prohibitive.
static constexpr int64_t kMaxUnrollIterations = 128;

// Analysis result for a fusible scf.for loop.
struct LoopFusionInfo {
  scf::ForOp forOp;
  int64_t tripCount;
  int64_t lowerBound;
  int64_t step;

  // The single async.execute op in the loop body.
  IREE::Stream::AsyncExecuteOp executeOp;

  // The timepoint.await op that consumes the execute's results.
  IREE::Stream::TimepointAwaitOp awaitOp;

  // Mapping: iter_arg index -> execute resource operand index.
  // Tracks which execute operands correspond to loop-carried values.
  SmallVector<std::pair<unsigned, unsigned>> iterArgToExecOperand;

  // Execute resource operand indices that are NOT iter_args (e.g., read-only
  // buffers captured from outside the loop).
  SmallVector<unsigned> nonIterArgExecOperandIndices;

  // Execute's await_timepoint, if from outside the loop.
  Value externalAwaitTimepoint;
};

// Analyzes an scf.for loop to determine if it can be fused.
// Returns std::nullopt if the loop cannot be fused.
static std::optional<LoopFusionInfo> analyzeForFusion(scf::ForOp forOp) {
  LoopFusionInfo info;
  info.forOp = forOp;

  // 1. Check constant bounds.
  auto lb = getConstantIntValue(forOp.getLowerBound());
  auto ub = getConstantIntValue(forOp.getUpperBound());
  auto step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0) {
    LLVM_DEBUG(llvm::dbgs() << "Loop bounds not constant\n");
    return std::nullopt;
  }

  info.lowerBound = *lb;
  info.step = *step;
  if (*ub <= *lb) {
    LLVM_DEBUG(llvm::dbgs() << "Upper bound <= lower bound, no iterations\n");
    return std::nullopt;
  }
  info.tripCount = (*ub - *lb + *step - 1) / *step;
  if (info.tripCount <= 1) {
    LLVM_DEBUG(llvm::dbgs() << "Trip count <= 1, no fusion needed\n");
    return std::nullopt;
  }
  if (info.tripCount > kMaxUnrollIterations) {
    LLVM_DEBUG(llvm::dbgs() << "Trip count " << info.tripCount
                            << " exceeds max unroll " << kMaxUnrollIterations
                            << "\n");
    return std::nullopt;
  }

  // 2. Find exactly one AsyncExecuteOp in the body.
  Block *body = forOp.getBody();
  IREE::Stream::AsyncExecuteOp execOp;
  for (auto &op : *body) {
    if (auto exec = dyn_cast<IREE::Stream::AsyncExecuteOp>(&op)) {
      if (execOp) {
        LLVM_DEBUG(llvm::dbgs() << "Multiple execute ops in loop body\n");
        return std::nullopt;
      }
      execOp = exec;
    }
  }
  if (!execOp) {
    LLVM_DEBUG(llvm::dbgs() << "No execute op in loop body\n");
    return std::nullopt;
  }
  info.executeOp = execOp;

  // 3. Find the TimepointAwaitOp that awaits the execute's results.
  Value timepoint = execOp.getResultTimepoint();
  IREE::Stream::TimepointAwaitOp awaitOp;
  for (auto *user : timepoint.getUsers()) {
    if (auto await = dyn_cast<IREE::Stream::TimepointAwaitOp>(user)) {
      if (awaitOp) {
        LLVM_DEBUG(llvm::dbgs() << "Multiple awaits on execute timepoint\n");
        return std::nullopt;
      }
      awaitOp = await;
    }
  }
  if (!awaitOp) {
    LLVM_DEBUG(llvm::dbgs() << "No await on execute timepoint\n");
    return std::nullopt;
  }
  // Verify the await is inside the loop body, not elsewhere.
  if (awaitOp->getParentOp() != forOp.getOperation()) {
    LLVM_DEBUG(llvm::dbgs() << "Await op is not in the loop body\n");
    return std::nullopt;
  }
  info.awaitOp = awaitOp;

  // 4. Check that all iter_args are stream resource types.
  for (auto iterArg : forOp.getRegionIterArgs()) {
    if (!isa<IREE::Stream::ResourceType>(iterArg.getType())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Non-resource iter_arg: " << iterArg.getType() << "\n");
      return std::nullopt;
    }
  }

  // 5. Check that scf.yield values are the awaited execute results.
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  for (auto yieldVal : yieldOp.getOperands()) {
    bool found = false;
    for (auto awaitResult : awaitOp.getResults()) {
      if (yieldVal == awaitResult) {
        found = true;
        break;
      }
    }
    if (!found) {
      LLVM_DEBUG(llvm::dbgs()
                 << "scf.yield operand is not an awaited execute result\n");
      return std::nullopt;
    }
  }

  // 6. Map iter_args to execute resource operands.
  for (auto [execOpIdx, execOperand] :
       llvm::enumerate(execOp.getResourceOperands())) {
    bool isIterArg = false;
    for (auto [iterArgIdx, iterArg] :
         llvm::enumerate(forOp.getRegionIterArgs())) {
      if (execOperand == iterArg) {
        info.iterArgToExecOperand.push_back({iterArgIdx, execOpIdx});
        isIterArg = true;
        break;
      }
    }
    if (!isIterArg) {
      info.nonIterArgExecOperandIndices.push_back(execOpIdx);
    }
  }

  // Verify all iter_args are captured by the execute.
  if (info.iterArgToExecOperand.size() != forOp.getNumRegionIterArgs()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Not all iter_args are captured by the execute ("
               << info.iterArgToExecOperand.size() << " vs "
               << forOp.getNumRegionIterArgs() << ")\n");
    return std::nullopt;
  }

  // Also verify the number of execute results matches the number of awaited
  // results and the number of iter_args (for the carry chain).
  if (execOp.getResults().size() != forOp.getNumRegionIterArgs()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Execute result count doesn't match iter_arg count\n");
    return std::nullopt;
  }

  // 7. Check execute's await timepoint (if any) is from outside the loop.
  if (execOp.getAwaitTimepoint()) {
    Value tp = execOp.getAwaitTimepoint();
    if (auto *defOp = tp.getDefiningOp()) {
      if (forOp.getBody()->findAncestorOpInBlock(*defOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Execute awaits a timepoint defined inside the loop\n");
        return std::nullopt;
      }
    }
    // Also check it's not an iter_arg (timepoint shouldn't be loop-carried).
    for (auto iterArg : forOp.getRegionIterArgs()) {
      if (tp == iterArg) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Execute awaits a loop-carried timepoint\n");
        return std::nullopt;
      }
    }
    info.externalAwaitTimepoint = tp;
  }

  // 8. Check that the execute has no tied operands (the fusion drops tied
  // operand semantics since it creates a new execute with untied results).
  if (auto tiedOp =
          dyn_cast<IREE::Util::TiedOpInterface>(execOp.getOperation())) {
    for (unsigned i = 0; i < execOp.getResults().size(); ++i) {
      if (tiedOp.getTiedResultOperandIndex(i).has_value()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Execute has tied operands, cannot fuse\n");
        return std::nullopt;
      }
    }
  }

  // 9. Check that non-execute ops in the body don't have side effects
  // that prevent fusion (except arith/index ops for offset computation).
  for (auto &op : *body) {
    if (&op == execOp.getOperation())
      continue;
    if (&op == awaitOp.getOperation())
      continue;
    if (isa<scf::YieldOp>(op))
      continue;
    // Allow pure operations (arith, index, etc.).
    if (isPure(&op))
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "Non-pure non-execute op in loop body: " << op << "\n");
    return std::nullopt;
  }

  // 10. Check that pure non-execute ops don't reference iter_args. Outer ops
  // are cloned with iter_args mapped to initial values, which is incorrect for
  // iterations k>0 if they actually use the carried values.
  for (auto &op : *body) {
    if (&op == execOp.getOperation() || &op == awaitOp.getOperation() ||
        isa<scf::YieldOp>(op))
      continue;
    for (auto operand : op.getOperands()) {
      for (auto iterArg : forOp.getRegionIterArgs()) {
        if (operand == iterArg) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Non-execute op references iter_arg: " << op << "\n");
          return std::nullopt;
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Loop is fusible with trip count "
                          << info.tripCount << "\n");
  return info;
}

// Maps from the yield operands of the original execute to the iter_arg indices.
// Returns the mapping from execute body yield operand index -> iter_arg index.
static SmallVector<unsigned>
getYieldToIterArgMapping(LoopFusionInfo &info) {
  auto yieldOp = cast<scf::YieldOp>(info.forOp.getBody()->getTerminator());
  SmallVector<unsigned> mapping(info.executeOp.getResults().size(), -1u);

  for (auto [yieldIdx, yieldVal] : llvm::enumerate(yieldOp.getOperands())) {
    // Find which await result this yield value comes from.
    for (auto [awaitIdx, awaitResult] :
         llvm::enumerate(info.awaitOp.getResults())) {
      if (yieldVal == awaitResult) {
        // awaitIdx corresponds to the execute result index, which in turn
        // corresponds to the execute body yield operand index.
        mapping[awaitIdx] = yieldIdx;
        break;
      }
    }
  }
  return mapping;
}

// Performs the loop fusion transformation.
static LogicalResult fuseForLoop(LoopFusionInfo &info) {
  scf::ForOp forOp = info.forOp;
  IREE::Stream::AsyncExecuteOp origExec = info.executeOp;
  Block *loopBody = forOp.getBody();
  Block &origExecBody = origExec.getBody().front();

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  // Build the mapping from execute body yield -> iter_arg index.
  auto yieldToIterArg = getYieldToIterArgMapping(info);

  // Build the fused execute's capture list.
  // Order: [iter_arg captures (carries)..., non-iter_arg captures...]
  SmallVector<Value> fusedOperands;
  SmallVector<Value> fusedOperandSizes;
  SmallVector<Type> fusedOperandTypes;

  // Map: original execute body block arg index -> fused execute body block arg
  // index.
  DenseMap<unsigned, unsigned> origToFusedArgMap;

  // Track which fused args are carries and their iter_arg index.
  // carries[i] = {fusedArgIdx, iterArgIdx}
  struct CarryInfo {
    unsigned fusedArgIdx;
    unsigned iterArgIdx;
    unsigned origExecArgIdx;
  };
  SmallVector<CarryInfo> carries;

  for (auto [iterArgIdx, execOpIdx] : info.iterArgToExecOperand) {
    Value initVal = forOp.getInitArgs()[iterArgIdx];
    fusedOperands.push_back(initVal);
    fusedOperandTypes.push_back(initVal.getType());

    // Get the size from the original execute's operand sizes.
    // The operand sizes array is parallel to the resource operands.
    fusedOperandSizes.push_back(
        origExec.getResourceOperandSizes()[execOpIdx]);

    unsigned fusedIdx = fusedOperands.size() - 1;
    origToFusedArgMap[execOpIdx] = fusedIdx;
    carries.push_back({fusedIdx, iterArgIdx, execOpIdx});
  }

  // Add non-iter_arg captures.
  for (unsigned execOpIdx : info.nonIterArgExecOperandIndices) {
    Value capture = origExec.getResourceOperands()[execOpIdx];
    fusedOperands.push_back(capture);
    fusedOperandTypes.push_back(capture.getType());
    fusedOperandSizes.push_back(
        origExec.getResourceOperandSizes()[execOpIdx]);
    origToFusedArgMap[execOpIdx] = fusedOperands.size() - 1;
  }

  // Result types and sizes match the original execute's.
  SmallVector<Type> resultTypes(origExec.getResults().getTypes());
  SmallVector<Value> resultSizes(origExec.getResultSizes());

  // Create the fused execute op.
  SmallVector<int64_t> tiedOperands; // empty - no tied operands
  auto fusedExec = IREE::Stream::AsyncExecuteOp::create(
      builder, loc, resultTypes, resultSizes, info.externalAwaitTimepoint,
      fusedOperands, fusedOperandSizes, tiedOperands);

  // Copy affinity from original execute if present.
  if (origExec.getAffinityAttr()) {
    fusedExec.setAffinityAttr(origExec.getAffinityAttr());
  }

  // Add entry block with arguments matching captures.
  auto &entryBlock = fusedExec.getBody().emplaceBlock();
  SmallVector<Location> argLocs(fusedOperandTypes.size(), loc);
  entryBlock.addArguments(fusedOperandTypes, argLocs);

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&entryBlock);

  // Initialize the current carry values to the fused execute's block args
  // for the carry captures.
  SmallVector<Value> currentCarries(carries.size());
  for (auto [i, carry] : llvm::enumerate(carries)) {
    currentCarries[i] = entryBlock.getArgument(carry.fusedArgIdx);
  }

  // Track the last iteration's body yield values (for the fused execute yield).
  SmallVector<Value> lastYieldValues;

  // Unroll loop iterations into the fused execute body.
  for (int64_t k = 0; k < info.tripCount; ++k) {
    LLVM_DEBUG(llvm::dbgs() << "Unrolling iteration " << k << "\n");

    // Insert outer ops BEFORE the fused execute so they dominate its body.
    builder.setInsertionPoint(fusedExec);

    // Create constant for this iteration's induction variable value.
    // Must match the original induction variable type (may be i32, i64,
    // or index depending on the frontend lowering).
    int64_t inductionVal = info.lowerBound + k * info.step;
    Value inductionConst;
    Type ivType = forOp.getInductionVar().getType();
    if (isa<IndexType>(ivType)) {
      inductionConst =
          arith::ConstantIndexOp::create(builder, loc, inductionVal);
    } else {
      inductionConst = arith::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(ivType, inductionVal));
    }

    // Clone non-execute loop body ops (placed before the fused execute).
    // These compute iteration-dependent values like offsets.
    IRMapping outerMapping;
    outerMapping.map(forOp.getInductionVar(), inductionConst);

    // Map iter_args for the outer ops. These are resource values that the
    // outer ops might reference (though typically outer ops only use scalars).
    // For k=0, use the initial values; for k>0, use the values from outside
    // the fused execute (the initials - outer ops shouldn't depend on carries
    // flowing through the execute, but we map them just in case).
    for (auto [iterArgIdx, initArg] :
         llvm::enumerate(forOp.getRegionIterArgs())) {
      outerMapping.map(initArg, forOp.getInitArgs()[iterArgIdx]);
    }

    for (auto &op : *loopBody) {
      if (&op == origExec.getOperation())
        continue;
      if (&op == info.awaitOp.getOperation())
        continue;
      if (isa<scf::YieldOp>(op))
        continue;
      builder.clone(op, outerMapping);
    }

    // Build the mapping for cloning execute body ops into the fused body.
    IRMapping innerMapping;

    // Map execute body block args.
    for (auto [origArgIdx, origArg] :
         llvm::enumerate(origExecBody.getArguments())) {
      auto it = origToFusedArgMap.find(origArgIdx);
      assert(it != origToFusedArgMap.end() && "unmapped execute body arg");
      unsigned fusedArgIdx = it->second;

      // Check if this is a carry arg.
      bool isCarry = false;
      for (auto [carryIdx, carry] : llvm::enumerate(carries)) {
        if (carry.origExecArgIdx == origArgIdx) {
          innerMapping.map(origArg, currentCarries[carryIdx]);
          isCarry = true;
          break;
        }
      }
      if (!isCarry) {
        // Non-carry capture: always maps to the same fused execute block arg.
        innerMapping.map(origArg, entryBlock.getArgument(fusedArgIdx));
      }
    }

    // Map values from outside the execute that are used inside (e.g., offsets).
    // The outerMapping contains original -> cloned for this iteration.
    for (auto &op : origExecBody) {
      for (auto operand : op.getOperands()) {
        // Skip values defined inside the execute body.
        if (operand.getParentRegion() == &origExec.getBody())
          continue;
        // Skip values already mapped (block args).
        if (innerMapping.contains(operand))
          continue;
        // Check if this value was cloned in the outer mapping.
        if (outerMapping.contains(operand)) {
          innerMapping.map(operand, outerMapping.lookup(operand));
        }
        // Otherwise, it's a value from outside the loop (constant, etc.) -
        // no remapping needed, the execute body can reference it directly.
      }
    }

    // Clone execute body ops into fused execute body.
    SmallVector<Value> iterationYieldValues;
    auto origYieldOp =
        cast<IREE::Stream::YieldOp>(origExecBody.getTerminator());

    for (auto &op : origExecBody) {
      if (isa<IREE::Stream::YieldOp>(op)) {
        // Collect only the remapped resource yield values for carry threading
        // (not the size operands).
        for (auto operand : origYieldOp.getResourceOperands()) {
          iterationYieldValues.push_back(
              innerMapping.lookupOrDefault(operand));
        }
        continue;
      }
      bodyBuilder.clone(op, innerMapping);
    }

    // Update carries for the next iteration.
    // The execute body's yield values become the next iteration's carry inputs.
    // Use the yieldToIterArg mapping to connect them properly.
    for (auto [yieldIdx, yieldVal] : llvm::enumerate(iterationYieldValues)) {
      unsigned iterArgIdx = yieldToIterArg[yieldIdx];
      if (iterArgIdx == -1u)
        continue;
      for (auto [carryIdx, carry] : llvm::enumerate(carries)) {
        if (carry.iterArgIdx == iterArgIdx) {
          currentCarries[carryIdx] = yieldVal;
          break;
        }
      }
    }

    lastYieldValues = iterationYieldValues;
  }

  // Create the yield for the fused execute body.
  // The yield values are the last iteration's results.
  IREE::Stream::YieldOp::create(bodyBuilder, loc, lastYieldValues, resultSizes);

  // Create timepoint.await ops for the fused execute's results.
  builder.setInsertionPointAfter(fusedExec);
  auto awaitOp = IREE::Stream::TimepointAwaitOp::create(
      builder, loc, fusedExec.getResults(), resultSizes,
      fusedExec.getResultTimepoint());

  // Replace the scf.for's results with the awaited fused execute results.
  // Need to match the scf.for result order with the awaited values.
  // The scf.for results correspond to iter_args, and the fused execute results
  // correspond to the original execute's results.
  //
  // Use yieldToIterArg mapping: execute result[i] -> iter_arg[yieldToIterArg[i]]
  // And scf.for result[j] corresponds to iter_arg[j].
  for (auto [execResultIdx, iterArgIdx] : llvm::enumerate(yieldToIterArg)) {
    if (iterArgIdx == -1u)
      continue;
    forOp.getResult(iterArgIdx)
        .replaceAllUsesWith(awaitOp.getResults()[execResultIdx]);
  }

  // Erase the original scf.for.
  forOp.erase();

  LLVM_DEBUG(llvm::dbgs() << "Fused " << info.tripCount
                          << " loop iterations into single execute\n");
  return success();
}

//===----------------------------------------------------------------------===//
// While-loop batching
//===----------------------------------------------------------------------===//

// Analysis result for a batchable scf.while loop.
struct WhileLoopBatchInfo {
  scf::WhileOp whileOp;
  IREE::Stream::AsyncExecuteOp executeOp;
  IREE::Stream::TimepointAwaitOp awaitOp;
  int64_t batchSize;

  // Indices of "after" region block args that are resource types flowing
  // through the execute.
  SmallVector<unsigned> resourceArgIndices;

  // Indices of "after" region block args that are scalars (not resources).
  SmallVector<unsigned> scalarArgIndices;
};

// Analyzes an scf.while loop to determine if it can be batched.
// Returns std::nullopt if the loop cannot be batched.
static std::optional<WhileLoopBatchInfo>
analyzeWhileFusion(scf::WhileOp whileOp, int64_t batchSize) {
  if (batchSize <= 1) {
    LLVM_DEBUG(llvm::dbgs() << "While batch size <= 1, skipping\n");
    return std::nullopt;
  }

  WhileLoopBatchInfo info;
  info.whileOp = whileOp;
  info.batchSize = batchSize;

  // 1. Check that the "before" region does NOT contain any async.execute ops.
  Block *beforeBody = whileOp.getBeforeBody();
  for (auto &op : *beforeBody) {
    if (isa<IREE::Stream::AsyncExecuteOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "While condition region contains async.execute\n");
      return std::nullopt;
    }
  }

  // 2. Find exactly one AsyncExecuteOp in the "after" (body) region.
  Block *afterBody = whileOp.getAfterBody();
  IREE::Stream::AsyncExecuteOp execOp;
  for (auto &op : *afterBody) {
    if (auto exec = dyn_cast<IREE::Stream::AsyncExecuteOp>(&op)) {
      if (execOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Multiple execute ops in while body\n");
        return std::nullopt;
      }
      execOp = exec;
    }
  }
  if (!execOp) {
    LLVM_DEBUG(llvm::dbgs() << "No execute op in while body\n");
    return std::nullopt;
  }
  info.executeOp = execOp;

  // 3. Find the TimepointAwaitOp in the after body.
  Value timepoint = execOp.getResultTimepoint();
  IREE::Stream::TimepointAwaitOp awaitOp;
  for (auto *user : timepoint.getUsers()) {
    if (auto await = dyn_cast<IREE::Stream::TimepointAwaitOp>(user)) {
      if (awaitOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Multiple awaits on execute timepoint in while body\n");
        return std::nullopt;
      }
      awaitOp = await;
    }
  }
  if (!awaitOp) {
    LLVM_DEBUG(llvm::dbgs() << "No await on execute timepoint in while body\n");
    return std::nullopt;
  }
  if (awaitOp->getParentOp() != whileOp.getOperation()) {
    LLVM_DEBUG(llvm::dbgs() << "Await is not in while body\n");
    return std::nullopt;
  }
  info.awaitOp = awaitOp;

  // 4. Check that the execute has no tied operands.
  if (auto tiedOp =
          dyn_cast<IREE::Util::TiedOpInterface>(execOp.getOperation())) {
    for (unsigned i = 0; i < execOp.getResults().size(); ++i) {
      if (tiedOp.getTiedResultOperandIndex(i).has_value()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "While body execute has tied operands\n");
        return std::nullopt;
      }
    }
  }

  // 5. Classify after-region block args into resource and scalar types.
  for (auto [idx, arg] : llvm::enumerate(afterBody->getArguments())) {
    if (isa<IREE::Stream::ResourceType>(arg.getType())) {
      info.resourceArgIndices.push_back(idx);
    } else {
      info.scalarArgIndices.push_back(idx);
    }
  }

  // Must have at least one resource arg (otherwise nothing to batch).
  if (info.resourceArgIndices.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No resource args in while body\n");
    return std::nullopt;
  }

  // 6. Check that non-execute/non-await ops in the body are pure.
  for (auto &op : *afterBody) {
    if (&op == execOp.getOperation())
      continue;
    if (&op == awaitOp.getOperation())
      continue;
    if (isa<scf::YieldOp>(op))
      continue;
    if (isPure(&op))
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "Non-pure non-execute op in while body: " << op << "\n");
    return std::nullopt;
  }

  // 7. Check that the yield operands are either awaited results (for resource
  // args) or unchanged block args / pure op results (for scalar args).
  auto yieldOp = cast<scf::YieldOp>(afterBody->getTerminator());
  for (auto [yieldIdx, yieldVal] : llvm::enumerate(yieldOp.getOperands())) {
    bool isAwaitResult = false;
    for (auto result : awaitOp.getResults()) {
      if (yieldVal == result) {
        isAwaitResult = true;
        break;
      }
    }
    if (isAwaitResult)
      continue;
    // For non-await yield operands: allow after-body block args passed through
    // unchanged, results of pure ops in the body, or values defined outside the
    // while loop (loop-invariant values like function arguments or constants).
    bool isAfterBlockArg = isa<BlockArgument>(yieldVal) &&
                           cast<BlockArgument>(yieldVal).getOwner() == afterBody;
    bool isPureResult =
        yieldVal.getDefiningOp() && isPure(yieldVal.getDefiningOp());
    bool isExternalValue =
        !isa<BlockArgument>(yieldVal)
            ? (yieldVal.getDefiningOp() &&
               !whileOp->isAncestor(yieldVal.getDefiningOp()))
            : (cast<BlockArgument>(yieldVal).getOwner() != afterBody &&
               cast<BlockArgument>(yieldVal).getOwner() !=
                   whileOp.getBeforeBody());
    if (!isAfterBlockArg && !isPureResult && !isExternalValue) {
      LLVM_DEBUG(llvm::dbgs()
                 << "While yield operand is neither await result, "
                    "passthrough, pure, nor external: "
                 << yieldVal << "\n");
      return std::nullopt;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "While loop is batchable with batch size "
                          << batchSize << "\n");
  return info;
}

// Batches a while loop by wrapping its body in an scf.for with K iterations.
// The existing for-loop fusion pass will then fuse the inner for-loop.
//
// The scf.while structure is:
//   scf.while (%before_args = %inits) : (init_types) -> (result_types)
//     // "before" region: block args match init_types
//     scf.condition(%cond) %forwarded_args : forwarded_types
//   } do {
//     // "after" region: block args match forwarded_types (from condition)
//     ... body ops ...
//     scf.yield %new_inits : init_types  // fed back to before region
//   }
//
// The inner for carries the after-body block args as iter_args:
//   } do {
//   ^bb0(%after_args: forwarded_types):
//     %batched = scf.for %i = 0 to K step 1
//         iter_args(%carry = %after_args) -> (forwarded_types) {
//       ... cloned body ops ...
//       scf.yield %new_carry : forwarded_types
//     }
//     scf.yield %batched : init_types  // outer yield, possibly different types
//   }
//
// The tricky part: the inner for's yield types must match its iter_arg types
// (= after-body block arg types = forwarded_types from condition), but the
// outer yield types must match init_types (= while loop operand types).
// These may differ if the condition forwards different values/types than
// what the yield sends back.
//
// For our supported pattern, the execute consumes resource after-args and
// produces resources. The yield sends back the awaited resources plus any
// scalars (which may be after-body args, external values, or pure op results).
// The inner for carries the after-body args and produces the same types for
// its yield, making the outer yield straightforward.
static LogicalResult batchWhileLoop(WhileLoopBatchInfo &info) {
  scf::WhileOp whileOp = info.whileOp;
  Block *afterBody = whileOp.getAfterBody();

  // Capture the outer yield and its operands before we start modifying.
  auto outerYield = cast<scf::YieldOp>(afterBody->getTerminator());

  // The inner for carries the after-body block args as iter_args.
  // These are the values forwarded from scf.condition into the body.
  SmallVector<Value> innerInitValues;
  for (auto arg : afterBody->getArguments())
    innerInitValues.push_back(arg);

  OpBuilder builder(afterBody, afterBody->begin());
  Location loc = whileOp.getLoc();

  // Create the inner scf.for: for %i = 0 to batchSize step 1.
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value cBatchSize =
      arith::ConstantIndexOp::create(builder, loc, info.batchSize);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

  // Note: scf::ForOp::build with initArgs but no bodyBuilder creates a block
  // WITHOUT a terminator. We provide a bodyBuilder that creates a placeholder
  // yield, which we'll replace after cloning the body ops.
  auto innerFor = scf::ForOp::create(
      builder, loc, c0, cBatchSize, c1, innerInitValues,
      [&](OpBuilder &b, Location l, Value /*iv*/, ValueRange iterArgs) {
        scf::YieldOp::create(b, l, iterArgs);
      });

  Block *innerBody = innerFor.getBody();

  // Map after-body block args -> inner for iter_args.
  IRMapping mapping;
  for (auto [afterArg, innerIterArg] : llvm::zip(afterBody->getArguments(),
                                                  innerFor.getRegionIterArgs()))
    mapping.map(afterArg, innerIterArg);

  // Collect ops to clone (everything except the yield and the constants/for
  // we just created).
  SmallVector<Operation *> opsToClone;
  for (auto &op : *afterBody) {
    if (isa<scf::YieldOp>(op))
      continue;
    // Skip ops we just created at the start of the block.
    if (&op == c0.getDefiningOp() || &op == cBatchSize.getDefiningOp() ||
        &op == c1.getDefiningOp() || &op == innerFor.getOperation())
      continue;
    opsToClone.push_back(&op);
  }

  // Clone into the inner for body (before its auto-generated yield).
  OpBuilder innerBuilder = OpBuilder::atBlockTerminator(innerBody);
  for (auto *op : opsToClone) {
    innerBuilder.clone(*op, mapping);
  }

  // Build the inner for's yield. The inner for's yield values feed back as
  // the next iteration's iter_args, which have the same types as the
  // after-body block args. We need to find, for each after-body arg, what
  // value the original outer yield would send for that position.
  //
  // The outer yield sends values back to the before-region (matching
  // whileOp.getInits() types). The condition then may forward a subset to
  // the after region. For a proper inner-for carry, we need the inner yield
  // to produce after-body-arg-typed values that represent the "next iteration"
  // state.
  //
  // Strategy: For each after-body block arg at index i, find which outer yield
  // operand corresponds to the same loop state, and remap it. This works when
  // the condition forwards a subset of before-args to the after region, and
  // the yield sends values back in the same order as the before-args.
  //
  // For the simple case we support: condition forwards before-args 0..N-1 to
  // after-body as args 0..N-1, and the yield has M operands where M >= N.
  // The inner for carries the N after-body args and its yield must produce
  // N values. We take the first N outer yield operands (remapped).
  unsigned numAfterArgs = afterBody->getNumArguments();
  SmallVector<Value> innerYieldValues;
  for (unsigned i = 0; i < numAfterArgs; ++i) {
    Value outerYieldVal = outerYield.getOperand(i);
    innerYieldValues.push_back(mapping.lookupOrDefault(outerYieldVal));
  }

  // Replace the inner for's auto-generated yield.
  auto existingInnerYield = cast<scf::YieldOp>(innerBody->getTerminator());
  innerBuilder.setInsertionPoint(existingInnerYield);
  scf::YieldOp::create(innerBuilder, loc, innerYieldValues);
  existingInnerYield.erase();

  // Update the outer yield: replace the first N operands with inner for
  // results, keep the rest (extra scalars going back to before-region).
  // First, update outer yield to use inner for results for the carried values.
  outerYield->setOperands(0, numAfterArgs, innerFor.getResults());

  // Now erase the original ops from the after body (in reverse order to
  // handle use-def chains). The outer yield no longer references them.
  for (auto it = opsToClone.rbegin(); it != opsToClone.rend(); ++it) {
    (*it)->erase();
  }

  LLVM_DEBUG(llvm::dbgs() << "Batched while loop with batch size "
                          << info.batchSize << "\n");
  return success();
}

//===----------------------------------------------------------------------===//
// --iree-stream-fuse-loop-iteration-execution
//===----------------------------------------------------------------------===//

struct FuseLoopIterationExecutionPass
    : public IREE::Stream::impl::FuseLoopIterationExecutionPassBase<
          FuseLoopIterationExecutionPass> {
  using IREE::Stream::impl::FuseLoopIterationExecutionPassBase<
      FuseLoopIterationExecutionPass>::FuseLoopIterationExecutionPassBase;

  void runOnOperation() override {
    mlir::CallableOpInterface parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    auto &region = *parentOp.getCallableRegion();

    // Phase 1: Batch scf.while loops by wrapping their body in an inner
    // scf.for. This creates fusible for-loops that Phase 2 will handle.
    SmallVector<scf::WhileOp> whileOps;
    region.walk([&](scf::WhileOp whileOp) { whileOps.push_back(whileOp); });

    for (auto whileOp : whileOps) {
      auto info = analyzeWhileFusion(whileOp, whileBatchSize);
      if (!info)
        continue;

      (void)batchWhileLoop(*info);
    }

    // Phase 2: Fuse scf.for loops (both original and batched inner loops).
    // Process inner-most loops first (post-order).
    SmallVector<scf::ForOp> forOps;
    region.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : forOps) {
      auto info = analyzeForFusion(forOp);
      if (!info)
        continue;

      (void)fuseForLoop(*info);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
