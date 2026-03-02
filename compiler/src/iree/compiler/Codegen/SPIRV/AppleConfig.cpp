// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AppleConfig.h - Apple CodeGen Configurations -----------------------===//
//
// This file contains CodeGen configurations for Apple GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

using llvm::APInt;
using llvm::APIntOps::GreatestCommonDivisor;

namespace mlir::iree_compiler::detail {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

/// Sets the CodeGen configuration for reductions on Apple GPUs.
///
/// The generic SPIR-V reduction config aggressively shrinks workgroup size
/// when there are enough parallel workgroups (parallelThreshold=256). This
/// is counterproductive on Apple GPUs where larger workgroup sizes (more
/// SIMD groups) provide better memory latency hiding through the hardware
/// scheduler.
///
/// This config uses the maximum feasible workgroup size without shrinking,
/// which means more SIMD groups per workgroup for better occupancy.
static LogicalResult setAppleReductionConfig(linalg::LinalgOp op,
                                             IREE::GPU::TargetAttr target) {
  if (!target.supportsSubgroupShuffle())
    return failure();

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  int64_t numParallelDims = op.getNumParallelLoops();

  if (reductionDims.empty())
    return failure();

  // Only handle static, innermost reduction dimensions.
  // Dynamic reductions fall back to the generic path.
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim]))
      return failure();
    if (dim < numParallelDims)
      return failure();
  }

  if (op.getRegionOutputArgs().size() != 1)
    return failure();

  // Only support projected permutation inputs.
  if (llvm::any_of(op.getDpsInputOperands(), [&](OpOperand *input) {
        return !op.getMatchingIndexingMap(input).isProjectedPermutation();
      }))
    return failure();

  // Only support single combiner reductions.
  bool foundSingleReductionOutput = false;
  for (int64_t i = 0, e = op.getDpsInits().size(); i < e; i++) {
    SmallVector<Operation *> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), i, combinerOps) &&
        combinerOps.size() == 1) {
      if (foundSingleReductionOutput)
        return failure();
      foundSingleReductionOutput = true;
      continue;
    }
    if (!op.getMatchingIndexingMap(op.getDpsInitOperand(i)).isIdentity())
      return failure();
  }
  if (!foundSingleReductionOutput)
    return failure();

  int subgroupSize = target.getPreferredSubgroupSize();

  // Tile all parallel dimensions to 1.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);

  int64_t reductionSize = 1;
  for (int64_t dim : reductionDims)
    reductionSize *= bounds[dim];
  if (reductionSize % subgroupSize != 0)
    return failure();

  const Type elementType =
      cast<ShapedType>(op.getDpsInits()[0].getType()).getElementType();
  if (!elementType.isIntOrFloat())
    return failure();
  unsigned bitWidth = IREE::Util::getTypeBitWidth(elementType);
  if (bitWidth != 32 && bitWidth != 16 && bitWidth != 8)
    return failure();

  // Let each thread handle `vectorSize` elements via vector loads.
  constexpr int kMaxVectorNumBits = 128;
  unsigned vectorSize = kMaxVectorNumBits / bitWidth;
  while ((reductionSize / vectorSize) % subgroupSize != 0)
    vectorSize /= 2;

  // Compute workgroup size without the aggressive shrinking used by the
  // generic path. Apple GPUs benefit from larger workgroup sizes for better
  // memory latency hiding through SIMD group interleaving.
  const int64_t maxWorkgroupSize =
      target.getWgp().getMaxThreadCountPerWorkgroup();
  int64_t groupSize = reductionSize / vectorSize;
  if (groupSize > maxWorkgroupSize) {
    groupSize = GreatestCommonDivisor(APInt(64, uint64_t(groupSize)),
                                      APInt(64, uint64_t(maxWorkgroupSize)))
                    .getZExtValue();
  }

  // The two-step butterfly reduction requires subgroup count <= subgroup size.
  if ((groupSize / subgroupSize) > subgroupSize)
    return failure();

  std::array<int64_t, 3> workgroupSize = {groupSize, 1, 1};

  // Compute reduction tile sizes matching the workgroup distribution.
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  int64_t remainingGroupSize = groupSize;
  for (int i = reductionDims.size() - 1; i >= 0; --i) {
    int64_t dim = reductionDims[i];
    int64_t bound = bounds[dim];
    if (i == static_cast<int>(reductionDims.size()) - 1)
      bound /= vectorSize;
    APInt size = GreatestCommonDivisor(APInt(64, uint64_t(remainingGroupSize)),
                                       APInt(64, uint64_t(bound)));
    reductionTileSizes[dim] = size.getSExtValue();
    if (i == static_cast<int>(reductionDims.size()) - 1)
      reductionTileSizes[dim] *= vectorSize;
    remainingGroupSize /= size.getSExtValue();
  }

  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes));
  tileSizes.emplace_back(std::move(reductionTileSizes));
  if (failed(setOpConfigAndEntryPointFnTranslation(
          op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
          CodeGenPipeline::SPIRVSubgroupReduce, workgroupSize))) {
    return failure();
  }

  // Set lowering configuration for other Linalg ops in the dispatch.
  op->getParentOfType<FunctionOpInterface>().walk([&](linalg::LinalgOp op) {
    setLoweringConfig(
        op, IREE::Codegen::LoweringConfigAttr::get(op.getContext(), tileSizes));
  });
  return success();
}

static LogicalResult setAppleMatmulConfig(linalg::LinalgOp op,
                                          IREE::GPU::TargetAttr target) {
  const std::array<int64_t, 2> workgroupXY = {256, 1};
  std::array<int64_t, 3> threadMNK;
  auto inputType = cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  if (IREE::Util::getTypeBitWidth(inputType.getElementType()) == 16) {
    threadMNK = {4, 8, 8};
  } else {
    threadMNK = {4, 4, 4};
  }
  return setMatmulOpConfig(target, op, workgroupXY, threadMNK);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setAppleCodeGenConfig(IREE::GPU::TargetAttr target,
                                    Operation *rootOp) {
  int subgroupSize = target.getPreferredSubgroupSize();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setAppleMatmulConfig(linalgOp, target);
  }

  if (auto convOp = dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
    // Use the result type in case of larger bitwidth for accumulators.
    auto type = cast<ShapedType>(convOp->getResult(0).getType());
    const int bitwidth = type.getElementTypeBitWidth();
    if (bitwidth > 32)
      return failure();
    const int multipler = 32 / bitwidth;
    const int bestTilingFactor = 16 * multipler;
    return setConvOpConfig(cast<linalg::LinalgOp>(rootOp), subgroupSize,
                           bestTilingFactor);
  }

  // Try Apple-optimized reduction config with larger workgroup sizes.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (succeeded(setAppleReductionConfig(linalgOp, target)))
      return success();
  }

  return failure();
}

} // namespace mlir::iree_compiler::detail
