//===-- LinalgMatmulToDenseBlas.cpp - Matmul to BLAS lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts linalg.matmul operations to dense_blas.gemm calls when
// they exceed a configurable size threshold.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalDispatch/Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LINALGMATMULTODENSEBLASPASS
#include "iree/compiler/Codegen/ExternalDispatch/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Matmul Analysis Utilities
//===----------------------------------------------------------------------===//

/// Returns true if the operation is a matmul or batch_matmul.
static bool isMatmulOp(Operation *op) {
  return isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op);
}

/// Get the M, N, K dimensions from a matmul operation.
/// Returns nullopt if the dimensions cannot be determined statically.
static std::optional<std::tuple<int64_t, int64_t, int64_t>>
getMatmulDimensions(linalg::LinalgOp op) {
  SmallVector<int64_t> dims = op.getStaticLoopRanges();
  if (dims.size() < 3) return std::nullopt;

  // For matmul: [M, N, K]
  // For batch_matmul: [B, M, N, K]
  int64_t offset = dims.size() - 3;
  int64_t M = dims[offset];
  int64_t N = dims[offset + 1];
  int64_t K = dims[offset + 2];

  // All dimensions must be static.
  if (M == ShapedType::kDynamic || N == ShapedType::kDynamic ||
      K == ShapedType::kDynamic) {
    return std::nullopt;
  }

  return std::make_tuple(M, N, K);
}

/// Check if the matmul exceeds the size threshold.
static bool exceedsSizeThreshold(linalg::LinalgOp op, int64_t threshold) {
  auto dims = getMatmulDimensions(op);
  if (!dims) return false;

  auto [M, N, K] = *dims;
  return (M * N * K) >= threshold;
}

//===----------------------------------------------------------------------===//
// Matmul to Dense BLAS Pattern
//===----------------------------------------------------------------------===//

/// Pattern that converts linalg.matmul to dense_blas.gemm.
struct MatmulToDenseBlasPattern : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToDenseBlasPattern(MLIRContext *context, int64_t threshold,
                            bool preferGpu, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        sizeThreshold(threshold), preferGpu(preferGpu) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this matmul exceeds the threshold.
    if (!exceedsSizeThreshold(op, sizeThreshold)) {
      return rewriter.notifyMatchFailure(op, "matmul below size threshold");
    }

    // Get operands.
    Value A = op.getInputs()[0];
    Value B = op.getInputs()[1];
    Value C = op.getOutputs()[0];

    // Get dimensions.
    auto dims = getMatmulDimensions(op);
    if (!dims) {
      return rewriter.notifyMatchFailure(op, "could not determine dimensions");
    }
    auto [M, N, K] = *dims;

    // Get element type.
    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    Type elementType = resultType.getElementType();

    // Only support f32 and f64 for now.
    if (!elementType.isF32() && !elementType.isF64()) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    Location loc = op.getLoc();

    // Create constants for GEMM parameters.
    // C = alpha * A * B + beta * C
    // For standard matmul: alpha = 1.0, beta = 0.0
    Value alpha = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0f), rewriter.getF32Type());
    Value beta = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0f), rewriter.getF32Type());

    // Create constants for transpose flags (no transpose for standard matmul).
    Value transA = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value transB = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    // Create dimension constants.
    Value mVal = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value nVal = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, K);

    // TODO: Create the actual dispatch to dense_blas.gemm.
    // This requires:
    // 1. Creating a flow.dispatch or hal.dispatch.extern operation
    // 2. Setting up the executable target for the dense_blas module
    // 3. Marshaling the tensor arguments correctly
    //
    // For now, we emit a marker attribute that can be processed later.
    auto newOp = rewriter.create<linalg::MatmulOp>(
        loc, resultType, ValueRange{A, B}, ValueRange{C});
    newOp->setAttr("dense_blas.dispatch", rewriter.getUnitAttr());
    newOp->setAttr("dense_blas.dimensions",
                   rewriter.getDenseI64ArrayAttr({M, N, K}));
    newOp->setAttr("dense_blas.prefer_gpu", rewriter.getBoolAttr(preferGpu));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }

private:
  int64_t sizeThreshold;
  bool preferGpu;
};

/// Pattern that converts linalg.batch_matmul to batched dense_blas.gemm.
struct BatchMatmulToDenseBlasPattern
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  BatchMatmulToDenseBlasPattern(MLIRContext *context, int64_t threshold,
                                 bool preferGpu, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::BatchMatmulOp>(context, benefit),
        sizeThreshold(threshold), preferGpu(preferGpu) {}

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this batch_matmul exceeds the threshold.
    if (!exceedsSizeThreshold(op, sizeThreshold)) {
      return rewriter.notifyMatchFailure(op, "batch_matmul below size threshold");
    }

    // Get dimensions.
    auto dims = getMatmulDimensions(op);
    if (!dims) {
      return rewriter.notifyMatchFailure(op, "could not determine dimensions");
    }
    auto [M, N, K] = *dims;

    // Get batch size from the first dimension.
    auto inputType = cast<RankedTensorType>(op.getInputs()[0].getType());
    int64_t batchSize = inputType.getShape()[0];
    if (batchSize == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(op, "dynamic batch size not supported");
    }

    // Get element type.
    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    Type elementType = resultType.getElementType();

    if (!elementType.isF32() && !elementType.isF64()) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    Location loc = op.getLoc();

    // Similar to MatmulToDenseBlasPattern, mark for later processing.
    auto newOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, resultType, op.getInputs(), op.getOutputs());
    newOp->setAttr("dense_blas.dispatch", rewriter.getUnitAttr());
    newOp->setAttr("dense_blas.dimensions",
                   rewriter.getDenseI64ArrayAttr({batchSize, M, N, K}));
    newOp->setAttr("dense_blas.prefer_gpu", rewriter.getBoolAttr(preferGpu));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }

private:
  int64_t sizeThreshold;
  bool preferGpu;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LinalgMatmulToDenseBlasPass
    : public impl::LinalgMatmulToDenseBlasPassBase<LinalgMatmulToDenseBlasPass> {
  using impl::LinalgMatmulToDenseBlasPassBase<
      LinalgMatmulToDenseBlasPass>::LinalgMatmulToDenseBlasPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<MatmulToDenseBlasPattern>(context, sizeThreshold, preferGpu);
    patterns.add<BatchMatmulToDenseBlasPattern>(context, sizeThreshold,
                                                 preferGpu);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgMatmulToDenseBlasPass(int64_t sizeThreshold, bool preferGpu) {
  LinalgMatmulToDenseBlasPassOptions options;
  options.sizeThreshold = sizeThreshold;
  options.preferGpu = preferGpu;
  return std::make_unique<LinalgMatmulToDenseBlasPass>(options);
}

}  // namespace mlir::iree_compiler
