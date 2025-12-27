//===-- SparseCholeskySolver.cpp - Sparse Cholesky lowering -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass detects sparse Cholesky factorization patterns and lowers them
// to sparse_solver module calls (backed by BaSpaCho).
//
// Patterns detected:
// 1. JAX's scipy.linalg.cho_factor equivalent
// 2. Sparse matrix factorization via linalg ops with sparse tensors
// 3. Custom sparse solve patterns
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalDispatch/Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Sparse Cholesky Detection Utilities
//===----------------------------------------------------------------------===//

namespace {

/// Check if a tensor type has sparse encoding.
static bool isSparseType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getEncoding() != nullptr;
  }
  return false;
}

/// Check if an operation is a symmetric positive definite solve pattern.
/// This looks for patterns like: solve(A @ A.T + regularization, b)
static bool isSymmetricSolvePattern(Operation* op) {
  // Look for linalg.generic or func.call patterns that indicate
  // a symmetric solve operation.

  // Pattern 1: Direct sparse solve call
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    StringRef callee = callOp.getCallee();
    if (callee.contains("cho_factor") || callee.contains("cholesky") ||
        callee.contains("sparse_solve") || callee.contains("spd_solve")) {
      return true;
    }
  }

  return false;
}

/// Get sparse tensor info (dimensions, nnz estimation).
struct SparseTensorInfo {
  int64_t rows;
  int64_t cols;
  int64_t estimatedNnz;
  bool isSymmetric;
  bool isPositiveDefinite;
};

static std::optional<SparseTensorInfo> analyzeSparseTensor(Value tensor) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType || tensorType.getRank() != 2) {
    return std::nullopt;
  }

  SparseTensorInfo info;
  info.rows = tensorType.getDimSize(0);
  info.cols = tensorType.getDimSize(1);

  // Check if dimensions are static
  if (info.rows == ShapedType::kDynamic ||
      info.cols == ShapedType::kDynamic) {
    return std::nullopt;
  }

  // Estimate nnz based on sparse encoding or assume dense
  if (auto encoding = tensorType.getEncoding()) {
    // Sparse tensor - estimate based on typical sparsity
    // For SLAM/bundle adjustment, typical fill is 0.1-1%
    info.estimatedNnz = (info.rows * info.cols) / 100;
  } else {
    // Dense tensor - nnz = n * n
    info.estimatedNnz = info.rows * info.cols;
  }

  // Check symmetry from context (conservative: assume symmetric if square)
  info.isSymmetric = (info.rows == info.cols);
  info.isPositiveDefinite = false;  // Would need dataflow analysis

  return info;
}

//===----------------------------------------------------------------------===//
// Sparse Cholesky Lowering Pattern
//===----------------------------------------------------------------------===//

/// Pattern that detects sparse Cholesky and marks for external dispatch.
struct SparseCholeskyPattern : public OpRewritePattern<func::CallOp> {
  SparseCholeskyPattern(MLIRContext* context, int64_t threshold,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<func::CallOp>(context, benefit),
        sizeThreshold(threshold) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter& rewriter) const override {
    StringRef callee = op.getCallee();

    // Check for Cholesky-related function names
    bool isCholeskyOp = callee.contains("cho_factor") ||
                        callee.contains("cholesky") ||
                        callee.contains("potrf");
    bool isSolveOp = callee.contains("cho_solve") ||
                     callee.contains("solve_triangular") ||
                     callee.contains("trsm");

    if (!isCholeskyOp && !isSolveOp) {
      return rewriter.notifyMatchFailure(op, "not a Cholesky operation");
    }

    // Get input matrix
    if (op.getOperands().empty()) {
      return rewriter.notifyMatchFailure(op, "no operands");
    }

    Value matrix = op.getOperands()[0];
    auto info = analyzeSparseTensor(matrix);
    if (!info) {
      return rewriter.notifyMatchFailure(op, "could not analyze tensor");
    }

    // Check if matrix is large enough for external dispatch
    if (info->rows * info->cols < sizeThreshold) {
      return rewriter.notifyMatchFailure(op, "matrix below size threshold");
    }

    // Mark for sparse solver dispatch
    Location loc = op.getLoc();

    // Add attributes to indicate sparse solver dispatch
    op->setAttr("sparse_solver.dispatch", rewriter.getUnitAttr());
    op->setAttr("sparse_solver.operation",
                rewriter.getStringAttr(isCholeskyOp ? "factor" : "solve"));
    op->setAttr("sparse_solver.dimensions",
                rewriter.getDenseI64ArrayAttr({info->rows, info->cols}));
    op->setAttr("sparse_solver.estimated_nnz",
                rewriter.getI64IntegerAttr(info->estimatedNnz));

    // The actual lowering to sparse_solver module calls will happen
    // in a later pass that handles the marked operations.
    return success();
  }

private:
  int64_t sizeThreshold;
};

/// Pattern for detecting sparse.convert operations that indicate
/// sparse matrix construction for solving.
struct SparseConvertPattern
    : public OpRewritePattern<sparse_tensor::ConvertOp> {
  SparseConvertPattern(MLIRContext* context, PatternBenefit benefit = 1)
      : OpRewritePattern<sparse_tensor::ConvertOp>(context, benefit) {}

  LogicalResult matchAndRewrite(sparse_tensor::ConvertOp op,
                                PatternRewriter& rewriter) const override {
    // Check if this sparse tensor is used in a solve pattern
    Value result = op.getResult();

    bool usedInSolve = false;
    for (Operation* user : result.getUsers()) {
      if (isSymmetricSolvePattern(user)) {
        usedInSolve = true;
        break;
      }
    }

    if (!usedInSolve) {
      return rewriter.notifyMatchFailure(op, "not used in solve pattern");
    }

    // Mark the sparse tensor for BaSpaCho format conversion
    op->setAttr("sparse_solver.input_tensor", rewriter.getUnitAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct SparseCholeskySolverPass
    : public PassWrapper<SparseCholeskySolverPass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseCholeskySolverPass)

  SparseCholeskySolverPass() = default;
  SparseCholeskySolverPass(int64_t threshold) : sizeThreshold(threshold) {}

  StringRef getArgument() const override {
    return "iree-sparse-cholesky-to-solver";
  }

  StringRef getDescription() const override {
    return "Lower sparse Cholesky patterns to sparse_solver module calls";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<sparse_tensor::SparseTensorDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<SparseCholeskyPattern>(context, sizeThreshold);
    patterns.add<SparseConvertPattern>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

private:
  int64_t sizeThreshold = 1000;  // Default: 1000x1000 matrices
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createSparseCholeskySolverPass(int64_t sizeThreshold) {
  return std::make_unique<SparseCholeskySolverPass>(sizeThreshold);
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace {
static PassRegistration<SparseCholeskySolverPass> pass([] {
  return std::make_unique<SparseCholeskySolverPass>();
});
}  // namespace

}  // namespace mlir::iree_compiler
