// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Lowers stablehlo.cholesky to linalg operations.
//
// The Cholesky decomposition computes L such that A = L * L^T for a symmetric
// positive definite matrix A. This implementation uses the standard algorithm
// with column-by-column updates.
//
// For GPU efficiency, this could be replaced with calls to cuSOLVER/rocSOLVER
// or for sparse matrices, the sparse_solver module (BaSpaCho).
//===----------------------------------------------------------------------===//

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCHOLESKYTOLINALG
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Cholesky Lowering Pattern
//===----------------------------------------------------------------------===//

/// Lowers stablehlo.cholesky to a sequence of SCF loops with linalg operations.
///
/// The algorithm:
/// for j = 0 to n-1:
///   L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2 for k < j))
///   for i = j+1 to n-1:
///     L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k < j)) / L[j,j]
///
struct CholeskyToLinalgPattern
    : public OpRewritePattern<mlir::stablehlo::CholeskyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CholeskyOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());
    Type elementType = inputType.getElementType();

    // Only support 2D matrices for now.
    if (inputType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matrices supported");
    }

    // Only support float types.
    if (!isa<FloatType>(elementType)) {
      return rewriter.notifyMatchFailure(op, "only float types supported");
    }

    int64_t n = inputType.getDimSize(0);
    int64_t m = inputType.getDimSize(1);

    // Must be square.
    if (n != m || n == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(op, "matrix must be square and static");
    }

    // Only lower triangular is currently implemented.
    if (!op.getLower()) {
      return rewriter.notifyMatchFailure(op, "only lower triangular Cholesky supported");
    }

    ImplicitLocOpBuilder b(loc, rewriter);

    // Create zero constant.
    Value zero = b.create<arith::ConstantOp>(b.getZeroAttr(elementType));

    // Initialize output with zeros (we'll fill in the lower/upper triangle).
    Value init = b.create<tensor::EmptyOp>(inputType.getShape(), elementType);
    Value result = b.create<linalg::FillOp>(zero, init).getResult(0);

    // Loop bounds.
    Value nVal = b.create<arith::ConstantIndexOp>(n);
    Value zeroIdx = b.create<arith::ConstantIndexOp>(0);
    Value oneIdx = b.create<arith::ConstantIndexOp>(1);

    // Outer loop over columns (j).
    auto outerLoop = b.create<scf::ForOp>(
        zeroIdx, nVal, oneIdx, ValueRange{result},
        [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
          ImplicitLocOpBuilder b(loc, builder);
          Value L = iterArgs[0];

          // Compute L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2 for k < j))
          Value Ajj = b.create<tensor::ExtractOp>(input, ValueRange{j, j});

          // Sum of L[j,k]^2 for k < j
          // For lower triangular, L[j,k] is at position (j, k).
          auto sumLoop = b.create<scf::ForOp>(
              zeroIdx, j, oneIdx, ValueRange{zero},
              [&](OpBuilder &builder, Location loc, Value k, ValueRange sumArgs) {
                ImplicitLocOpBuilder b(loc, builder);
                Value sum = sumArgs[0];

                Value Ljk = b.create<tensor::ExtractOp>(L, ValueRange{j, k});
                Value Ljk2 = b.create<arith::MulFOp>(Ljk, Ljk);
                Value newSum = b.create<arith::AddFOp>(sum, Ljk2);
                b.create<scf::YieldOp>(ValueRange{newSum});
              });
          Value sumSq = sumLoop.getResult(0);

          Value diag = b.create<arith::SubFOp>(Ajj, sumSq);
          Value Ljj = b.create<math::SqrtOp>(diag);

          // Store L[j,j]
          Value L1 = b.create<tensor::InsertOp>(Ljj, L, ValueRange{j, j});

          // Inner loop over rows (i = j+1 to n-1)
          // Only pass the tensor as iter_arg; recompute j-based values inside.
          Value jPlusOne = b.create<arith::AddIOp>(j, oneIdx);
          // Compute j-1 as the column index for extracting the diagonal.
          // We store j in the tensor at an unused position temporarily, or
          // compute it from the loop bounds: j = innerLoop.lowerBound - 1.
          // Instead, just capture and use j directly - MLIR SSA should work.
          auto innerLoop = b.create<scf::ForOp>(
              jPlusOne, nVal, oneIdx, ValueRange{L1},
              [&, j, Ljj, input, zeroIdx, zero](OpBuilder &builder, Location loc, Value i, ValueRange innerArgs) {
                ImplicitLocOpBuilder b(loc, builder);
                Value Linner = innerArgs[0];

                // L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k < j)) / L[j,j]
                Value Aij = b.create<tensor::ExtractOp>(input, ValueRange{i, j});

                // Sum of L[i,k]*L[j,k] for k < j
                auto dotLoop = b.create<scf::ForOp>(
                    zeroIdx, j, oneIdx, ValueRange{zero},
                    [&, i, j, Linner](OpBuilder &builder, Location loc, Value k, ValueRange dotArgs) {
                      ImplicitLocOpBuilder b(loc, builder);
                      Value dotSum = dotArgs[0];

                      Value Lik = b.create<tensor::ExtractOp>(
                          Linner, ValueRange{i, k});
                      Value Ljk = b.create<tensor::ExtractOp>(
                          Linner, ValueRange{j, k});
                      Value prod = b.create<arith::MulFOp>(Lik, Ljk);
                      Value newDotSum = b.create<arith::AddFOp>(dotSum, prod);
                      b.create<scf::YieldOp>(ValueRange{newDotSum});
                    });
                Value dotProd = dotLoop.getResult(0);

                Value num = b.create<arith::SubFOp>(Aij, dotProd);
                Value Lij = b.create<arith::DivFOp>(num, Ljj);

                // Store L[i,j] for lower triangular (i > j)
                Value Lnew = b.create<tensor::InsertOp>(
                    Lij, Linner, ValueRange{i, j});
                b.create<scf::YieldOp>(ValueRange{Lnew});
              });

          b.create<scf::YieldOp>(ValueRange{innerLoop.getResult(0)});
        });

    rewriter.replaceOp(op, outerLoop.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LegalizeStableHLOCholeskyToLinalgPass
    : public impl::LegalizeStableHLOCholeskyToLinalgBase<
          LegalizeStableHLOCholeskyToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<CholeskyToLinalgPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<>> createLegalizeStableHLOCholeskyToLinalgPass() {
  return std::make_unique<LegalizeStableHLOCholeskyToLinalgPass>();
}

}  // namespace mlir::iree_compiler::stablehlo
