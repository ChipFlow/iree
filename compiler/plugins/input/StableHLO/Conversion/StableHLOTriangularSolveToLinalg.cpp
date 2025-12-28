// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Lowers stablehlo.triangular_solve to linalg operations.
//
// Triangular solve computes X such that A*X=B (left_side=true) or X*A=B
// (left_side=false) where A is a triangular matrix.
//
// Currently supports forward substitution for:
// - lower triangular matrices (lower=true, transpose=NO_TRANSPOSE)
// - upper triangular with transpose (lower=false, transpose=TRANSPOSE)
//
// TODO: Back substitution for upper triangular matrices has a bug with
// lambda captures in the generated IR. Needs investigation.
//
// NOTE: There is a Metal backend bug with tensor<2x1xf32> specifically that
// causes incorrect results for 2x2 matrices with single RHS column. This is
// a Metal backend issue, not a problem with this pass. Larger matrices and
// multiple RHS columns work correctly.
//
// For GPU efficiency, this could be replaced with calls to cuBLAS/rocBLAS
// or platform-specific BLAS libraries.
//===----------------------------------------------------------------------===//

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOTRIANGULARSOLVETOLINALG
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// TriangularSolve Lowering Pattern
//===----------------------------------------------------------------------===//

/// Lowers stablehlo.triangular_solve to SCF loops with tensor operations.
///
/// Uses forward substitution:
/// for i = 0 to n-1:
///   X[i] = (B[i] - sum(A[i,k]*X[k] for k < i)) / A[i,i]
///
/// For transpose case, A[i,k] becomes A[k,i].
struct TriangularSolveToLinalgPattern
    : public OpRewritePattern<mlir::stablehlo::TriangularSolveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TriangularSolveOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMatrix = op.getA();
    Value bMatrix = op.getB();
    auto aType = cast<RankedTensorType>(aMatrix.getType());
    auto bType = cast<RankedTensorType>(bMatrix.getType());
    Type elementType = aType.getElementType();

    // Only support 2D matrices for now.
    if (aType.getRank() != 2 || bType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matrices supported");
    }

    // Only support float types.
    if (!isa<FloatType>(elementType)) {
      return rewriter.notifyMatchFailure(op, "only float types supported");
    }

    // Only support left_side = true for now.
    if (!op.getLeftSide()) {
      return rewriter.notifyMatchFailure(op, "only left_side=true supported");
    }

    auto transposeA = op.getTransposeA();
    if (transposeA != mlir::stablehlo::Transpose::NO_TRANSPOSE &&
        transposeA != mlir::stablehlo::Transpose::TRANSPOSE) {
      return rewriter.notifyMatchFailure(op, "only NO_TRANSPOSE or TRANSPOSE supported");
    }

    int64_t n = aType.getDimSize(0);
    int64_t m = bType.getDimSize(1);

    if (n == ShapedType::kDynamic || m == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(op, "only static shapes supported");
    }

    bool lower = op.getLower();
    bool transpose = (transposeA == mlir::stablehlo::Transpose::TRANSPOSE);
    bool unitDiagonal = op.getUnitDiagonal();

    // Determine if we can use forward substitution.
    // lower + no_transpose -> forward (rows 0 to n-1, sum k < i)
    // upper + transpose -> forward (A^T is lower, rows 0 to n-1, sum k < i)
    // Other cases need back substitution which is not yet supported.
    bool canUseForward = (lower != transpose);

    if (!canUseForward) {
      return rewriter.notifyMatchFailure(op,
          "back substitution (upper+no_transpose or lower+transpose) not yet supported");
    }

    ImplicitLocOpBuilder builder(loc, rewriter);

    // Create constants.
    Value zero = builder.create<arith::ConstantOp>(builder.getZeroAttr(elementType));
    Value nVal = builder.create<arith::ConstantIndexOp>(n);
    Value mVal = builder.create<arith::ConstantIndexOp>(m);
    Value zeroIdx = builder.create<arith::ConstantIndexOp>(0);
    Value oneIdx = builder.create<arith::ConstantIndexOp>(1);

    // Initialize output (X) with zeros.
    Value init = builder.create<tensor::EmptyOp>(bType.getShape(), elementType);
    Value result = builder.create<linalg::FillOp>(zero, init).getResult(0);

    // Forward substitution: iterate rows from 0 to n-1.
    auto rowLoop = builder.create<scf::ForOp>(
        zeroIdx, nVal, oneIdx, ValueRange{result},
        [&](OpBuilder &rowBuilder, Location rowLoc, Value i, ValueRange rowArgs) {
          ImplicitLocOpBuilder rb(rowLoc, rowBuilder);
          Value X = rowArgs[0];

          // Get diagonal element A[i,i].
          Value Aii = rb.create<tensor::ExtractOp>(aMatrix, ValueRange{i, i});

          // Loop over columns of B/X.
          auto colLoop = rb.create<scf::ForOp>(
              zeroIdx, mVal, oneIdx, ValueRange{X},
              [&, i, Aii, transpose, unitDiagonal, zero, zeroIdx, oneIdx, aMatrix, bMatrix](
                  OpBuilder &colBuilder, Location colLoc, Value j, ValueRange colArgs) {
                ImplicitLocOpBuilder cb(colLoc, colBuilder);
                Value Xcol = colArgs[0];

                // Get B[i,j].
                Value Bij = cb.create<tensor::ExtractOp>(bMatrix, ValueRange{i, j});

                // Compute sum(A[i,k]*X[k,j] for k < i).
                auto sumLoop = cb.create<scf::ForOp>(
                    zeroIdx, i, oneIdx, ValueRange{zero},
                    [&, i, transpose, aMatrix, Xcol](
                        OpBuilder &sumBuilder, Location sumLoc, Value k, ValueRange sumArgs) {
                      ImplicitLocOpBuilder sb(sumLoc, sumBuilder);
                      Value sum = sumArgs[0];

                      // Get A[i,k] or A^T[i,k] = A[k,i] if transposed.
                      Value Aik;
                      if (transpose) {
                        Aik = sb.create<tensor::ExtractOp>(aMatrix, ValueRange{k, i});
                      } else {
                        Aik = sb.create<tensor::ExtractOp>(aMatrix, ValueRange{i, k});
                      }
                      Value Xkj = sb.create<tensor::ExtractOp>(Xcol, ValueRange{k, j});
                      Value prod = sb.create<arith::MulFOp>(Aik, Xkj);
                      Value newSum = sb.create<arith::AddFOp>(sum, prod);
                      sb.create<scf::YieldOp>(ValueRange{newSum});
                    });
                Value dotProd = sumLoop.getResult(0);

                // X[i,j] = (B[i,j] - sum) / A[i,i].
                Value num = cb.create<arith::SubFOp>(Bij, dotProd);
                Value Xij;
                if (unitDiagonal) {
                  Xij = num;  // Diagonal is assumed to be 1.
                } else {
                  Xij = cb.create<arith::DivFOp>(num, Aii);
                }

                Value Xnew = cb.create<tensor::InsertOp>(Xij, Xcol, ValueRange{i, j});
                cb.create<scf::YieldOp>(ValueRange{Xnew});
              });

          rb.create<scf::YieldOp>(ValueRange{colLoop.getResult(0)});
        });

    rewriter.replaceOp(op, rowLoop.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LegalizeStableHLOTriangularSolveToLinalgPass
    : public impl::LegalizeStableHLOTriangularSolveToLinalgBase<
          LegalizeStableHLOTriangularSolveToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TriangularSolveToLinalgPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<>> createLegalizeStableHLOTriangularSolveToLinalgPass() {
  return std::make_unique<LegalizeStableHLOTriangularSolveToLinalgPass>();
}

}  // namespace mlir::iree_compiler::stablehlo
