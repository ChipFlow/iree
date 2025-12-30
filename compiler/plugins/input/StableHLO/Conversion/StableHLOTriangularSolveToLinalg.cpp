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
// NOTE: We use util.optimization_barrier on the inner loop upper bound to
// prevent IREE's OptimizeIntArithmeticPass from incorrectly replacing loop
// indices with constants. This is a workaround for an issue where integer
// range analysis determines that small loop variables can only take one
// value, but replacing them with constants is incorrect for loop-carried
// tensor accesses. See the IREE bug report for details.
//
// For GPU efficiency, this could be replaced with calls to cuBLAS/rocBLAS
// or platform-specific BLAS libraries.
//===----------------------------------------------------------------------===//

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
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

/// Lowers stablehlo.triangular_solve to linalg.generic with scf.for.
///
/// Uses forward substitution with a row-by-row computation:
/// for i = 0 to n-1:
///   X[i,:] = (B[i,:] - A[i,0:i] @ X[0:i,:]) / A[i,i]
///
/// This lowering avoids the problematic loop-carried tensor update pattern
/// that IREE's stream conversion doesn't handle correctly. Instead, we:
/// 1. Create the result tensor once with all elements set to the initial values
/// 2. Use linalg.generic to compute each row, reading from the result tensor
///    for the dot product and writing to a fresh tensor for that row
/// 3. Update the result tensor row-by-row using tensor.insert_slice
///
/// For small matrices, we unroll the computation completely to avoid loops.
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

    // For small matrices, unroll completely to avoid loop-carried tensor issues.
    // This works around IREE's stream conversion bug with dynamic tensor updates.
    // We compute all scalars first, then create the tensor using tensor.from_elements.
    if (n <= 16) {
      // Compute all X[i,j] values as scalars first.
      // Store them in a 2D vector for later tensor creation.
      SmallVector<SmallVector<Value, 16>, 16> xValues(n);
      for (int64_t i = 0; i < n; ++i) {
        xValues[i].resize(m);
      }

      // Process each element in row-major order.
      for (int64_t i = 0; i < n; ++i) {
        Value iIdx = builder.create<arith::ConstantIndexOp>(i);

        // Get diagonal element A[i,i].
        Value Aii = builder.create<tensor::ExtractOp>(aMatrix, ValueRange{iIdx, iIdx});

        for (int64_t j = 0; j < m; ++j) {
          Value jIdx = builder.create<arith::ConstantIndexOp>(j);

          // Get B[i,j].
          Value Bij = builder.create<tensor::ExtractOp>(bMatrix, ValueRange{iIdx, jIdx});

          // Compute sum(A[i,k]*X[k,j] for k < i) using previously computed values.
          Value sum = zero;
          for (int64_t k = 0; k < i; ++k) {
            Value kIdx = builder.create<arith::ConstantIndexOp>(k);

            // Get A[i,k] or A[k,i] if transposed.
            Value Aik;
            if (transpose) {
              Aik = builder.create<tensor::ExtractOp>(aMatrix, ValueRange{kIdx, iIdx});
            } else {
              Aik = builder.create<tensor::ExtractOp>(aMatrix, ValueRange{iIdx, kIdx});
            }

            // Use the previously computed scalar value X[k,j].
            Value Xkj = xValues[k][j];
            Value prod = builder.create<arith::MulFOp>(Aik, Xkj);
            sum = builder.create<arith::AddFOp>(sum, prod);
          }

          // X[i,j] = (B[i,j] - sum) / A[i,i].
          Value num = builder.create<arith::SubFOp>(Bij, sum);
          Value Xij;
          if (unitDiagonal) {
            Xij = num;
          } else {
            Xij = builder.create<arith::DivFOp>(num, Aii);
          }

          xValues[i][j] = Xij;
        }
      }

      // Create the result tensor using linalg.generic with index-based selection.
      // This uses a parallel map operation that IREE can properly dispatch.
      Value init = builder.create<tensor::EmptyOp>(bType.getShape(), elementType);

      // Create affine maps for linalg.generic.
      AffineMap resultMap = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
      SmallVector<AffineMap> indexingMaps = {resultMap};
      SmallVector<utils::IteratorType> iteratorTypes = {
          utils::IteratorType::parallel, utils::IteratorType::parallel};

      // Create a linalg.generic that computes each element based on its indices.
      auto genericOp = builder.create<linalg::GenericOp>(
          TypeRange{bType}, ValueRange{}, ValueRange{init},
          indexingMaps, iteratorTypes,
          [&](OpBuilder &nestedBuilder, Location nestedLoc,
              ValueRange /*args*/) {
            // Get the current indices.
            Value rowIdx = nestedBuilder.create<linalg::IndexOp>(nestedLoc, 0);
            Value colIdx = nestedBuilder.create<linalg::IndexOp>(nestedLoc, 1);

            // Build a selection tree based on indices.
            // Start with the last value as the default.
            Value result = xValues[n - 1][m - 1];

            // Build selection tree in reverse order.
            for (int64_t i = n - 1; i >= 0; --i) {
              for (int64_t j = m - 1; j >= 0; --j) {
                if (i == n - 1 && j == m - 1) continue;  // Skip last (default)

                Value iConst = nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, i);
                Value jConst = nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, j);
                Value rowMatch = nestedBuilder.create<arith::CmpIOp>(
                    nestedLoc, arith::CmpIPredicate::eq, rowIdx, iConst);
                Value colMatch = nestedBuilder.create<arith::CmpIOp>(
                    nestedLoc, arith::CmpIPredicate::eq, colIdx, jConst);
                Value match = nestedBuilder.create<arith::AndIOp>(nestedLoc, rowMatch, colMatch);
                result = nestedBuilder.create<arith::SelectOp>(
                    nestedLoc, match, xValues[i][j], result);
              }
            }

            nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
          });

      rewriter.replaceOp(op, genericOp.getResult(0));
      return success();
    }

    // For larger matrices, we still need the loop-based approach.
    // TODO: Route to dense_blas.trsm for larger matrices.
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
                // Use optimization barrier on loop bound to prevent incorrect
                // loop index constant propagation (workaround for IREE bug).
                Value iBarrier =
                    cb.create<IREE::Util::OptimizationBarrierOp>(i).getResult(0);
                auto sumLoop = cb.create<scf::ForOp>(
                    zeroIdx, iBarrier, oneIdx, ValueRange{zero},
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
