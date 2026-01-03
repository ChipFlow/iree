// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering CHLO ops to StableHLO and Shape dialect ops,
// taking care of CHLO's broadcasting semantics

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCUSTOMCALLS
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

// Computes householder using vector `v` and a `tau` matrice
// with the k-th element. See householder transformation as below:
// https://en.wikipedia.org/wiki/Householder_transformation
static Value computeHouseholder(Value v, Value tau, Value k,
                                ImplicitLocOpBuilder b) {
  auto vTy = cast<ShapedType>(v.getType());

  SmallVector<int64_t> hShape(vTy.getShape());
  hShape.push_back(hShape.back());

  auto hTy = RankedTensorType::get(hShape, vTy.getElementType());
  Value empty = tensor::EmptyOp::create(b, hShape, vTy.getElementType());

  auto outMap = b.getMultiDimIdentityMap(hShape.size());

  SmallVector<AffineExpr> exprs;
  for (int i = 0, s = vTy.getRank(); i < s; ++i) {
    exprs.push_back(b.getAffineDimExpr(i));
  }

  AffineMap vMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  exprs.back() = b.getAffineDimExpr(vTy.getRank());
  AffineMap vTMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  SmallVector<AffineMap> affineMaps = {vMap, vTMap, outMap};

  SmallVector<utils::IteratorType> iterTypes(hShape.size(),
                                             utils::IteratorType::parallel);

  Value zero =
      arith::ConstantOp::create(b, b.getZeroAttr(vTy.getElementType()));
  Value one =
      arith::ConstantOp::create(b, b.getFloatAttr(vTy.getElementType(), 1.0));

  return linalg::GenericOp::create(
             b, hTy, ValueRange{v, v}, empty, affineMaps, iterTypes,
             [&](OpBuilder &bb, Location loc, ValueRange args) {
               ImplicitLocOpBuilder b(loc, bb);
               SmallVector<Value> indices;
               for (int i = 0, s = hTy.getRank(); i < s; ++i) {
                 indices.push_back(linalg::IndexOp::create(b, loc, i));
               }

               SmallVector<Value> tauIndices(indices.begin(),
                                             indices.end() - 2);
               tauIndices.push_back(k);
               Value t = tensor::ExtractOp::create(b, tau, tauIndices);

               // Generates the lower triangularization of the matrix with
               // one values on the diagonal.
               auto tri = [&](Value v, Value i) {
                 Value eq =
                     arith::CmpIOp::create(b, arith::CmpIPredicate::eq, i, k);
                 Value lt =
                     arith::CmpIOp::create(b, arith::CmpIPredicate::ult, i, k);
                 Value sel = arith::SelectOp::create(b, eq, one, v);
                 return arith::SelectOp::create(b, lt, zero, sel);
               };

               Value v = tri(args[0], indices[indices.size() - 2]);
               Value vT = tri(args[1], indices[indices.size() - 1]);

               Value h = arith::MulFOp::create(b, v, vT);
               h = arith::MulFOp::create(b, h, t);

               Value isDiag = arith::CmpIOp::create(
                   b, arith::CmpIPredicate::eq, indices[indices.size() - 2],
                   indices[indices.size() - 1]);
               Value diag = arith::SelectOp::create(b, isDiag, one, zero);
               Value sub = arith::SubFOp::create(b, diag, h);

               linalg::YieldOp::create(b, sub);
             })
      .getResult(0);
}

// Slices the k-th column of matrix and computes the householder transformation
// for using the `tau` value.
static Value computeHouseholderSlice(Value matrix, Value tau, Value k,
                                     ImplicitLocOpBuilder b) {
  auto matrixTy = cast<ShapedType>(matrix.getType());
  int rank = matrixTy.getRank();

  SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
  SmallVector<int64_t> vShape(matrixTy.getShape());
  vShape[vShape.size() - 1] = 1;

  SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
  vOffsets[vOffsets.size() - 1] = k;

  SmallVector<OpFoldResult> vSizes;
  for (auto v : vShape) {
    vSizes.push_back(b.getIndexAttr(v));
  }

  auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
  Value v = tensor::ExtractSliceOp::create(b, sliceTy, matrix, vOffsets, vSizes,
                                           vStrides);

  SmallVector<ReassociationIndices> reass;
  for (int i = 0; i < rank - 2; ++i) {
    reass.push_back({i});
  }
  reass.push_back({rank - 2, rank - 1});

  ArrayRef<int64_t> collapseVShape(vShape.begin(), vShape.end() - 1);
  auto collapseVTy =
      RankedTensorType::get(collapseVShape, matrixTy.getElementType());
  Value collapseV = tensor::CollapseShapeOp::create(b, collapseVTy, v, reass);

  Value householder = computeHouseholder(collapseV, tau, k, b);
  return householder;
}

struct HouseholderReflectorRewriter final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;
  using OpAdaptor = mlir::stablehlo::CustomCallOp::Adaptor;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != "ProductOfElementaryHouseholderReflectors") {
      return rewriter.notifyMatchFailure(
          op, "not ProductOfElementaryHouseholderReflectors");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto matrix = op.getOperand(0);
    auto tau = op.getOperand(1);
    auto matrixTy = cast<ShapedType>(matrix.getType());
    auto tauTy = cast<ShapedType>(tau.getType());
    auto rank = matrixTy.getRank();

    if (isa<ComplexType>(matrixTy.getElementType())) {
      return rewriter.notifyMatchFailure(op, "complex types not supported");
    }

    if (rank < 2) {
      return rewriter.notifyMatchFailure(op, "requires minimum rank 2 matrix");
    }

    // Implementation needs to be checked to work with variable dimension
    // lengths. Should be relatively straightforward.
    if (!matrixTy.hasStaticShape() || !tauTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "not supported for dynamic shapes");
    }

    Value zero = arith::ConstantIndexOp::create(b, 0);
    Value one = arith::ConstantIndexOp::create(b, 1);
    Value k = arith::ConstantIndexOp::create(b, tauTy.getShape().back());
    Value householder0 = computeHouseholderSlice(matrix, tau, zero, b);
    auto scf = scf::ForOp::create(
        b, one, k, one, ValueRange{householder0},
        [&](OpBuilder &bb, Location loc, Value iv, ValueRange args) {
          ImplicitLocOpBuilder b(loc, bb);
          Value householder = computeHouseholderSlice(matrix, tau, iv, b);

          std::vector<int64_t> batch(rank - 2);
          for (int i = 0; i < rank - 2; ++i)
            batch[i] = i;
          std::vector<int64_t> lhsContract = {rank - 1};
          std::vector<int64_t> rhsContract = {rank - 2};

          auto dotNums = mlir::stablehlo::DotDimensionNumbersAttr::get(
              b.getContext(), batch, batch, lhsContract, rhsContract);
          Value dot = mlir::stablehlo::DotGeneralOp::create(
              b, householder0.getType(), args[0], householder, dotNums, nullptr,
              mlir::stablehlo::DotAlgorithmAttr{});
          scf::YieldOp::create(b, loc, dot);
        });

    SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
    SmallVector<int64_t> vShape(matrixTy.getShape());
    SmallVector<OpFoldResult> vSizes;
    for (auto v : vShape) {
      vSizes.push_back(b.getIndexAttr(v));
    }

    auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
    Value v = tensor::ExtractSliceOp::create(b, sliceTy, scf.getResult(0),
                                             vOffsets, vSizes, vStrides);

    rewriter.replaceOp(op, v);
    return success();
  }
};

struct ShapeAssertionDrop final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;
  using OpAdaptor = mlir::stablehlo::CustomCallOp::Adaptor;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != "shape_assertion") {
      return rewriter.notifyMatchFailure(op, "not shape_assertion");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LAPACK FFI Custom Calls - LU Factorization (getrf)
//===----------------------------------------------------------------------===//
// Implements LU factorization with partial pivoting using pure linalg/scf ops.
// This handles JAX's lapack_*getrf_ffi custom calls.

// Helper: Find the pivot row (row with max absolute value in column k, from k onwards)
static Value findPivotRow(Value matrix, Value k, int64_t m,
                          ImplicitLocOpBuilder &b) {
  auto matrixTy = cast<ShapedType>(matrix.getType());
  Type elemTy = matrixTy.getElementType();

  Value mIndex = arith::ConstantIndexOp::create(b, m);
  Value one = arith::ConstantIndexOp::create(b, 1);

  // Initialize: best_row = k, best_val = |A[k, k]|
  Value initVal = tensor::ExtractOp::create(b, matrix, ValueRange{k, k});
  Value initAbs = arith::SelectOp::create(
      b,
      arith::CmpFOp::create(b, arith::CmpFPredicate::OLT, initVal,
                            arith::ConstantOp::create(b, b.getZeroAttr(elemTy))),
      arith::NegFOp::create(b, initVal), initVal);

  // Loop from k+1 to m to find max
  Value kPlusOne = arith::AddIOp::create(b, k, one);

  auto forOp = scf::ForOp::create(
      b, kPlusOne, mIndex, one,
      ValueRange{k, initAbs},  // iter_args: (best_row, best_abs)
      [&](OpBuilder &bb, Location loc, Value i, ValueRange args) {
        ImplicitLocOpBuilder lb(loc, bb);
        Value bestRow = args[0];
        Value bestAbs = args[1];

        // Get |A[i, k]|
        Value val = tensor::ExtractOp::create(lb, matrix, ValueRange{i, k});
        Value absVal = arith::SelectOp::create(
            lb,
            arith::CmpFOp::create(lb, arith::CmpFPredicate::OLT, val,
                                  arith::ConstantOp::create(lb, lb.getZeroAttr(elemTy))),
            arith::NegFOp::create(lb, val), val);

        // Update if this is larger
        Value isLarger =
            arith::CmpFOp::create(lb, arith::CmpFPredicate::OGT, absVal, bestAbs);
        Value newBestRow = arith::SelectOp::create(lb, isLarger, i, bestRow);
        Value newBestAbs = arith::SelectOp::create(lb, isLarger, absVal, bestAbs);

        scf::YieldOp::create(lb, ValueRange{newBestRow, newBestAbs});
      });

  return forOp.getResult(0);  // Return best row index
}

// Helper: Swap rows i and j in matrix
static Value swapRows(Value matrix, Value i, Value j, int64_t n,
                      ImplicitLocOpBuilder &b) {
  auto matrixTy = cast<ShapedType>(matrix.getType());

  Value one = arith::ConstantIndexOp::create(b, 1);
  Value nIndex = arith::ConstantIndexOp::create(b, n);
  Value zero = arith::ConstantIndexOp::create(b, 0);

  // Check if i == j (no swap needed)
  Value needSwap = arith::CmpIOp::create(b, arith::CmpIPredicate::ne, i, j);

  // Create IfOp with result types and else region
  auto ifOp = scf::IfOp::create(b, TypeRange{matrixTy}, needSwap,
                                /*withElseRegion=*/true);

  // Build the then block
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    // Loop over columns and swap elements
    auto forOp = scf::ForOp::create(
        b, zero, nIndex, one, ValueRange{matrix},
        [&](OpBuilder &bbb, Location loc2, Value col, ValueRange args) {
          ImplicitLocOpBuilder llb(loc2, bbb);
          Value mat = args[0];

          // Extract A[i, col] and A[j, col]
          Value valI = tensor::ExtractOp::create(llb, mat, ValueRange{i, col});
          Value valJ = tensor::ExtractOp::create(llb, mat, ValueRange{j, col});

          // Insert swapped values
          Value mat1 = tensor::InsertOp::create(llb, valJ, mat, ValueRange{i, col});
          Value mat2 = tensor::InsertOp::create(llb, valI, mat1, ValueRange{j, col});

          scf::YieldOp::create(llb, mat2);
        });
    scf::YieldOp::create(b, forOp.getResult(0));
  }

  // Build the else block
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&ifOp.getElseRegion().front());
    scf::YieldOp::create(b, matrix);
  }

  return ifOp.getResult(0);
}

// Helper: Perform one step of LU factorization (column k)
static std::pair<Value, Value> luStep(Value matrix, Value pivots, Value k,
                                       int64_t m, int64_t n,
                                       ImplicitLocOpBuilder &b) {
  auto matrixTy = cast<ShapedType>(matrix.getType());
  (void)matrixTy;  // Used for type checking assertions

  Value one = arith::ConstantIndexOp::create(b, 1);
  Value mIndex = arith::ConstantIndexOp::create(b, m);
  Value nIndex = arith::ConstantIndexOp::create(b, n);

  // Find pivot row
  Value pivotRow = findPivotRow(matrix, k, m, b);

  // Swap rows k and pivotRow
  Value swapped = swapRows(matrix, k, pivotRow, n, b);

  // Store pivot index (convert to i32 for output, 0-indexed)
  Value pivotI32 = arith::IndexCastOp::create(b, b.getI32Type(), pivotRow);
  Value newPivots = tensor::InsertOp::create(b, pivotI32, pivots, ValueRange{k});

  // Get pivot element A[k, k]
  Value pivot = tensor::ExtractOp::create(b, swapped, ValueRange{k, k});

  // Compute multipliers and update submatrix
  Value kPlusOne = arith::AddIOp::create(b, k, one);

  // Loop over rows below k
  auto rowLoop = scf::ForOp::create(
      b, kPlusOne, mIndex, one, ValueRange{swapped},
      [&](OpBuilder &bb, Location loc, Value i, ValueRange args) {
        ImplicitLocOpBuilder lb(loc, bb);
        Value mat = args[0];

        // Compute multiplier: A[i,k] = A[i,k] / A[k,k]
        Value aik = tensor::ExtractOp::create(lb, mat, ValueRange{i, k});
        Value multiplier = arith::DivFOp::create(lb, aik, pivot);
        Value mat1 = tensor::InsertOp::create(lb, multiplier, mat, ValueRange{i, k});

        // Update row i: A[i,j] -= multiplier * A[k,j] for j > k
        auto colLoop = scf::ForOp::create(
            lb, kPlusOne, nIndex, one, ValueRange{mat1},
            [&](OpBuilder &bbb, Location loc2, Value j, ValueRange args2) {
              ImplicitLocOpBuilder llb(loc2, bbb);
              Value mat2 = args2[0];

              Value aij = tensor::ExtractOp::create(llb, mat2, ValueRange{i, j});
              Value akj = tensor::ExtractOp::create(llb, mat2, ValueRange{k, j});
              Value product = arith::MulFOp::create(llb, multiplier, akj);
              Value newAij = arith::SubFOp::create(llb, aij, product);
              Value mat3 = tensor::InsertOp::create(llb, newAij, mat2, ValueRange{i, j});

              scf::YieldOp::create(llb, mat3);
            });

        scf::YieldOp::create(lb, colLoop.getResult(0));
      });

  return {rowLoop.getResult(0), newPivots};
}

struct LapackGetrfFfiRewriter final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const final {
    StringRef target = op.getCallTargetName();

    // Match LAPACK getrf FFI calls (LU factorization).
    if (target != "lapack_sgetrf_ffi" && target != "lapack_dgetrf_ffi" &&
        target != "lapack_cgetrf_ffi" && target != "lapack_zgetrf_ffi") {
      return rewriter.notifyMatchFailure(op, "not a LAPACK getrf FFI call");
    }

    // Complex types not yet supported
    if (target == "lapack_cgetrf_ffi" || target == "lapack_zgetrf_ffi") {
      return rewriter.notifyMatchFailure(op, "complex LU not yet supported");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // JAX's getrf FFI has:
    // - Input: A matrix (to be factored in-place)
    // - Outputs: (LU matrix, pivots, info)
    if (op.getNumOperands() != 1) {
      return rewriter.notifyMatchFailure(op, "expected 1 operand");
    }
    if (op.getNumResults() != 3) {
      return rewriter.notifyMatchFailure(op, "expected 3 results");
    }

    Value inputMatrix = op.getOperand(0);
    auto matrixTy = cast<RankedTensorType>(inputMatrix.getType());
    auto pivotsTy = cast<RankedTensorType>(op.getResult(1).getType());
    auto infoTy = cast<RankedTensorType>(op.getResult(2).getType());

    // Only support 2D matrices for now
    if (matrixTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matrices supported");
    }

    // Only support static shapes for now
    if (!matrixTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");
    }

    int64_t m = matrixTy.getShape()[0];
    int64_t n = matrixTy.getShape()[1];
    int64_t minMN = std::min(m, n);

    // Initialize LU matrix as a copy of input
    Value lu = inputMatrix;

    // Initialize pivots tensor
    Value pivots = tensor::EmptyOp::create(b, pivotsTy.getShape(),
                                           pivotsTy.getElementType());

    // Main LU loop
    Value zero = arith::ConstantIndexOp::create(b, 0);
    Value one = arith::ConstantIndexOp::create(b, 1);
    Value minMNVal = arith::ConstantIndexOp::create(b, minMN);

    auto mainLoop = scf::ForOp::create(
        b, zero, minMNVal, one, ValueRange{lu, pivots},
        [&](OpBuilder &bb, Location loc, Value k, ValueRange args) {
          ImplicitLocOpBuilder lb(loc, bb);
          Value curLU = args[0];
          Value curPivots = args[1];

          auto [newLU, newPivots] = luStep(curLU, curPivots, k, m, n, lb);
          scf::YieldOp::create(lb, ValueRange{newLU, newPivots});
        });

    Value resultLU = mainLoop.getResult(0);
    Value resultPivots = mainLoop.getResult(1);

    // Info = 0 (success) - we don't check for singularity in this simple impl
    Value infoZero = arith::ConstantOp::create(
        b, DenseElementsAttr::get(infoTy, rewriter.getI32IntegerAttr(0)));

    rewriter.replaceOp(op, ValueRange{resultLU, resultPivots, infoZero});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition.
//===----------------------------------------------------------------------===//

struct LegalizeStableHLOCustomCalls final
    : impl::LegalizeStableHLOCustomCallsBase<LegalizeStableHLOCustomCalls> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    mlir::stablehlo::StablehloDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto f = getOperation();
    MLIRContext *ctx = f.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<HouseholderReflectorRewriter, ShapeAssertionDrop,
                 LapackGetrfFfiRewriter>(ctx);
    if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::stablehlo
