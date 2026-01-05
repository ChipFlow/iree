// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverDialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverOps.h"
#include "iree/compiler/Dialect/SparseSolver/sparse_solver.imports.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

// Include generated dialect definitions.
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverDialect.cpp.inc"

namespace mlir::iree_compiler::IREE::SparseSolver {

namespace {

// Forward declarations for the conversion patterns population functions.
void populateSparseSolverToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns);

void populateStreamToSparseSolverPatterns(MLIRContext *context,
                                           ConversionTarget &target,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

class SparseSolverToVMConversionInterface : public VMConversionDialectInterface {
public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(iree_sparse_solver_imports_create()->data,
                  iree_sparse_solver_imports_create()->size),
        getDialect()->getContext());
  }

  void
  populateVMConversionPatterns(SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) const override {
    // Mark all SparseSolver ops as illegal - they must be converted to VM calls.
    // The HAL-level ops (spsolve_complete, etc.) are converted via
    // VMImportOpConversion to vm.call ops.
    conversionTarget.addIllegalDialect<IREE::SparseSolver::SparseSolverDialect>();
    populateSparseSolverToVMPatterns(getDialect()->getContext(), importSymbols,
                                      typeConverter, patterns);
  }
};

// Interface for Stream→HAL conversion.
// Converts tensor-level SparseSolver ops to HAL-level ops with buffer_views.
class StreamToSparseSolverConversionInterface
    : public HALConversionDialectInterface {
public:
  using HALConversionDialectInterface::HALConversionDialectInterface;

  void setupConversionTarget(ConversionTarget &target,
                             RewritePatternSet &patterns,
                             TypeConverter &typeConverter) const override {
    populateStreamToSparseSolverPatterns(getDialect()->getContext(), target,
                                          typeConverter, patterns);
  }
};

} // namespace

void SparseSolverDialect::initialize() {
  addInterfaces<SparseSolverToVMConversionInterface>();
  addInterfaces<StreamToSparseSolverConversionInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SparseSolver to VM Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

// Pattern to convert sparse_solver.spsolve (with buffer_views) to VM calls.
// At this point, the op has buffer_view operands from HAL conversion.
// We convert directly to vm.call @sparse_solver.spsolve_complete.
class SpsolveOpConversion : public OpConversionPattern<SpsolveOp> {
public:
  SpsolveOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                      TypeConverter &typeConverter)
      : OpConversionPattern<SpsolveOp>(typeConverter, context),
        importSymbols(importSymbols) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(
        "sparse_solver.spsolve_complete");
  }

  LogicalResult
  matchAndRewrite(SpsolveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!importOp) {
      return rewriter.notifyMatchFailure(
          op, "sparse_solver.spsolve_complete import not found");
    }

    auto loc = op.getLoc();

    // At this point, adaptor values are VM refs to buffer_views.
    // Get the buffer_views for each CSR component:
    //   data (values), indices (col_idx), indptr (row_ptr), rhs
    Value values = adaptor.getData();      // CSR values (nnz elements)
    Value colIdx = adaptor.getIndices();   // CSR column indices (nnz elements)
    Value rowPtr = adaptor.getIndptr();    // CSR row pointers (n+1 elements)
    Value rhs = adaptor.getRhs();          // Right-hand side vector (n elements)

    // Read n and nnz from attributes stored during HAL conversion.
    // At VM conversion time, the original tensor types are gone (converted to
    // buffer_views), so we read from attributes set by SpsolveHALConversion.
    auto nAttr = op->getAttrOfType<IntegerAttr>("sparse_solver.n");
    auto nnzAttr = op->getAttrOfType<IntegerAttr>("sparse_solver.nnz");

    if (!nAttr || !nnzAttr) {
      return rewriter.notifyMatchFailure(
          op, "missing sparse_solver.n or sparse_solver.nnz attributes - "
              "dynamic shapes not yet supported");
    }

    int64_t n = nAttr.getInt();
    int64_t nnz = nnzAttr.getInt();

    // Create i64 constants for n and nnz
    Value nVal = IREE::VM::ConstI64Op::create(rewriter, loc, n);
    Value nnzVal = IREE::VM::ConstI64Op::create(rewriter, loc, nnz);

    // The solution buffer - for in-place solve, we use the rhs buffer.
    // Note: The spsolve_complete import expects a pre-allocated solution buffer.
    // We're reusing rhs as solution (in-place solve).
    Value solution = rhs;

    // Emit vm.call @sparse_solver.spsolve_complete
    // Signature: (i64, i64, buffer_view, buffer_view, buffer_view, buffer_view, buffer_view) -> void
    auto callOp = IREE::VM::CallOp::create(
        rewriter, loc, importOp.getSymNameAttr(),
        TypeRange{},  // No return values (void)
        ValueRange{nVal, nnzVal, rowPtr, colIdx, values, rhs, solution});
    copyImportAttrs(importOp, callOp);

    // The result of spsolve is the solution buffer (same as rhs for in-place)
    rewriter.replaceOp(op, rhs);
    return success();
  }

private:
  SymbolTable &importSymbols;
  mutable IREE::VM::ImportOp importOp;
};

// Pattern to convert sparse_solver.cholesky_solve to VM calls.
class CholeskySolveOpConversion : public OpConversionPattern<CholeskySolveOp> {
public:
  CholeskySolveOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                            TypeConverter &typeConverter)
      : OpConversionPattern<CholeskySolveOp>(typeConverter, context),
        importSymbols(importSymbols) {}

  LogicalResult
  matchAndRewrite(CholeskySolveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Same placeholder implementation as SpsolveOpConversion
    rewriter.replaceOp(op, adaptor.getRhs());
    return success();
  }

private:
  SymbolTable &importSymbols;
};

void populateSparseSolverToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  // Tensor-level ops (for error reporting if they reach VM conversion).
  // These ops should be lowered to HAL-level ops before reaching VM conversion.
  patterns.insert<SpsolveOpConversion, CholeskySolveOpConversion>(
      context, importSymbols, typeConverter);

  // HAL-level ops → vm.call conversion via VMImportOpConversion.
  // These ops take HAL buffer_views directly and convert to VM calls.
  patterns.insert<VMImportOpConversion<SpsolveCompleteOp>>(
      context, importSymbols, typeConverter, "sparse_solver.spsolve_complete");
  patterns.insert<VMImportOpConversion<SpsolveCompleteF64Op>>(
      context, importSymbols, typeConverter,
      "sparse_solver.spsolve_complete.f64");
}

//===----------------------------------------------------------------------===//
// Stream→SparseSolver HAL Conversion Patterns
//===----------------------------------------------------------------------===//

// During Stream→HAL conversion, SparseSolver ops need to have their operand
// and result types converted from tensors to HAL buffer_views.

class SpsolveHALConversion : public OpConversionPattern<SpsolveOp> {
public:
  using OpConversionPattern<SpsolveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SpsolveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the converted result type
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    // Extract n and nnz from original tensor types before they're converted
    auto dataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto indptrType = dyn_cast<RankedTensorType>(op.getIndptr().getType());

    int64_t nnz = -1;
    int64_t n = -1;

    if (dataType && dataType.hasStaticShape()) {
      nnz = dataType.getShape()[0];
    }
    if (indptrType && indptrType.hasStaticShape()) {
      n = indptrType.getShape()[0] - 1;
    }

    // Create new op with converted types using adaptor operands
    auto newOp = rewriter.create<SpsolveOp>(
        op.getLoc(), resultType, adaptor.getData(), adaptor.getIndices(),
        adaptor.getIndptr(), adaptor.getRhs());

    // Store n and nnz as attributes for VM conversion to use
    if (n >= 0) {
      newOp->setAttr("sparse_solver.n",
                     rewriter.getIntegerAttr(rewriter.getI64Type(), n));
    }
    if (nnz >= 0) {
      newOp->setAttr("sparse_solver.nnz",
                     rewriter.getIntegerAttr(rewriter.getI64Type(), nnz));
    }

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

class CholeskySolveHALConversion : public OpConversionPattern<CholeskySolveOp> {
public:
  using OpConversionPattern<CholeskySolveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CholeskySolveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    rewriter.replaceOpWithNewOp<CholeskySolveOp>(
        op, resultType, adaptor.getData(), adaptor.getIndices(),
        adaptor.getIndptr(), adaptor.getRhs());
    return success();
  }
};

void populateStreamToSparseSolverPatterns(MLIRContext *context,
                                           ConversionTarget &target,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  // Add conversion patterns for tensor→HAL type conversion
  patterns.insert<SpsolveHALConversion, CholeskySolveHALConversion>(
      typeConverter, context);

  // Mark ops as dynamically legal once converted to HAL types
  target.addDynamicallyLegalOp<SpsolveOp, CholeskySolveOp>(
      [](Operation *op) {
        // Legal once all operands/results are NOT tensors
        for (Type type : op->getOperandTypes()) {
          if (isa<TensorType>(type))
            return false;
        }
        for (Type type : op->getResultTypes()) {
          if (isa<TensorType>(type))
            return false;
        }
        return true;
      });

  // HAL-level ops are always legal
  target.addLegalOp<SpsolveCompleteOp, SpsolveCompleteF64Op>();
}

} // namespace

} // namespace mlir::iree_compiler::IREE::SparseSolver

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverOps.cpp.inc"
