// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SparseSolver to VM Conversion
//===----------------------------------------------------------------------===//
//
// Converts SparseSolver HAL-level ops to VM calls to the sparse_solver module.
// This must run during the HAL→VM conversion phase when buffer_views are
// available.
//
// Pipeline integration:
// 1. StableHLOCustomCalls creates sparse_solver.spsolve (tensor level)
// 2. [This file] Converts to sparse_solver.spsolve_complete (HAL level)
// 3. HAL→VM conversion converts to vm.call @sparse_solver.spsolve_complete
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverDialect.h"
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// SpsolveComplete to VM Call Conversion
//===----------------------------------------------------------------------===//

// Converts sparse_solver.spsolve_complete to vm.call @sparse_solver.spsolve_complete
struct SpsolveCompleteOpConversion
    : public OpConversionPattern<IREE::SparseSolver::SpsolveCompleteOp> {
  SpsolveCompleteOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                               TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(
        "sparse_solver.spsolve_complete");
  }

  LogicalResult
  matchAndRewrite(IREE::SparseSolver::SpsolveCompleteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!importOp) {
      return rewriter.notifyMatchFailure(
          op, "sparse_solver.spsolve_complete import not found");
    }

    // Build the VM call with all arguments
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, importOp.getSymNameAttr(), TypeRange{},
        ValueRange{
            adaptor.getN(),
            adaptor.getNnz(),
            adaptor.getRowPtr(),
            adaptor.getColIdx(),
            adaptor.getValues(),
            adaptor.getRhs(),
            adaptor.getSolution(),
        });
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

// Converts sparse_solver.spsolve_complete.f64 to vm.call
struct SpsolveCompleteF64OpConversion
    : public OpConversionPattern<IREE::SparseSolver::SpsolveCompleteF64Op> {
  SpsolveCompleteF64OpConversion(MLIRContext *context,
                                  SymbolTable &importSymbols,
                                  TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(
        "sparse_solver.spsolve_complete.f64");
  }

  LogicalResult
  matchAndRewrite(IREE::SparseSolver::SpsolveCompleteF64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!importOp) {
      return rewriter.notifyMatchFailure(
          op, "sparse_solver.spsolve_complete.f64 import not found");
    }

    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, importOp.getSymNameAttr(), TypeRange{},
        ValueRange{
            adaptor.getN(),
            adaptor.getNnz(),
            adaptor.getRowPtr(),
            adaptor.getColIdx(),
            adaptor.getValues(),
            adaptor.getRhs(),
            adaptor.getSolution(),
        });
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateSparseSolverToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.insert<SpsolveCompleteOpConversion>(context, importSymbols,
                                                typeConverter);
  patterns.insert<SpsolveCompleteF64OpConversion>(context, importSymbols,
                                                   typeConverter);
}

} // namespace mlir::iree_compiler
