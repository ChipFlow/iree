// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SPARSESOLVER_CONVERSION_CONVERTSPARSESOLVERTOVM_H_
#define IREE_COMPILER_DIALECT_SPARSESOLVER_CONVERSION_CONVERTSPARSESOLVERTOVM_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SymbolTable;
} // namespace mlir

namespace mlir::iree_compiler {

// Populates conversion patterns for SparseSolver ops to VM calls.
// The SparseSolver HAL-level ops (spsolve_complete, etc.) are converted
// to vm.call operations targeting the sparse_solver runtime module.
void populateSparseSolverToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_SPARSESOLVER_CONVERSION_CONVERTSPARSESOLVERTOVM_H_
