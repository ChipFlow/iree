//===-- Passes.h - External Dispatch Passes ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes for lowering operations to external library calls.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_EXTERNALDISPATCH_PASSES_H_
#define IREE_COMPILER_CODEGEN_EXTERNALDISPATCH_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Creates a pass that converts linalg.matmul operations to dense_blas.gemm
/// calls when they exceed the size threshold.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgMatmulToDenseBlasPass(int64_t sizeThreshold = 4096,
                                   bool preferGpu = true);

/// Creates a pass that converts linalg contractions (syrk, trsm patterns) to
/// dense_blas module calls.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgContractionToDenseBlasPass(int64_t sizeThreshold = 1024);

/// Creates a pass that detects sparse Cholesky patterns and lowers them
/// to sparse_solver module calls (backed by BaSpaCho).
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createSparseCholeskySolverPass(int64_t sizeThreshold = 1000);

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void registerExternalDispatchPasses();

//===----------------------------------------------------------------------===//
// Generated Pass Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/ExternalDispatch/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/ExternalDispatch/Passes.h.inc"

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_CODEGEN_EXTERNALDISPATCH_PASSES_H_
