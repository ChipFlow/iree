// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IREE-specific passes
//===----------------------------------------------------------------------===//

def ConvertStableHloToIreeInputDialects :
    Pass<"iree-stablehlo-to-iree-input", "ModuleOp"> {
  let summary = "Convert StableHLO ops to IREE input dialects ops";
  let description = [{
    Converts StableHLO input to IREE input dialects. These include: Linalg,
    Arith, Tensor, Complex, Shape, ML Program, Flow, and Util.
  }];
}

def ConvertStableHloToLinalgExt :
    InterfacePass<"iree-stablehlo-to-linalg-ext", "mlir::FunctionOpInterface"> {
  let summary =
      "Converts from StableHLO ops to LinalgExt ops and distribute to Flow ops";
}

//===----------------------------------------------------------------------===//
// General passes
//===----------------------------------------------------------------------===//

def LegalizeControlFlow :
    InterfacePass<"iree-stablehlo-legalize-control-flow", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes from StableHLO control flow to SCF control flow";
}

def LegalizeStableHLOCustomCalls :
    InterfacePass<"iree-stablehlo-legalize-custom-calls", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes specialized custom calls to decomposed implementations";
}

def LegalizeStableHLOCholeskyToLinalg :
    InterfacePass<"iree-stablehlo-legalize-cholesky-to-linalg", "mlir::FunctionOpInterface"> {
  let summary = "Lowers stablehlo.cholesky to linalg and scf operations";
  let description = [{
    Converts stablehlo.cholesky operations to a sequence of linalg and scf
    operations that compute Cholesky decomposition. This enables the operation
    to be executed on any target without requiring external library support.

    For high-performance execution, consider using platform-specific libraries
    (Accelerate, cuSOLVER) or the sparse_solver module for sparse matrices.
  }];
}

def LegalizeStableHLOTriangularSolveToLinalg :
    InterfacePass<"iree-stablehlo-legalize-triangular-solve-to-linalg", "mlir::FunctionOpInterface"> {
  let summary = "Lowers stablehlo.triangular_solve to linalg and scf operations";
  let description = [{
    Converts stablehlo.triangular_solve operations to a sequence of linalg and
    scf operations that compute forward or back substitution. This enables the
    operation to be executed on any target without requiring external library
    support.

    For high-performance execution, consider using platform-specific BLAS
    libraries (cuBLAS, rocBLAS, Accelerate).
  }];
}

def LegalizeShapeComputations :
    InterfacePass<"iree-stablehlo-legalize-shape-computations", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes StableHLO shape operations to core-mlir operations";
}

def VerifyCompilerStableHloInputLegality :
    Pass<"iree-stablehlo-verify-compiler-input-legality", "ModuleOp"> {
  let summary =
      "Verifies that only supported IR constructs are passed to the compiler";
}

def CheckVHLOStableHloMixUsage :
    Pass<"iree-check-vhlostablehlo-mix-usage", "ModuleOp"> {
  let summary =
      "Check and report an error when VHLO and StableHLO are used in the same module";
}


#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES
