//===-- Passes.td - External Dispatch Pass Definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes for lowering operations to external library
// calls (dense BLAS, sparse solvers, etc.).
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_EXTERNALDISPATCH_PASSES_TD
#define IREE_COMPILER_CODEGEN_EXTERNALDISPATCH_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Dense BLAS Dispatch Passes
//===----------------------------------------------------------------------===//

def LinalgMatmulToDenseBlasPass :
    InterfacePass<"iree-linalg-matmul-to-dense-blas", "mlir::FunctionOpInterface"> {
  let summary = "Convert linalg.matmul operations to dense_blas.gemm calls";
  let description = [{
    This pass identifies linalg.matmul and linalg.batch_matmul operations that
    exceed a configurable size threshold and converts them to calls to the
    external dense_blas module.

    The threshold is based on the product M*N*K for the matmul dimensions.
    Operations below the threshold are left for IREE's native codegen.

    Example:
    ```mlir
    %result = linalg.matmul ins(%A, %B : tensor<1024x512xf32>, tensor<512x256xf32>)
                            outs(%C : tensor<1024x256xf32>) -> tensor<1024x256xf32>
    ```

    Is converted to a flow.dispatch calling dense_blas.gemm when the size
    exceeds the threshold (default: 4096 = 16*16*16).
  }];

  let options = [
    Option<"sizeThreshold", "size-threshold", "int64_t", /*default=*/"4096",
           "Minimum M*N*K product to dispatch to external BLAS">,
    Option<"preferGpu", "prefer-gpu", "bool", /*default=*/"true",
           "Prefer GPU BLAS backends over CPU">,
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def LinalgContractionToDenseBlasPass :
    InterfacePass<"iree-linalg-contraction-to-dense-blas", "mlir::FunctionOpInterface"> {
  let summary = "Convert linalg contractions (syrk, trsm) to dense_blas calls";
  let description = [{
    This pass identifies linalg contraction patterns that match BLAS Level 3
    operations beyond matmul:

    - Symmetric rank-k update (syrk): C = alpha * A * A^T + beta * C
    - Triangular solve (trsm): op(A) * X = alpha * B

    These patterns are detected via indexing map analysis and converted to
    the appropriate dense_blas module calls.
  }];

  let options = [
    Option<"sizeThreshold", "size-threshold", "int64_t", /*default=*/"1024",
           "Minimum matrix size to dispatch to external BLAS">,
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Sparse Solver Dispatch Passes
//===----------------------------------------------------------------------===//

def SparseCholeskySolverPass :
    InterfacePass<"iree-sparse-cholesky-to-solver", "mlir::FunctionOpInterface"> {
  let summary = "Lower sparse Cholesky patterns to sparse_solver module calls";
  let description = [{
    This pass detects sparse Cholesky factorization and solve patterns,
    marking them for dispatch to the sparse_solver module (BaSpaCho).

    Patterns detected:
    - JAX's scipy.linalg.cho_factor / cho_solve
    - Sparse matrix factorization with sparse_tensor dialect
    - Symmetric positive definite solve patterns

    The sparse_solver module provides GPU-accelerated sparse Cholesky via
    BaSpaCho, which is optimized for:
    - Levenberg-Marquardt optimization
    - Bundle adjustment / SLAM
    - Batched solves with same sparsity pattern
  }];

  let options = [
    Option<"sizeThreshold", "size-threshold", "int64_t", /*default=*/"1000",
           "Minimum matrix dimension to dispatch to sparse solver">,
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::sparse_tensor::SparseTensorDialect",
    "IREE::Flow::FlowDialect",
  ];
}

#endif // IREE_COMPILER_CODEGEN_EXTERNALDISPATCH_PASSES_TD
