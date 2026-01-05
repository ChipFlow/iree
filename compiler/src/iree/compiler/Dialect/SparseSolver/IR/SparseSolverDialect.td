//===- SparseSolverDialect.td - Sparse Solver dialect -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SparseSolver dialect for IREE.
// The sparse solver dialect provides operations for sparse linear algebra
// that are lowered to calls into the BaSpaCho runtime module.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_DIALECT_SPARSESOLVER_IR_SPARSESOLVERDIALECT_TD
#define IREE_DIALECT_SPARSESOLVER_IR_SPARSESOLVERDIALECT_TD

include "mlir/IR/OpBase.td"

def SparseSolver_Dialect : Dialect {
  let name = "sparse_solver";
  let cppNamespace = "::mlir::iree_compiler::IREE::SparseSolver";
  let summary = "Sparse direct solver operations for IREE";
  let description = [{
    The SparseSolver dialect provides operations for sparse linear algebra
    that are lowered to calls into the BaSpaCho sparse direct solver.

    BaSpaCho supports multiple backends:
    - Metal (Apple GPUs)
    - CUDA (NVIDIA GPUs)
    - OpenCL (generic fallback)
    - CPU (reference implementation)

    Typical usage:
    1. sparse_solver.spsolve - Solve Ax=b for sparse A in CSR format

    The operations in this dialect are lowered to vm.call operations that
    invoke the sparse_solver runtime module.
  }];
}

#endif // IREE_DIALECT_SPARSESOLVER_IR_SPARSESOLVERDIALECT_TD
