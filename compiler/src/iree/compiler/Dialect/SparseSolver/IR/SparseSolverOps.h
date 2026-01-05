// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SPARSESOLVER_IR_SPARSESOLVEROPS_H_
#define IREE_COMPILER_DIALECT_SPARSESOLVER_IR_SPARSESOLVEROPS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/SparseSolver/IR/SparseSolverOps.h.inc"  // IWYU pragma: keep
// clang-format on

#endif // IREE_COMPILER_DIALECT_SPARSESOLVER_IR_SPARSESOLVEROPS_H_
