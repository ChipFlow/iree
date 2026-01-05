# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# SparseSolver Dialect Conversion Patterns
#-------------------------------------------------------------------------------
#
# Provides conversion patterns for lowering SparseSolver ops to VM calls.
#
# Integration path:
# 1. StableHLOCustomCalls creates sparse_solver.spsolve (tensor level)
# 2. ConvertSparseSolverToFlow converts to flow.func/flow.call (TODO)
# 3. Flow→Stream→HAL pipeline bufferizes tensors to buffer_views
# 4. ConvertSparseSolverToVM converts HAL-level ops to vm.call
#
# Current status:
# - ConvertSparseSolverToVM is implemented
# - ConvertSparseSolverToFlow is TODO (requires integration with Stream bufferization)
#
#-------------------------------------------------------------------------------

iree_cc_library(
  NAME
    SparseSolverToVM
  HDRS
    "ConvertSparseSolverToVM.h"
  SRCS
    "ConvertSparseSolverToVM.cpp"
  DEPS
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::SparseSolver::IR
    iree::compiler::Dialect::VM::Conversion::ImportUtils
    iree::compiler::Dialect::VM::IR
    MLIRPass
    MLIRTransforms
  PUBLIC
)
