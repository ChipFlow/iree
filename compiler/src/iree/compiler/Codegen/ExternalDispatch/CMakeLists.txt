# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

option(IREE_EXTERNAL_DISPATCH "Enable external BLAS/solver dispatch passes" OFF)

if(NOT IREE_EXTERNAL_DISPATCH)
  return()
endif()

# Generate pass registration from tablegen.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ExternalDispatch)
add_public_tablegen_target(IREECodegenExternalDispatchPassIncGen)

iree_cc_library(
  NAME
    ExternalDispatch
  HDRS
    "Passes.h"
  SRCS
    "LinalgMatmulToDenseBlas.cpp"
    "SparseCholeskySolver.cpp"
  DEPS
    ::ExternalDispatchPassIncGen
    IREEFlowDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRPass
    MLIRRewrite
    MLIRSparseTensorDialect
    MLIRTransforms
  PUBLIC
)

# Register passes when the library is built.
iree_cc_library(
  NAME
    ExternalDispatchPassRegistration
  SRCS
    "PassRegistration.cpp"
  DEPS
    ::ExternalDispatch
  PUBLIC
)
