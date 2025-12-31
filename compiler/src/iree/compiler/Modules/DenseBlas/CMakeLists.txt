# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Dense BLAS module compiler support.
#
# This module provides compiler integration for the dense_blas runtime module,
# which dispatches optimized BLAS operations to platform-native libraries:
#   - Apple Silicon: Accelerate framework (cblas_sgemm via AMX)
#   - CUDA: cuBLAS (planned)
#   - ROCm: rocBLAS (planned)
#   - Generic: CLBlast (planned)
#
# Current Status:
# - Runtime module: Implemented (iree/runtime/src/iree/modules/dense_blas/)
# - VM imports: Defined (dense_blas.imports.mlir)
# - Pattern detection: Implemented (Codegen/ExternalDispatch/LinalgMatmulToDenseBlas.cpp)
# - Full lowering: TODO - need conversion pass to emit vm.call @dense_blas.matmul
#
# The full integration path is:
#   linalg.matmul
#   -> [LinalgMatmulToDenseBlasPass marks operation]
#   -> [Bufferization creates HAL buffer_views]
#   -> [DenseBlasToVM conversion emits vm.call] (TODO)
#   -> vm.call @dense_blas.matmul
#   -> Runtime: dense_blas module executes via Accelerate/cuBLAS/etc.

# Embed the imports file for use by the DenseBlas dialect.
iree_c_embed_data(
  NAME
    dense_blas_imports
  SRCS
    "dense_blas.imports.mlir"
  C_FILE_OUTPUT
    "dense_blas.imports.c"
  H_FILE_OUTPUT
    "dense_blas.imports.h"
  IDENTIFIER
    "iree_dense_blas_imports"
  FLATTEN
  PUBLIC
)
