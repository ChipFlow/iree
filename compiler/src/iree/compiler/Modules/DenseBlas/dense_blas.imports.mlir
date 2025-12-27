// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM module imports for the dense_blas external module.
// These declarations are used by the compiler to generate vm.call operations
// that invoke the dense_blas runtime module functions.
//
// The dense_blas module is implemented in:
//   iree/runtime/src/iree/modules/dense_blas/module.c
//
// Calling convention for matmul: 0rrr_v
//   - 3 buffer_view refs in (lhs, rhs, out)
//   - void return
//
// The operation performs: out = lhs @ rhs (standard matrix multiplication)

vm.module @dense_blas {

// Matrix multiply: C = A @ B
// Performs general matrix multiplication using the platform-optimized BLAS.
// On Apple Silicon, uses Accelerate framework's cblas_sgemm (AMX-accelerated).
//
// Arguments:
//   lhs: [M, K] input matrix A
//   rhs: [K, N] input matrix B
//   out: [M, N] output matrix C (modified in-place)
//
// Supports: f32 element type (f64 support planned)
// The output buffer must be pre-allocated with correct dimensions.
vm.import private @matmul(
  %lhs : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>,
  %out : !vm.ref<!hal.buffer_view>
)

}  // module
