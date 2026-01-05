// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_SPARSE_SOLVER_BASPACHO_WRAPPER_H_
#define IREE_MODULES_SPARSE_SOLVER_BASPACHO_WRAPPER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// BaSpaCho C API Wrapper
//===----------------------------------------------------------------------===//
// This provides a C interface to BaSpaCho's C++ API for use from IREE's
// C runtime modules.
//
// BaSpaCho uses a two-phase approach:
// 1. Symbolic analysis (SymbolicCtx) - analyzes sparsity pattern
// 2. Numeric factorization (NumericCtx) - performs actual factorization
// 3. Solve (SolveCtx) - forward/backward substitution

// Backend selection enum.
typedef enum baspacho_backend_e {
  BASPACHO_BACKEND_AUTO = 0,    // Auto-detect best backend
  BASPACHO_BACKEND_CPU = 1,     // Eigen/OpenBLAS on CPU
  BASPACHO_BACKEND_CUDA = 2,    // CUDA + cuBLAS
  BASPACHO_BACKEND_METAL = 3,   // Metal compute shaders
  BASPACHO_BACKEND_OPENCL = 4,  // OpenCL + CLBlast
} baspacho_backend_t;

// Opaque handle to BaSpaCho context.
typedef struct baspacho_context_s* baspacho_handle_t;

//===----------------------------------------------------------------------===//
// Context Management
//===----------------------------------------------------------------------===//

// Create a BaSpaCho context with the specified backend.
// Returns NULL on failure.
baspacho_handle_t baspacho_create(baspacho_backend_t backend);

// Create a BaSpaCho context using a specific GPU device.
// For Metal: pass id<MTLDevice> as device_handle
// For CUDA: pass CUdevice as device_handle
// For OpenCL: pass cl_device_id as device_handle
baspacho_handle_t baspacho_create_with_device(baspacho_backend_t backend,
                                               void* device_handle);

// Destroy a BaSpaCho context and free all resources.
void baspacho_destroy(baspacho_handle_t h);

// Get the active backend for a context.
baspacho_backend_t baspacho_get_backend(baspacho_handle_t h);

//===----------------------------------------------------------------------===//
// Symbolic Analysis
//===----------------------------------------------------------------------===//

// Perform symbolic analysis on a sparse matrix in CSR format.
// The sparsity pattern is assumed to be symmetric.
//
// Parameters:
//   h       - BaSpaCho context
//   n       - Matrix dimension (n x n)
//   nnz     - Number of non-zeros in lower triangle
//   row_ptr - CSR row pointers (n+1 elements)
//   col_idx - CSR column indices (nnz elements)
//
// Returns:
//   0 on success, non-zero error code on failure.
int baspacho_analyze(baspacho_handle_t h, int64_t n, int64_t nnz,
                     const int64_t* row_ptr, const int64_t* col_idx);

// Get the number of non-zeros in the L factor after analysis.
int64_t baspacho_get_factor_nnz(baspacho_handle_t h);

// Get the number of supernodes after analysis.
int64_t baspacho_get_num_supernodes(baspacho_handle_t h);

//===----------------------------------------------------------------------===//
// Numeric Factorization
//===----------------------------------------------------------------------===//

// Perform numeric Cholesky factorization with f32 values.
// baspacho_analyze() must be called first.
//
// Parameters:
//   h      - BaSpaCho context
//   values - CSR values array (nnz elements, same order as analyze)
//
// Returns:
//   0 on success
//   > 0 if matrix is not positive definite (returns row where failure occurred)
//   < 0 on other errors
int baspacho_factor_f32(baspacho_handle_t h, const float* values);

// Perform numeric Cholesky factorization with f64 values.
int baspacho_factor_f64(baspacho_handle_t h, const double* values);

// GPU variants that operate directly on device pointers (zero-copy).
// These avoid CPU<->GPU transfers when data is already on GPU.
int baspacho_factor_f32_device(baspacho_handle_t h, void* device_ptr);
int baspacho_factor_f64_device(baspacho_handle_t h, void* device_ptr);

//===----------------------------------------------------------------------===//
// LU Factorization (for non-symmetric/indefinite matrices)
//===----------------------------------------------------------------------===//

// Perform numeric LU factorization with partial pivoting.
// Use this for general (non-symmetric) or symmetric indefinite matrices.
// baspacho_analyze() must be called first.
//
// Parameters:
//   h      - BaSpaCho context
//   values - CSR values array (nnz elements, same order as analyze)
//   pivots - Output pivot array (n elements, caller-allocated)
//
// Returns:
//   0 on success
//   > 0 if matrix is singular (returns row where zero pivot occurred)
//   < 0 on other errors
int baspacho_factor_lu_f32(baspacho_handle_t h, const float* values,
                            int64_t* pivots);
int baspacho_factor_lu_f64(baspacho_handle_t h, const double* values,
                            int64_t* pivots);

// GPU variants for LU factorization.
int baspacho_factor_lu_f32_device(baspacho_handle_t h, void* device_ptr,
                                   int64_t* pivots);
int baspacho_factor_lu_f64_device(baspacho_handle_t h, void* device_ptr,
                                   int64_t* pivots);

//===----------------------------------------------------------------------===//
// Solve Operations
//===----------------------------------------------------------------------===//

// Solve L*L^T*x = b (or L*D*L^T*x = b depending on factorization).
// baspacho_factor must be called first.
//
// Parameters:
//   h        - BaSpaCho context
//   rhs      - Right-hand side vector (n elements)
//   solution - Solution vector (n elements, can be same as rhs for in-place)
void baspacho_solve_f32(baspacho_handle_t h, const float* rhs, float* solution);
void baspacho_solve_f64(baspacho_handle_t h, const double* rhs,
                        double* solution);

// GPU variants for zero-copy operations.
void baspacho_solve_f32_device(baspacho_handle_t h, void* rhs_device,
                                void* solution_device);
void baspacho_solve_f64_device(baspacho_handle_t h, void* rhs_device,
                                void* solution_device);

//===----------------------------------------------------------------------===//
// LU Solve Operations
//===----------------------------------------------------------------------===//

// Solve P*L*U*x = b using LU factorization with pivoting.
// baspacho_factor_lu must be called first with the same pivots array.
//
// Parameters:
//   h        - BaSpaCho context
//   pivots   - Pivot array from baspacho_factor_lu (n elements)
//   rhs      - Right-hand side vector (n elements)
//   solution - Solution vector (n elements, can be same as rhs for in-place)
void baspacho_solve_lu_f32(baspacho_handle_t h, const int64_t* pivots,
                            const float* rhs, float* solution);
void baspacho_solve_lu_f64(baspacho_handle_t h, const int64_t* pivots,
                            const double* rhs, double* solution);

// GPU variants for LU solve.
void baspacho_solve_lu_f32_device(baspacho_handle_t h, const int64_t* pivots,
                                   void* rhs_device, void* solution_device);
void baspacho_solve_lu_f64_device(baspacho_handle_t h, const int64_t* pivots,
                                   void* rhs_device, void* solution_device);

// Batched solve: solve for multiple right-hand sides simultaneously.
// This is BaSpaCho's specialty - optimized for batched operations.
//
// Parameters:
//   h        - BaSpaCho context
//   rhs      - Right-hand side matrix (n x num_rhs, column-major)
//   solution - Solution matrix (n x num_rhs, column-major)
//   num_rhs  - Number of right-hand sides
void baspacho_solve_batched_f32(baspacho_handle_t h, const float* rhs,
                                 float* solution, int64_t num_rhs);
void baspacho_solve_batched_f64(baspacho_handle_t h, const double* rhs,
                                 double* solution, int64_t num_rhs);

// GPU batched solve variants.
void baspacho_solve_batched_f32_device(baspacho_handle_t h, void* rhs_device,
                                        void* solution_device, int64_t num_rhs);
void baspacho_solve_batched_f64_device(baspacho_handle_t h, void* rhs_device,
                                        void* solution_device, int64_t num_rhs);

//===----------------------------------------------------------------------===//
// Async Operations (for zero-copy GPU integration)
//===----------------------------------------------------------------------===//

// Set the command queue/stream for async operations.
// For Metal: pass id<MTLCommandBuffer>
// For CUDA: pass CUstream
// For OpenCL: pass cl_command_queue
void baspacho_set_command_queue(baspacho_handle_t h, void* queue);

// Async factor - encodes work to command buffer/stream.
// Does not block; use synchronization primitives to wait.
int baspacho_factor_f32_async(baspacho_handle_t h, void* device_ptr);
int baspacho_factor_f64_async(baspacho_handle_t h, void* device_ptr);

// Async solve - encodes work to command buffer/stream.
void baspacho_solve_f32_async(baspacho_handle_t h, void* rhs_device,
                               void* solution_device);
void baspacho_solve_f64_async(baspacho_handle_t h, void* rhs_device,
                               void* solution_device);

//===----------------------------------------------------------------------===//
// Buffer Registration (for zero-copy GPU integration)
//===----------------------------------------------------------------------===//

// Register a Metal buffer with BaSpaCho's buffer registry.
// This enables zero-copy operations when using baspacho_*_device() functions.
//
// Parameters:
//   host_ptr   - CPU-accessible pointer to buffer contents (from [MTLBuffer contents])
//   mtl_buffer - The MTLBuffer handle (as void*)
//   size       - Size of the buffer in bytes
void baspacho_register_metal_buffer(void* host_ptr, void* mtl_buffer,
                                     size_t size);

// Unregister a previously registered Metal buffer.
void baspacho_unregister_metal_buffer(void* host_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_MODULES_SPARSE_SOLVER_BASPACHO_WRAPPER_H_
