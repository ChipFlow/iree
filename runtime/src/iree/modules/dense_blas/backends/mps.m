// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/dense_blas/backends/mps.h"

#if defined(__APPLE__) && defined(IREE_DENSE_BLAS_HAVE_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

//===----------------------------------------------------------------------===//
// MPS Context
//===----------------------------------------------------------------------===//

typedef struct iree_dense_blas_mps_context_s {
  iree_allocator_t allocator;
  id<MTLDevice> device;
  // Cache commonly-used MPS objects for performance.
  // Note: MPS kernels are lightweight and can be created per-call if needed.
} iree_dense_blas_mps_context_s;

iree_status_t iree_dense_blas_mps_context_create(
    void* metal_device,
    iree_allocator_t allocator,
    iree_dense_blas_mps_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(metal_device);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;

  iree_dense_blas_mps_context_s* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(*context), (void**)&context));

  context->allocator = allocator;
  context->device = (__bridge id<MTLDevice>)metal_device;
  // Retain the device to ensure it outlives the context.
  CFRetain((__bridge CFTypeRef)context->device);

  *out_context = context;
  return iree_ok_status();
}

void iree_dense_blas_mps_context_destroy(
    iree_dense_blas_mps_context_t context) {
  if (!context) return;
  CFRelease((__bridge CFTypeRef)context->device);
  iree_allocator_free(context->allocator, context);
}

//===----------------------------------------------------------------------===//
// GEMM via MPSMatrixMultiplication
//===----------------------------------------------------------------------===//

iree_status_t iree_dense_blas_mps_gemm_f32(
    iree_dense_blas_mps_context_t context,
    void* command_buffer_ptr,
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    void* A_ptr,
    int64_t lda,
    void* B_ptr,
    int64_t ldb,
    float beta,
    void* C_ptr,
    int64_t ldc) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(command_buffer_ptr);
  IREE_ASSERT_ARGUMENT(A_ptr);
  IREE_ASSERT_ARGUMENT(B_ptr);
  IREE_ASSERT_ARGUMENT(C_ptr);

  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)command_buffer_ptr;
    id<MTLBuffer> A = (__bridge id<MTLBuffer>)A_ptr;
    id<MTLBuffer> B = (__bridge id<MTLBuffer>)B_ptr;
    id<MTLBuffer> C = (__bridge id<MTLBuffer>)C_ptr;

    // Create MPS matrix descriptors.
    // For GEMM: C = alpha * op(A) * op(B) + beta * C
    // A is (M x K), B is (K x N), C is (M x N) in column-major.

    NSUInteger rowBytesA = (trans_a ? K : M) * sizeof(float);
    NSUInteger rowBytesB = (trans_b ? N : K) * sizeof(float);
    NSUInteger rowBytesC = M * sizeof(float);

    // MPS uses row-major by default, so we swap dimensions and transposes
    // to handle column-major BLAS convention.
    // C^T = (alpha * op(A) * op(B) + beta * C)^T
    //     = alpha * op(B)^T * op(A)^T + beta * C^T

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(trans_a ? M : K)
                         columns:(trans_a ? K : M)
                        rowBytes:lda * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(trans_b ? K : N)
                         columns:(trans_b ? N : K)
                        rowBytes:ldb * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:N
                         columns:M
                        rowBytes:ldc * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    // Create MPS matrix objects wrapping the Metal buffers.
    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:A descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:B descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:C descriptor:descC];

    // Create the matrix multiplication kernel.
    // Swap A and B for column-major handling.
    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:context->device
         transposeLeft:trans_b
        transposeRight:trans_a
            resultRows:N
         resultColumns:M
       interiorColumns:K
                 alpha:(double)alpha
                  beta:(double)beta];

    // Encode to command buffer.
    [gemm encodeToCommandBuffer:commandBuffer
                     leftMatrix:matB   // B becomes left in transposed view
                    rightMatrix:matA   // A becomes right in transposed view
                   resultMatrix:matC];
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// TRSM via MPSMatrixSolveTriangular
//===----------------------------------------------------------------------===//

iree_status_t iree_dense_blas_mps_trsm_f32(
    iree_dense_blas_mps_context_t context,
    void* command_buffer_ptr,
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    float alpha,
    void* A_ptr,
    int64_t lda,
    void* B_ptr,
    int64_t ldb) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(command_buffer_ptr);
  IREE_ASSERT_ARGUMENT(A_ptr);
  IREE_ASSERT_ARGUMENT(B_ptr);

  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)command_buffer_ptr;
    id<MTLBuffer> A = (__bridge id<MTLBuffer>)A_ptr;
    id<MTLBuffer> B = (__bridge id<MTLBuffer>)B_ptr;

    // Determine matrix sizes.
    NSUInteger order = left_side ? M : N;

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:order
                         columns:order
                        rowBytes:lda * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:N
                        rowBytes:ldb * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:A descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:B descriptor:descB];

    // Create solve triangular kernel.
    MPSMatrixSolveTriangular* trsm = [[MPSMatrixSolveTriangular alloc]
        initWithDevice:context->device
                 right:!left_side
                 upper:upper
             transpose:trans_a
                  unit:unit_diag
                 order:order
      numberOfRightHandSides:(left_side ? N : M)
                 alpha:(double)alpha];

    // Encode to command buffer.
    [trsm encodeToCommandBuffer:commandBuffer
                 sourceMatrix:matA
              rightHandSideMatrix:matB
                 solutionMatrix:matB];  // In-place solve
  }

  return iree_ok_status();
}

#else  // !__APPLE__ || !IREE_DENSE_BLAS_HAVE_METAL

// Stub implementations for non-Apple platforms.

iree_status_t iree_dense_blas_mps_context_create(
    void* metal_device,
    iree_allocator_t allocator,
    iree_dense_blas_mps_context_t* out_context) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "MPS backend not available on this platform");
}

void iree_dense_blas_mps_context_destroy(
    iree_dense_blas_mps_context_t context) {
  // No-op.
}

iree_status_t iree_dense_blas_mps_gemm_f32(
    iree_dense_blas_mps_context_t context,
    void* command_buffer,
    bool trans_a, bool trans_b,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    void* A,
    int64_t lda,
    void* B,
    int64_t ldb,
    float beta,
    void* C,
    int64_t ldc) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "MPS backend not available on this platform");
}

iree_status_t iree_dense_blas_mps_trsm_f32(
    iree_dense_blas_mps_context_t context,
    void* command_buffer,
    bool left_side, bool upper, bool trans_a, bool unit_diag,
    int64_t M, int64_t N,
    float alpha,
    void* A,
    int64_t lda,
    void* B,
    int64_t ldb) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "MPS backend not available on this platform");
}

#endif  // __APPLE__ && IREE_DENSE_BLAS_HAVE_METAL
