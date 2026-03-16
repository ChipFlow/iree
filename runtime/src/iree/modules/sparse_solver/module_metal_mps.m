// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MPS-based dense solve for Metal.
// This file is compiled separately with -flax-vector-conversions because
// MetalPerformanceShaders headers use simd types that conflict with IREE's
// strict -fno-lax-vector-conversions flag.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "iree/modules/sparse_solver/module_metal_mps.h"

// Metal shader source for applying LAPACK-style pivot permutation to a vector.
// Pivots are sequential swaps: for i=0..n-1, swap vec[i] with vec[pivots[i]].
// MPS produces 0-based uint32 pivot indices.
static NSString* const kPivotKernelSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void apply_pivots(\n"
     "    device const uint* pivots [[buffer(0)]],\n"
     "    device float* vec [[buffer(1)]],\n"
     "    constant uint& n [[buffer(2)]],\n"
     "    uint tid [[thread_position_in_grid]])\n"
     "{\n"
     "    if (tid != 0) return;\n"
     "    for (uint i = 0; i < n; i++) {\n"
     "        uint j = pivots[i];\n"
     "        if (j != i) {\n"
     "            float tmp = vec[i];\n"
     "            vec[i] = vec[j];\n"
     "            vec[j] = tmp;\n"
     "        }\n"
     "    }\n"
     "}\n";

// Cached pivot kernel pipeline state (created once per device).
static id<MTLComputePipelineState> g_pivot_pso = nil;
static id<MTLDevice> g_pivot_pso_device = nil;
static NSObject* g_pivot_lock = nil;

static id<MTLComputePipelineState>
iree_sparse_solver_metal_get_pivot_pso(id<MTLDevice> device) {
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{ g_pivot_lock = [[NSObject alloc] init]; });
  @synchronized(g_pivot_lock) {
    if (g_pivot_pso && g_pivot_pso_device == device) return g_pivot_pso;
    NSError* error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kPivotKernelSource
                                             options:nil
                                               error:&error];
    if (!lib) return nil;
    id<MTLFunction> func = [lib newFunctionWithName:@"apply_pivots"];
    if (!func) return nil;
    g_pivot_pso = [device newComputePipelineStateWithFunction:func error:&error];
    g_pivot_pso_device = device;
    return g_pivot_pso;
  }
}

iree_status_t iree_sparse_solver_metal_mps_dense_solve(
    void* mtl_cmd_buf_ptr, void* mtl_device_ptr,
    void* mtl_matrix_ptr, uint64_t matrix_byte_off,
    void* mtl_rhs_ptr, uint64_t rhs_byte_off,
    void* mtl_solution_ptr, uint64_t sol_byte_off,
    int64_t n) {
  id<MTLCommandBuffer> mtl_cmd_buf =
      (__bridge id<MTLCommandBuffer>)mtl_cmd_buf_ptr;
  id<MTLDevice> device = (__bridge id<MTLDevice>)mtl_device_ptr;
  id<MTLBuffer> mtl_matrix = (__bridge id<MTLBuffer>)mtl_matrix_ptr;
  id<MTLBuffer> mtl_rhs = (__bridge id<MTLBuffer>)mtl_rhs_ptr;
  id<MTLBuffer> mtl_sol = (__bridge id<MTLBuffer>)mtl_solution_ptr;

  // Allocate scratch buffers for the solve pipeline.
  id<MTLBuffer> pivot_buf =
      [device newBufferWithLength:n * sizeof(uint32_t)
                         options:MTLResourceStorageModePrivate];
  id<MTLBuffer> temp1_buf =
      [device newBufferWithLength:n * sizeof(float)
                         options:MTLResourceStorageModePrivate];
  id<MTLBuffer> temp2_buf =
      [device newBufferWithLength:n * sizeof(float)
                         options:MTLResourceStorageModePrivate];
  if (!pivot_buf || !temp1_buf || !temp2_buf) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to allocate scratch buffers for MPS solve");
  }

  @autoreleasepool {
    // --- Step 1: Copy rhs → temp1 (will be permuted in-place) ---
    id<MTLBlitCommandEncoder> blit = [mtl_cmd_buf blitCommandEncoder];
    [blit copyFromBuffer:mtl_rhs
            sourceOffset:(NSUInteger)rhs_byte_off
                toBuffer:temp1_buf
       destinationOffset:0
                    size:n * sizeof(float)];
    [blit endEncoding];

    // --- Step 2: LU factorization of matrix in-place ---
    MPSMatrixDescriptor* mat_desc = [MPSMatrixDescriptor
        matrixDescriptorWithRows:n
                         columns:n
                        rowBytes:n * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrix* mps_a = [[MPSMatrix alloc]
        initWithBuffer:mtl_matrix
                offset:(NSUInteger)matrix_byte_off
            descriptor:mat_desc];

    MPSMatrixDescriptor* pivot_mat_desc = [MPSMatrixDescriptor
        matrixDescriptorWithRows:1
                         columns:n
                        rowBytes:n * sizeof(uint32_t)
                        dataType:MPSDataTypeUInt32];
    MPSMatrix* mps_pivots = [[MPSMatrix alloc] initWithBuffer:pivot_buf
                                                       offset:0
                                                   descriptor:pivot_mat_desc];

    MPSMatrixDecompositionLU* mps_lu = [[MPSMatrixDecompositionLU alloc]
        initWithDevice:device
                  rows:n
               columns:n];
    [mps_lu encodeToCommandBuffer:mtl_cmd_buf
                     sourceMatrix:mps_a
                     resultMatrix:mps_a
                     pivotIndices:mps_pivots
                           status:nil];

    // --- Step 3: Apply pivot permutation to temp1 ---
    id<MTLComputePipelineState> pivot_pso =
        iree_sparse_solver_metal_get_pivot_pso(device);
    if (!pivot_pso) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "failed to compile pivot permutation kernel");
    }
    {
      id<MTLComputeCommandEncoder> enc =
          [mtl_cmd_buf computeCommandEncoder];
      [enc setComputePipelineState:pivot_pso];
      [enc setBuffer:pivot_buf offset:0 atIndex:0];
      [enc setBuffer:temp1_buf offset:0 atIndex:1];
      uint32_t n32 = (uint32_t)n;
      [enc setBytes:&n32 length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }

    // --- Step 4: Forward solve: L * y = P*b ---
    // L is unit lower triangular (stored in lower triangle of factored A).
    MPSMatrixDescriptor* vec_desc = [MPSMatrixDescriptor
        matrixDescriptorWithRows:n
                         columns:1
                        rowBytes:sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrix* mps_pb = [[MPSMatrix alloc] initWithBuffer:temp1_buf
                                                   offset:0
                                               descriptor:vec_desc];
    MPSMatrix* mps_y = [[MPSMatrix alloc] initWithBuffer:temp2_buf
                                                  offset:0
                                              descriptor:vec_desc];

    MPSMatrixSolveTriangular* fwd_solve = [[MPSMatrixSolveTriangular alloc]
        initWithDevice:device
                 right:NO
                 upper:NO
             transpose:NO
                  unit:YES
                 order:n
        numberOfRightHandSides:1
                 alpha:1.0];
    [fwd_solve encodeToCommandBuffer:mtl_cmd_buf
                        sourceMatrix:mps_a
                rightHandSideMatrix:mps_pb
                     solutionMatrix:mps_y];

    // --- Step 5: Backward solve: U * x = y ---
    // U is upper triangular (stored in upper triangle of factored A).
    MPSMatrix* mps_x = [[MPSMatrix alloc]
        initWithBuffer:mtl_sol
                offset:(NSUInteger)sol_byte_off
            descriptor:vec_desc];

    MPSMatrixSolveTriangular* bwd_solve = [[MPSMatrixSolveTriangular alloc]
        initWithDevice:device
                 right:NO
                 upper:YES
             transpose:NO
                  unit:NO
                 order:n
        numberOfRightHandSides:1
                 alpha:1.0];
    [bwd_solve encodeToCommandBuffer:mtl_cmd_buf
                        sourceMatrix:mps_a
                rightHandSideMatrix:mps_y
                     solutionMatrix:mps_x];
  }

  return iree_ok_status();
}
