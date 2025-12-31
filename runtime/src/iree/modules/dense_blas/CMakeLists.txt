# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Option to enable the dense_blas module.
option(IREE_EXTERNAL_DENSE_BLAS "Enable external dense BLAS module" OFF)

if(NOT IREE_EXTERNAL_DENSE_BLAS)
  return()
endif()

#-------------------------------------------------------------------------------
# Backend detection and configuration
#-------------------------------------------------------------------------------

set(DENSE_BLAS_SOURCES
  module.c
  hal_buffer_helper.c
)

set(DENSE_BLAS_DEPS
  iree::base
  iree::hal
  iree::vm
)

set(DENSE_BLAS_DEFINES "")

# CUDA backend (cuBLAS)
if(IREE_HAL_DRIVER_CUDA)
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    list(APPEND DENSE_BLAS_DEFINES IREE_DENSE_BLAS_HAVE_CUDA=1)
    list(APPEND DENSE_BLAS_DEPS CUDA::cublas)
    # TODO: Add CUDA-specific source files
    # list(APPEND DENSE_BLAS_SOURCES backends/cublas.c)
    message(STATUS "dense_blas: cuBLAS backend enabled")
  endif()
endif()

# HIP backend (rocBLAS)
if(IREE_HAL_DRIVER_HIP)
  find_package(rocblas)
  if(rocblas_FOUND)
    list(APPEND DENSE_BLAS_DEFINES IREE_DENSE_BLAS_HAVE_HIP=1)
    list(APPEND DENSE_BLAS_DEPS roc::rocblas)
    # TODO: Add HIP-specific source files
    # list(APPEND DENSE_BLAS_SOURCES backends/rocblas.c)
    message(STATUS "dense_blas: rocBLAS backend enabled")
  endif()
endif()

# Metal backend (MPS)
if(APPLE AND IREE_HAL_DRIVER_METAL)
  list(APPEND DENSE_BLAS_DEFINES IREE_DENSE_BLAS_HAVE_METAL=1)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  list(APPEND DENSE_BLAS_DEPS
    ${ACCELERATE_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    iree::hal::drivers::metal
  )
  # Metal-specific source files (Objective-C/C++)
  list(APPEND DENSE_BLAS_SOURCES
    hal_buffer_helper_metal.m
    gemm_metal.mm
  )
  message(STATUS "dense_blas: MPS backend enabled")
endif()

# OpenCL backend (CLBlast)
option(IREE_DENSE_BLAS_USE_OPENCL "Use OpenCL for dense BLAS (via CLBlast)" OFF)
if(IREE_DENSE_BLAS_USE_OPENCL)
  find_package(OpenCL)
  if(OpenCL_FOUND)
    # Enable OpenCL HAL buffer support
    list(APPEND DENSE_BLAS_DEFINES IREE_DENSE_BLAS_HAVE_OPENCL=1)
    list(APPEND DENSE_BLAS_DEPS OpenCL::OpenCL)
    message(STATUS "dense_blas: OpenCL buffer support enabled")

    # Try to find CLBlast for optimized BLAS operations
    find_package(CLBlast)
    if(CLBlast_FOUND)
      list(APPEND DENSE_BLAS_DEFINES IREE_DENSE_BLAS_HAVE_CLBLAST=1)
      list(APPEND DENSE_BLAS_DEPS clblast)
      # TODO: Add CLBlast-specific source files
      # list(APPEND DENSE_BLAS_SOURCES backends/clblast.c)
      message(STATUS "dense_blas: CLBlast backend enabled")
    else()
      message(STATUS "dense_blas: CLBlast not found, using OpenCL buffer support only")
    endif()
  else()
    message(WARNING "OpenCL not found, OpenCL backend disabled")
  endif()
endif()

# CPU fallback
if(APPLE)
  # Accelerate is always available on macOS
  list(APPEND DENSE_BLAS_DEFINES IREE_DENSE_BLAS_HAVE_ACCELERATE=1)
  if(NOT ACCELERATE_FRAMEWORK)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    list(APPEND DENSE_BLAS_DEPS ${ACCELERATE_FRAMEWORK})
  endif()
  # Add Accelerate backend source file
  list(APPEND DENSE_BLAS_SOURCES backends/accelerate.c)
  message(STATUS "dense_blas: Accelerate CPU fallback enabled")
else()
  # Try to find OpenBLAS
  find_package(OpenBLAS)
  if(OpenBLAS_FOUND)
    list(APPEND DENSE_BLAS_DEFINES IREE_DENSE_BLAS_HAVE_OPENBLAS=1)
    list(APPEND DENSE_BLAS_DEPS OpenBLAS::OpenBLAS)
    message(STATUS "dense_blas: OpenBLAS CPU fallback enabled")
  else()
    message(WARNING "No CPU BLAS library found, CPU fallback disabled")
  endif()
endif()

#-------------------------------------------------------------------------------
# Module library
#-------------------------------------------------------------------------------

set(DENSE_BLAS_HDRS
  "module.h"
  "hal_buffer_helper.h"
)

if(APPLE AND IREE_HAL_DRIVER_METAL)
  list(APPEND DENSE_BLAS_HDRS "gemm_metal.h")
endif()

iree_cc_library(
  NAME
    dense_blas
  HDRS
    ${DENSE_BLAS_HDRS}
  SRCS
    ${DENSE_BLAS_SOURCES}
  DEPS
    ${DENSE_BLAS_DEPS}
  DEFINES
    ${DENSE_BLAS_DEFINES}
  PUBLIC
)
