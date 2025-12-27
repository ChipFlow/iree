# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

option(IREE_EXTERNAL_SPARSE_SOLVER "Enable external sparse solver module (BaSpaCho)" OFF)

if(NOT IREE_EXTERNAL_SPARSE_SOLVER)
  return()
endif()

#-------------------------------------------------------------------------------
# BaSpaCho dependency
#-------------------------------------------------------------------------------

find_package(BaSpaCho QUIET)
if(NOT BaSpaCho_FOUND)
  message(STATUS "sparse_solver: BaSpaCho not found via find_package, checking for source")

  # Allow building from source if BASPACHO_SOURCE_DIR is set.
  if(DEFINED BASPACHO_SOURCE_DIR)
    # Enable BaSpaCho Metal backend when building for Metal.
    # BaSpaCho has been modified to not use enable_language(OBJCXX) which
    # would conflict with IREE's Objective-C Metal driver.
    if(APPLE AND IREE_HAL_DRIVER_METAL)
      set(BASPACHO_USE_METAL ON CACHE BOOL "Enable Metal backend in BaSpaCho" FORCE)
    endif()
    # Disable tests and examples to speed up build
    set(BASPACHO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BASPACHO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${BASPACHO_SOURCE_DIR} ${CMAKE_BINARY_DIR}/baspacho)
    # BaSpaCho library target is always "BaSpaCho" (bundled for static builds)
    set(BASPACHO_LIBRARIES BaSpaCho)
    # BaSpaCho headers are at baspacho/baspacho/*.h, include repo root
    set(BASPACHO_INCLUDE_DIRS ${BASPACHO_SOURCE_DIR})
    # Eigen is fetched by BaSpaCho into _deps/eigen-src (at cmake binary root)
    set(EIGEN_FETCHCONTENT_DIR ${CMAKE_BINARY_DIR}/_deps/eigen-src)
    if(EXISTS ${EIGEN_FETCHCONTENT_DIR})
      list(APPEND BASPACHO_INCLUDE_DIRS ${EIGEN_FETCHCONTENT_DIR})
      message(STATUS "sparse_solver: Using Eigen from ${EIGEN_FETCHCONTENT_DIR}")
    elseif(DEFINED eigen_SOURCE_DIR)
      list(APPEND BASPACHO_INCLUDE_DIRS ${eigen_SOURCE_DIR})
      message(STATUS "sparse_solver: Using Eigen from ${eigen_SOURCE_DIR}")
    else()
      message(WARNING "sparse_solver: Eigen not found - BaSpaCho may fail to compile")
    endif()
    message(STATUS "sparse_solver: Building BaSpaCho from source")
  else()
    message(FATAL_ERROR
      "BaSpaCho not found. Please either:\n"
      "  1. Install BaSpaCho and ensure find_package can locate it, OR\n"
      "  2. Set BASPACHO_SOURCE_DIR to the BaSpaCho source directory")
  endif()
endif()

#-------------------------------------------------------------------------------
# Backend configuration
#-------------------------------------------------------------------------------

set(SPARSE_SOLVER_SOURCES
  module.c
  baspacho_wrapper.cpp
)

set(SPARSE_SOLVER_DEPS
  iree::base
  iree::hal
  iree::modules::hal
  iree::vm
)

set(SPARSE_SOLVER_DEFINES "")

# Note: BaSpaCho is linked separately after iree_cc_library because
# iree_cc_library doesn't properly handle external cmake targets.
# See target_link_libraries call after iree_cc_library below.

# Add BaSpaCho include directories.
if(BASPACHO_INCLUDE_DIRS)
  include_directories(${BASPACHO_INCLUDE_DIRS})
elseif(DEFINED BASPACHO_SOURCE_DIR)
  include_directories(${BASPACHO_SOURCE_DIR})
endif()

# Metal backend (requires BaSpaCho metal-backend branch).
if(APPLE AND IREE_HAL_DRIVER_METAL)
  list(APPEND SPARSE_SOLVER_DEFINES IREE_SPARSE_SOLVER_HAVE_METAL=1)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  list(APPEND SPARSE_SOLVER_DEPS ${METAL_FRAMEWORK})
  message(STATUS "sparse_solver: Metal backend enabled")
endif()

# CUDA backend.
if(IREE_HAL_DRIVER_CUDA)
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    list(APPEND SPARSE_SOLVER_DEFINES IREE_SPARSE_SOLVER_HAVE_CUDA=1)
    list(APPEND SPARSE_SOLVER_DEPS CUDA::cudart CUDA::cublas)
    message(STATUS "sparse_solver: CUDA backend enabled")
  endif()
endif()

# OpenCL backend (generic fallback).
option(IREE_SPARSE_SOLVER_USE_OPENCL "Use OpenCL backend for sparse solver" OFF)
if(IREE_SPARSE_SOLVER_USE_OPENCL)
  find_package(OpenCL)
  find_package(CLBlast)
  if(OpenCL_FOUND AND CLBlast_FOUND)
    list(APPEND SPARSE_SOLVER_DEFINES IREE_SPARSE_SOLVER_HAVE_OPENCL=1)
    list(APPEND SPARSE_SOLVER_DEPS OpenCL::OpenCL clblast)
    message(STATUS "sparse_solver: OpenCL backend enabled")
  else()
    message(WARNING "OpenCL or CLBlast not found, OpenCL backend disabled")
  endif()
endif()

#-------------------------------------------------------------------------------
# Exception-safe wrapper compilation
#-------------------------------------------------------------------------------
# BaSpaCho uses C++ exceptions and RTTI internally. IREE builds with
# -fno-exceptions -fno-rtti by default. We compile baspacho_wrapper.cpp
# with exceptions and RTTI enabled, with all exceptions caught at the
# C API boundary and converted to error codes.

set(BASPACHO_WRAPPER_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/baspacho_wrapper.cpp"
)

# Enable exceptions and RTTI for the wrapper file only.
# This overrides IREE's default -fno-exceptions -fno-rtti flags.
set_source_files_properties(${BASPACHO_WRAPPER_SOURCE}
  PROPERTIES
    COMPILE_OPTIONS "-fexceptions;-frtti"
)

#-------------------------------------------------------------------------------
# Module library
#-------------------------------------------------------------------------------

iree_cc_library(
  NAME
    sparse_solver
  HDRS
    "module.h"
    "baspacho_wrapper.h"
  SRCS
    ${SPARSE_SOLVER_SOURCES}
  DEPS
    ${SPARSE_SOLVER_DEPS}
  DEFINES
    ${SPARSE_SOLVER_DEFINES}
  PUBLIC
)

# Link BaSpaCho separately because iree_cc_library doesn't handle external
# cmake targets properly. Use the explicit library file path and add
# dependency on the bundling target to ensure proper build ordering.
if(DEFINED BASPACHO_SOURCE_DIR)
  # The bundled library is created by bundle_static_library in BaSpaCho
  set(BASPACHO_BUNDLED_LIB "${CMAKE_BINARY_DIR}/baspacho/baspacho/baspacho/libBaSpaCho.a")

  # Add the library directly by file path
  target_link_libraries(iree_modules_sparse_solver_sparse_solver
    PUBLIC "${BASPACHO_BUNDLED_LIB}")

  # Ensure the bundling is done before linking
  add_dependencies(iree_modules_sparse_solver_sparse_solver BaSpaCho_static)
endif()

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

if(IREE_BUILD_TESTS)
  # TODO: Add sparse solver tests.
endif()
