# Copyright 2024 IREE Metal PJRT Plugin Contributors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Collect optional module dependencies and defines
set(_OPTIONAL_DEPS "")
set(_OPTIONAL_DEFINES "")
if(IREE_EXTERNAL_DENSE_BLAS)
  list(APPEND _OPTIONAL_DEPS iree::modules::dense_blas::dense_blas)
  list(APPEND _OPTIONAL_DEFINES IREE_DENSE_BLAS_HAVE_METAL=1)
endif()
if(IREE_EXTERNAL_SPARSE_SOLVER)
  list(APPEND _OPTIONAL_DEPS iree::modules::sparse_solver::sparse_solver)
  list(APPEND _OPTIONAL_DEFINES IREE_SPARSE_SOLVER_HAVE_METAL=1)
endif()

iree_cc_library(
  NAME
    client
  HDRS
    "client.h"
  SRCS
    "client.cc"
  DEPS
    iree_pjrt::common
    iree::hal::drivers::metal
    iree::hal::drivers::metal::registration
    ${_OPTIONAL_DEPS}
  DEFINES
    ${_OPTIONAL_DEFINES}
)

iree_cc_library(
  SHARED
  NAME
    dylib
  DEFINES
    PJRT_PLUGIN_BUILDING_LIBRARY
  SRCS
    "dylib_entry_point.cc"
  DEPS
    ::client
    iree_pjrt::common
    iree_pjrt::common::dylib_platform
)

set(_NATIVE_PYTHON_DIR "${IREE_PJRT_PYTHON_BINARY_DIR}/iree/_pjrt_libs/metal")
file(WRITE "${_NATIVE_PYTHON_DIR}/__init__.py" "")
set_target_properties(iree_pjrt_metal_dylib
  PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_NAME pjrt_plugin_iree_metal
    RUNTIME_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}"
)

if(NOT IREE_ENABLE_ASAN)
  target_link_options(iree_pjrt_metal_dylib PRIVATE "-Wl,-undefined,error")
endif()
