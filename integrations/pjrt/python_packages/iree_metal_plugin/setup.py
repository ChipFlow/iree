#!/usr/bin/python3

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Build configuration for iree-pjrt-plugin-metal.

Metadata is in pyproject.toml; this file handles cmake build orchestration
using the setuptools SubCommand protocol for proper uv/pip compatibility.
"""

import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(THIS_DIR.parent / "_setup_support"))

import iree_pjrt_setup
from setuptools import setup

CMAKE_BUILD_DIR = THIS_DIR / "build" / "cmake"
OUTPUT_DIR = CMAKE_BUILD_DIR / "python" / "iree" / "_pjrt_libs" / "metal"

# Create the build_cmake subclass configured for metal plugin
MetalBuildCMake = iree_pjrt_setup.create_cmake_build_class(
    cmake_source_dir=str(THIS_DIR.parent.parent.parent),  # integrations/pjrt
    cmake_build_dir=str(CMAKE_BUILD_DIR),
    extra_cmake_args=(
        "-DIREE_HAL_DRIVER_METAL=ON",
        "-DIREE_EXTERNAL_DENSE_BLAS=ON",
        # Sparse solver module (BaSpaCho integration)
        # baspacho_wrapper.cpp is compiled with -fexceptions -frtti to handle
        # BaSpaCho's C++ exceptions, with all exceptions caught at the C API boundary.
        "-DIREE_EXTERNAL_SPARSE_SOLVER=ON",
        # Use the reference folder's baspacho which has Metal backend support
        f"-DBASPACHO_SOURCE_DIR={THIS_DIR.parent.parent.parent.parent.parent / 'baspacho'}",
        "-DBASPACHO_USE_CUBLAS=OFF",
        "-DBLA_VENDOR=Apple",
    ),
    output_dir="iree/_pjrt_libs/metal",
    output_files=["pjrt_plugin_iree_metal.*", "*.dylib"],
)

# Ensure the build output directory exists for setuptools package discovery
iree_pjrt_setup.populate_built_package(str(OUTPUT_DIR))

setup(
    # Metadata comes from pyproject.toml
    packages=[
        "jax_plugins.iree_metal",
        "iree._pjrt_libs.metal",
    ],
    package_dir={
        "jax_plugins.iree_metal": "jax_plugins/iree_metal",
        "iree._pjrt_libs.metal": "build/cmake/python/iree/_pjrt_libs/metal",
    },
    package_data={
        "iree._pjrt_libs.metal": ["pjrt_plugin_iree_metal.*", "*.dylib"],
    },
    cmdclass={
        "build": iree_pjrt_setup.PjrtPluginBuild,
        "build_cmake": MetalBuildCMake,
        "bdist_wheel": iree_pjrt_setup.bdist_wheel,
    },
    zip_safe=False,
)
