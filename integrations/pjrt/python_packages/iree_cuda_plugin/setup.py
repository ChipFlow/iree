#!/usr/bin/python3

# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Build configuration for iree-pjrt-plugin-cuda.

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
OUTPUT_DIR = CMAKE_BUILD_DIR / "python" / "iree" / "_pjrt_libs" / "cuda"

# Create the build_cmake subclass configured for cuda plugin
CudaBuildCMake = iree_pjrt_setup.create_cmake_build_class(
    cmake_source_dir=str(THIS_DIR.parent.parent.parent),  # integrations/pjrt
    cmake_build_dir=str(CMAKE_BUILD_DIR),
    extra_cmake_args=(
        "-DIREE_HAL_DRIVER_CUDA=ON",
    ),
    output_dir="iree/_pjrt_libs/cuda",
    output_files=["pjrt_plugin_iree_cuda.*", "*.so"],
)

# Ensure the build output directory exists for setuptools package discovery
iree_pjrt_setup.populate_built_package(str(OUTPUT_DIR))

setup(
    # Metadata comes from pyproject.toml
    packages=[
        "jax_plugins.iree_cuda",
        "iree._pjrt_libs.cuda",
    ],
    package_dir={
        "jax_plugins.iree_cuda": "jax_plugins/iree_cuda",
        "iree._pjrt_libs.cuda": "build/cmake/python/iree/_pjrt_libs/cuda",
    },
    package_data={
        "iree._pjrt_libs.cuda": ["pjrt_plugin_iree_cuda.*", "*.so"],
    },
    cmdclass={
        "build": iree_pjrt_setup.PjrtPluginBuild,
        "build_cmake": CudaBuildCMake,
        "bdist_wheel": iree_pjrt_setup.bdist_wheel,
    },
    zip_safe=False,
)
