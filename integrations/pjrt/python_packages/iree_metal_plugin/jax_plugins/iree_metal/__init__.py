# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
from pathlib import Path
import platform
import sys

import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)


def probe_iree_compiler_dylib() -> str:
    """Probes an installed iree.compiler for the compiler dylib.

    On macOS, also checks the IREE_PJRT_COMPILER_LIB_PATH environment variable.
    """
    # Check environment variable first (useful for development builds)
    env_path = os.environ.get("IREE_PJRT_COMPILER_LIB_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Fall back to probing installed iree.compiler
    try:
        from iree.compiler.api import ctypes_dl
        return ctypes_dl._probe_iree_compiler_dylib()
    except ImportError:
        logger.warning(
            "Could not import iree.compiler. Set IREE_PJRT_COMPILER_LIB_PATH "
            "environment variable to point to libIREECompiler.dylib"
        )
        raise


def initialize():
    # Metal is only available on macOS
    if platform.system() != "Darwin":
        logger.warning(
            f"Metal PJRT plugin is only available on macOS, "
            f"but running on {platform.system()}"
        )
        return

    import iree._pjrt_libs.metal as lib_package

    # On macOS, the library is a .dylib
    path = Path(lib_package.__file__).resolve().parent / "pjrt_plugin_iree_metal.dylib"
    if not path.exists():
        # Also try .so extension for compatibility
        path = Path(lib_package.__file__).resolve().parent / "pjrt_plugin_iree_metal.so"
    if not path.exists():
        logger.warning(
            f"WARNING: Native library {path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )
    xb.register_plugin(
        "iree_metal",
        priority=500,
        library_path=str(path),
        options={
            "COMPILER_LIB_PATH": str(probe_iree_compiler_dylib()),
        },
    )
