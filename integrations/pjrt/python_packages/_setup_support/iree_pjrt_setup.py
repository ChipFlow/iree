# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
PJRT plugin build support using setuptools SubCommand protocol.

This module provides a cmake-based build sub-command that follows the
setuptools.command.build.SubCommand protocol for proper integration
with modern Python packaging tools like uv.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional

from distutils.command.install import install
from setuptools import Command
from setuptools.command.build import build as _build

THIS_DIR = Path(__file__).parent.resolve()
IREE_PJRT_DIR = THIS_DIR.parent.parent


def load_version_info() -> dict:
    version_info_file = IREE_PJRT_DIR / "version_info.json"
    try:
        with open(version_info_file, "rt") as f:
            return json.load(f)
    except FileNotFoundError:
        print("version_info.json not found. Using defaults", file=sys.stderr)
        return {}


version_info = load_version_info()
PACKAGE_SUFFIX = version_info.get("package-suffix") or ""
PACKAGE_VERSION = version_info.get("package-version") or "0.1dev1"


def get_env_cmake_option(name: str, default_value: bool = False) -> str:
    """Get cmake option from environment variable."""
    svalue = os.getenv(name)
    if not svalue:
        svalue = "ON" if default_value else "OFF"
    return f"-D{name}={svalue}"


def add_env_cmake_setting(args: list, env_name: str, cmake_name: Optional[str] = None):
    """Add cmake setting from environment variable if set."""
    svalue = os.getenv(env_name)
    if svalue is not None:
        if not cmake_name:
            cmake_name = env_name
        args.append(f"-D{cmake_name}={svalue}")


def maybe_nuke_cmake_cache(cmake_build_dir: Path):
    """Remove cmake cache if ninja path changed."""
    ninja_path = ""
    try:
        import ninja
        ninja_path = ninja.__file__
    except ModuleNotFoundError:
        pass

    expected_stamp_contents = f"{ninja_path}"
    ninja_stamp_file = cmake_build_dir / "ninja_stamp.txt"

    if ninja_stamp_file.exists():
        actual_stamp_contents = ninja_stamp_file.read_text()
        if actual_stamp_contents == expected_stamp_contents:
            return

    cmake_cache_file = cmake_build_dir / "CMakeCache.txt"
    if cmake_cache_file.exists():
        print("Removing CMakeCache.txt because Ninja path changed", file=sys.stderr)
        cmake_cache_file.unlink()

    ninja_stamp_file.write_text(expected_stamp_contents)


def check_cmake_config_changed(cmake_build_dir: Path, cmake_args: List[str]) -> bool:
    """
    Check if cmake configuration has changed since last build.

    Returns True if reconfiguration is needed.
    """
    # Create a hash of the cmake args to detect config changes
    config_str = "\n".join(sorted(cmake_args))
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    config_stamp_file = cmake_build_dir / "cmake_config_stamp.txt"
    cmake_cache_file = cmake_build_dir / "CMakeCache.txt"

    if not cmake_cache_file.exists():
        # No cache, need to configure
        config_stamp_file.write_text(config_hash)
        return True

    if config_stamp_file.exists():
        stored_hash = config_stamp_file.read_text().strip()
        if stored_hash == config_hash:
            return False  # Config unchanged

    # Config changed - remove cache and update stamp
    print("Removing CMakeCache.txt because cmake configuration changed", file=sys.stderr)
    cmake_cache_file.unlink()
    config_stamp_file.write_text(config_hash)
    return True


class build_cmake(Command):
    """
    Build sub-command for cmake-based PJRT plugins.

    Implements the setuptools.command.build.SubCommand protocol for proper
    integration with editable installs and modern packaging tools.
    """

    description = "build cmake-based PJRT plugin"

    # SubCommand protocol attributes
    editable_mode: bool = False
    build_lib: Optional[str] = None

    # Plugin-specific settings (set by subclass)
    cmake_source_dir: str = ""
    cmake_build_dir: str = ""
    extra_cmake_args: tuple = ()
    output_dir: str = ""  # Relative path under build_lib for outputs
    output_files: List[str] = []  # List of output file patterns

    user_options = [
        ("build-lib=", "b", "directory to build to"),
        ("editable-mode", "e", "enable editable mode"),
    ]

    def initialize_options(self):
        self.build_lib = None
        self.editable_mode = False

    def finalize_options(self):
        # Get build_lib from parent build command
        self.set_undefined_options("build", ("build_lib", "build_lib"))

    def run(self):
        """Execute the cmake build."""
        if not self.cmake_source_dir or not self.cmake_build_dir:
            raise ValueError("cmake_source_dir and cmake_build_dir must be set")

        cmake_build_path = Path(self.cmake_build_dir)
        cmake_build_path.mkdir(parents=True, exist_ok=True)

        self._run_cmake_configure(cmake_build_path)
        self._run_cmake_build(cmake_build_path)

    def _run_cmake_configure(self, cmake_build_dir: Path):
        """Run cmake configure step."""
        subprocess.check_call(["cmake", "--version"])
        maybe_nuke_cmake_cache(cmake_build_dir)

        cfg = os.getenv("IREE_CMAKE_BUILD_TYPE", "Release")

        # Find ninja - prefer system ninja over Python wheel to avoid temp path issues
        ninja_path = None
        try:
            result = subprocess.run(["which", "ninja"], capture_output=True, text=True)
            if result.returncode == 0:
                ninja_path = result.stdout.strip()
        except Exception:
            pass

        cmake_args = [
            "-GNinja",
            "--log-level=VERBOSE",
            get_env_cmake_option("IREE_BUILD_COMPILER", default_value=True),
            get_env_cmake_option("IREE_BUILD_COMPILER_SHARED_LIBS", default_value=True),
            "-DIREE_BUILD_SAMPLES=OFF",
            "-DIREE_BUILD_TESTS=OFF",
            "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        # Use system ninja if available to avoid temp path issues
        if ninja_path:
            cmake_args.append(f"-DCMAKE_MAKE_PROGRAM={ninja_path}")

        cmake_args.extend(list(self.extra_cmake_args))

        # Add environment-based cmake settings
        add_env_cmake_setting(cmake_args, "IREE_TRACING_PROVIDER")
        add_env_cmake_setting(cmake_args, "IREE_TRACING_PROVIDER_H")
        add_env_cmake_setting(cmake_args, "CMAKE_OSX_ARCHITECTURES")
        add_env_cmake_setting(cmake_args, "MACOSX_DEPLOYMENT_TARGET", "CMAKE_OSX_DEPLOYMENT_TARGET")
        add_env_cmake_setting(cmake_args, "CMAKE_SYSTEM_PROCESSOR")
        add_env_cmake_setting(cmake_args, "CMAKE_C_COMPILER_LAUNCHER")
        add_env_cmake_setting(cmake_args, "CMAKE_CXX_COMPILER_LAUNCHER")
        add_env_cmake_setting(cmake_args, "CMAKE_C_FLAGS")
        add_env_cmake_setting(cmake_args, "CMAKE_CXX_FLAGS")

        # Check if config changed (handles cache removal if needed)
        needs_configure = check_cmake_config_changed(cmake_build_dir, cmake_args)

        cmake_cache_file = cmake_build_dir / "CMakeCache.txt"
        if needs_configure or not cmake_cache_file.exists():
            print(f"Configuring with: {cmake_args}", file=sys.stderr)
            subprocess.check_call(
                ["cmake", str(IREE_PJRT_DIR)] + cmake_args,
                cwd=cmake_build_dir,
            )
        else:
            print("Not re-configuring (config unchanged)", file=sys.stderr)

    def _run_cmake_build(self, cmake_build_dir: Path):
        """Run cmake build step."""
        subprocess.check_call(["cmake", "--build", "."], cwd=cmake_build_dir)
        print("Build complete.", file=sys.stderr)

        # Build compiler shared library for runtime JIT
        print("Building compiler shared library...", file=sys.stderr)
        try:
            subprocess.check_call(
                ["cmake", "--build", ".", "--target", "iree_compiler_API_SharedImpl"],
                cwd=cmake_build_dir,
            )
            print("Compiler shared library built.", file=sys.stderr)
        except subprocess.CalledProcessError:
            print(
                "WARNING: Failed to build compiler shared library. "
                "PJRT plugin may not work without iree-base-compiler pip package.",
                file=sys.stderr,
            )

    def get_source_files(self) -> List[str]:
        """Return list of source files needed for the build."""
        # For cmake builds, this would be extensive - return empty for now
        # as sdist is typically handled separately for cmake projects
        return []

    def get_outputs(self) -> List[str]:
        """Return list of output files produced by the build."""
        if not self.build_lib or not self.output_dir:
            return []

        outputs = []
        output_path = Path(self.build_lib) / self.output_dir
        if output_path.exists():
            for pattern in self.output_files:
                outputs.extend(str(p) for p in output_path.glob(pattern))
        return outputs

    def get_output_mapping(self) -> Dict[str, str]:
        """Return mapping of output files to source files."""
        # For cmake-generated files, there's no 1:1 source mapping
        # Return empty dict - files are generated, not copied
        return {}


# Note: build_cmake is registered via cmdclass in each plugin's setup.py
# Do NOT auto-register here to avoid conflicts


# Force platform specific wheel
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            python, abi = "py3", "none"
            return python, abi, plat

except ImportError:
    bdist_wheel = None


def create_cmake_build_class(
    cmake_source_dir: str,
    cmake_build_dir: str,
    extra_cmake_args: tuple = (),
    output_dir: str = "",
    output_files: Optional[List[str]] = None,
):
    """
    Factory function to create a configured build_cmake subclass.

    Args:
        cmake_source_dir: Path to cmake source directory
        cmake_build_dir: Path to cmake build directory
        extra_cmake_args: Additional cmake arguments
        output_dir: Relative path under build_lib for outputs
        output_files: List of glob patterns for output files

    Returns:
        Configured build_cmake subclass
    """
    if output_files is None:
        output_files = ["*.so", "*.dylib", "*.pyd"]

    class ConfiguredBuildCMake(build_cmake):
        pass

    ConfiguredBuildCMake.cmake_source_dir = cmake_source_dir
    ConfiguredBuildCMake.cmake_build_dir = cmake_build_dir
    ConfiguredBuildCMake.extra_cmake_args = extra_cmake_args
    ConfiguredBuildCMake.output_dir = output_dir
    ConfiguredBuildCMake.output_files = output_files

    return ConfiguredBuildCMake


# Legacy compatibility - keep for existing setup.py files that haven't migrated
class BaseCMakeBuildPy:
    """
    DEPRECATED: Use create_cmake_build_class() instead.

    This class is kept for backward compatibility but will be removed
    in a future version.
    """

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "BaseCMakeBuildPy is deprecated. Use create_cmake_build_class() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class PjrtPluginBuild(_build):
    """Custom build command that runs cmake sub-command."""

    # Register build_cmake as a sub-command that always runs
    # This ensures it's called during both regular and editable installs
    sub_commands = [("build_cmake", lambda self: True)] + _build.sub_commands

    def run(self):
        # sub_commands handles calling build_cmake, just run parent
        super().run()


def populate_built_package(abs_dir: str):
    """Ensure directory and __init__.py exist for setuptools discovery."""
    path = Path(abs_dir)
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    if not init_file.exists():
        init_file.touch()


# Force installation into platlib for platform-specific wheels
class platlib_install(install):
    """Force installation into platlib for platform-specific binaries."""

    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib
