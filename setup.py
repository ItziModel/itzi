#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


def parse_env_bool(name: str) -> bool | None:
    """Return a boolean override from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name!r} must be one of 1/0, true/false, yes/no, on/off"
    )


def parse_opt_level(name: str, default: str) -> str:
    """Return a validated optimization level from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().upper()
    if normalized not in {"O0", "O1", "O2", "O3"}:
        raise ValueError(f"Environment variable {name!r} must be one of O0, O1, O2, O3")

    return normalized


class BuildConfig:
    """Centralized build configuration management for itzi package.

    This class handles:
    - Detection of build mode (source vs wheel)
    - Platform and architecture detection
    - Compiler flag selection based on build mode and target platform
    """

    def __init__(self) -> None:
        self.is_wheel_build = os.getenv("ITZI_BDIST_WHEEL") is not None
        self.platform = self.detect_platform()
        self.architecture = self.detect_architecture()
        self.openmp_override = parse_env_bool("ITZI_USE_OPENMP")
        self.opt_level = parse_opt_level("ITZI_OPT_LEVEL", default="O3")
        self.disable_vectorize = bool(parse_env_bool("ITZI_DISABLE_VECTORIZE"))
        self.use_openmp = self.detect_openmp_usage()
        self.compiler_type = None  # Set during build
        self.base_compile_args = [f"-{self.opt_level}", "-w"]
        self.openmp_compile_args_unix = ["-fopenmp"]
        self.openmp_compile_args_macos = ["-Xpreprocessor", "-fopenmp"]
        self.base_compile_args_msvc = ["/Ox"]
        self.openmp_compile_args_msvc = ["/openmp"]
        self.base_link_args_unix = ["-lgomp"]

    def detect_openmp_usage(self) -> bool:
        """Return whether OpenMP should be enabled for this build."""
        if self.openmp_override is not None:
            return self.openmp_override

        if self.is_wheel_build and self.platform == "macos" and self.architecture == "arm64":
            return False

        return True

    def get_extra_unix_compile_args(self) -> list[str]:
        """Return optional extra compile arguments for Unix-like compilers."""
        extra_args: list[str] = []
        # These flags are Clang-specific and are only intended for macOS CI experiments.
        if self.disable_vectorize and self.platform == "macos":
            extra_args.extend(["-fno-vectorize", "-fno-slp-vectorize"])
        return extra_args

    def detect_platform(self) -> str:
        """Detect current platform (Linux, Windows, macOS)"""
        system = platform.system()
        if system == "Linux":
            return "linux"
        elif system == "Windows":
            return "windows"
        elif system == "Darwin":
            return "macos"
        else:
            return "unknown"

    def detect_architecture(self) -> str:
        """Detect current architecture (x86_64, ARM64, etc.)"""
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            return "x86_64"
        elif machine in ("arm64", "aarch64"):
            return "arm64"
        elif machine.startswith("arm"):
            return "arm"
        else:
            return machine

    def get_optimization_flags(self, compiler_type: str) -> tuple[list[str], list[str]]:
        """Return appropriate compiler flags based on build mode and platform"""
        self.compiler_type = compiler_type

        if self.is_wheel_build:
            return self._get_wheel_optimization_flags()
        else:
            return self._get_source_optimization_flags()

    def _get_source_optimization_flags(self) -> tuple[list[str], list[str]]:
        """Get optimization flags for source builds (aggressive, machine-specific)"""
        compile_args = []
        link_args = []

        if self.compiler_type == "msvc":
            # Conservative MSVC flags for source builds
            compile_args = self.base_compile_args_msvc.copy()
            if self.use_openmp:
                compile_args.extend(self.openmp_compile_args_msvc)
            link_args = []
        elif self.compiler_type == "mingw32":
            compile_args = self.base_compile_args.copy()
            compile_args.extend(self.get_extra_unix_compile_args())
            if self.use_openmp:
                compile_args.extend(self.openmp_compile_args_unix)
                compile_args.append("-lgomp")
            compile_args.extend(["-lpthread", "-march=native"])
            link_args = ["-lpthread"]
            if self.use_openmp:
                link_args.extend(self.base_link_args_unix)
        elif self.compiler_type == "unix":
            if self.platform == "macos":
                # macOS specific handling
                compile_args = self.base_compile_args.copy()
                compile_args.extend(self.get_extra_unix_compile_args())
                if self.use_openmp:
                    compile_args.extend(self.openmp_compile_args_macos)
                compile_args.append("-march=native")
                link_args = ["-lomp"] if self.use_openmp else []
            else:
                # Linux and other Unix systems
                compile_args = self.base_compile_args.copy()
                compile_args.extend(self.get_extra_unix_compile_args())
                if self.use_openmp:
                    compile_args.extend(self.openmp_compile_args_unix)
                compile_args.append("-march=native")
                link_args = self.base_link_args_unix.copy() if self.use_openmp else []

        return compile_args, link_args

    def _get_wheel_optimization_flags(self) -> tuple[list[str], list[str]]:
        """Get optimization flags for wheel builds (conservative, architecture-specific)"""
        compile_args = []
        link_args = []

        if self.compiler_type == "msvc":
            compile_args = self.base_compile_args_msvc.copy()
            if self.use_openmp:
                compile_args.extend(self.openmp_compile_args_msvc)
            if self.architecture == "x86_64":
                compile_args.append("/arch:AVX2")
            elif self.architecture == "arm64":
                compile_args.append("/arch:armv8.2")
            link_args = []

        elif self.compiler_type == "mingw32":
            compile_args = self.base_compile_args.copy()
            compile_args.extend(self.get_extra_unix_compile_args())
            if self.use_openmp:
                compile_args.extend(self.openmp_compile_args_unix)
            if self.architecture == "x86_64":
                if self.use_openmp:
                    compile_args.append("-lgomp")
                compile_args.extend(["-lpthread", "-march=x86-64-v3"])
            elif self.architecture == "arm64":
                compile_args.append("-march=armv8-a+simd")
            else:
                if self.use_openmp:
                    compile_args.append("-lgomp")
                compile_args.append("-lpthread")
            link_args = ["-lpthread"]
            if self.use_openmp:
                link_args.extend(self.base_link_args_unix)

        elif self.compiler_type == "unix":
            if self.platform == "macos":
                compile_args = self.base_compile_args.copy()
                compile_args.extend(self.get_extra_unix_compile_args())
                if self.use_openmp:
                    compile_args.extend(self.openmp_compile_args_macos)
                if self.architecture == "arm64":
                    compile_args.append("-march=armv8-a+simd")
                link_args = ["-lomp"] if self.use_openmp else []
            else:
                # Linux and other Unix systems
                compile_args = self.base_compile_args.copy()
                compile_args.extend(self.get_extra_unix_compile_args())
                if self.use_openmp:
                    compile_args.extend(self.openmp_compile_args_unix)
                if self.architecture == "x86_64":
                    compile_args.append("-march=x86-64-v3")
                elif self.architecture == "arm64":
                    compile_args.append("-march=armv8-a+simd")
                link_args = self.base_link_args_unix.copy() if self.use_openmp else []

        return compile_args, link_args


macos_includes = [
    "/opt/homebrew/include",
    "/usr/local/include",
    "/opt/homebrew/opt/llvm/include",
    "/opt/homebrew/opt/libomp/include",
]
macos_libs = [
    "/opt/homebrew/lib",
    "/usr/local/lib",
    "/opt/homebrew/opt/llvm/lib",
    "/opt/homebrew/opt/libomp/lib",
]


class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        build_config = BuildConfig()
        compiler = self.compiler.compiler_type
        print(f"Compiler detected: {compiler}")
        print(f"Build mode: {'wheel' if build_config.is_wheel_build else 'source'}")
        print(f"Platform: {build_config.platform}")
        print(f"Architecture: {build_config.architecture}")
        print(f"Optimization level: {build_config.opt_level}")
        print(f"Vectorization disabled: {build_config.disable_vectorize}")
        print(f"OpenMP enabled: {build_config.use_openmp}")

        # Get optimization flags from BuildConfig
        try:
            compile_args, link_args = build_config.get_optimization_flags(compiler)
            print("Using optimized build configuration")
            print(f"Compile args: {compile_args}")
            print(f"Link args: {link_args}")

            for ext in self.extensions:
                # Apply optimized flags
                ext.extra_compile_args = compile_args
                ext.extra_link_args = link_args

                # Add macOS-specific include and library paths if needed
                if (
                    compiler == "unix"
                    and platform.system() == "Darwin"
                    and build_config.use_openmp
                ):
                    for path in macos_includes:
                        if os.path.exists(path):
                            ext.include_dirs.append(path)
                            print(f"{path} added to include_dirs")
                    for path in macos_libs:
                        if os.path.exists(path):
                            ext.library_dirs.append(path)
                            print(f"{path} added to library_dirs")

        except Exception as e:
            print(
                f"Warning: Failed to get optimized flags ({e}), falling back to legacy configuration"
            )
            # Fallback to legacy system
            for ext in self.extensions:
                if compiler == "msvc":
                    ext.extra_compile_args = build_config.base_compile_args_msvc.copy()
                    if build_config.use_openmp:
                        ext.extra_compile_args.extend(build_config.openmp_compile_args_msvc)
                    ext.extra_link_args = []
                elif compiler == "mingw32":
                    ext.extra_compile_args = build_config.base_compile_args.copy()
                    ext.extra_compile_args.extend(build_config.get_extra_unix_compile_args())
                    if build_config.use_openmp:
                        ext.extra_compile_args.extend(build_config.openmp_compile_args_unix)
                        ext.extra_compile_args.append("-lgomp")
                    ext.extra_compile_args.append("-lpthread")
                    ext.extra_link_args = ["-lpthread"]
                    if build_config.use_openmp:
                        ext.extra_link_args.extend(build_config.base_link_args_unix)
                elif compiler == "unix":
                    ext.extra_compile_args = build_config.base_compile_args.copy()
                    ext.extra_compile_args.extend(build_config.get_extra_unix_compile_args())
                    if platform.system() == "Darwin":
                        if build_config.use_openmp:
                            ext.extra_compile_args.extend(build_config.openmp_compile_args_macos)
                            ext.extra_link_args = ["-lomp"]
                            for path in macos_includes:
                                if os.path.exists(path):
                                    ext.include_dirs.append(path)
                            for path in macos_libs:
                                if os.path.exists(path):
                                    ext.library_dirs.append(path)
                        else:
                            ext.extra_link_args = []
                    else:
                        if build_config.use_openmp:
                            ext.extra_compile_args.extend(build_config.openmp_compile_args_unix)
                            ext.extra_link_args = build_config.base_link_args_unix.copy()
                        else:
                            ext.extra_link_args = []

        build_ext.build_extensions(self)


extensions = [
    Extension("itzi.flow", sources=["src/itzi/flow.pyx"]),
    Extension("itzi.rastermetrics", sources=["src/itzi/rastermetrics.pyx"]),
]
setup(
    ext_modules=cythonize(extensions, nthreads=4),
    cmdclass={"build_ext": build_ext_compiler_check},
)
