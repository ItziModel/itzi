#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


class BuildConfig:
    """Centralized build configuration management for itzi package.

    This class handles:
    - Detection of build mode (source vs wheel)
    - Platform and architecture detection
    - Compiler flag selection based on build mode and target platform
    """

    def __init__(self):
        self.is_wheel_build = os.getenv("ITZI_BDIST_WHEEL") is not None
        self.platform = self.detect_platform()
        self.architecture = self.detect_architecture()
        self.compiler_type = None  # Set during build
        self.base_compile_args_unix = ["-O3", "-w", "-fopenmp"]
        self.base_compile_args_macos = [
            "-O3",
            "-w",
            "-Xpreprocessor",
            "-fopenmp",
        ]
        self.base_compile_args_msvc = ["/openmp", "/Ox"]
        self.base_link_args_unix = ["-lgomp", "-fopenmp"]

    def detect_platform(self):
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

    def detect_architecture(self):
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

    def get_optimization_flags(self, compiler_type):
        """Return appropriate compiler flags based on build mode and platform"""
        self.compiler_type = compiler_type

        if self.is_wheel_build:
            return self._get_wheel_optimization_flags()
        else:
            return self._get_source_optimization_flags()

    def _get_source_optimization_flags(self):
        """Get optimization flags for source builds (aggressive, machine-specific)"""
        compile_args = []
        link_args = []

        if self.compiler_type == "msvc":
            # Conservative MSVC flags for source builds
            compile_args = self.base_compile_args_msvc
            link_args = []
        elif self.compiler_type == "mingw32":
            compile_args = self.base_compile_args_unix + ["-lgomp", "-lpthread", "-march=native"]
            link_args = ["-lgomp", "-lpthread"]
        elif self.compiler_type == "unix":
            if self.platform == "macos":
                # macOS specific handling
                compile_args = self.base_compile_args_macos + ["-march=native"]
                link_args = ["-lomp"]
            else:
                # Linux and other Unix systems
                compile_args = self.base_compile_args_unix + ["-march=native"]
                link_args = self.base_link_args_unix

        return compile_args, link_args

    def _get_wheel_optimization_flags(self):
        """Get optimization flags for wheel builds (conservative, architecture-specific)"""
        compile_args = []
        link_args = []

        if self.compiler_type == "msvc":
            if self.architecture == "x86_64":
                compile_args = self.base_compile_args_msvc + ["/arch:AVX2"]
            elif self.architecture == "arm64":
                compile_args = self.base_compile_args_msvc + ["/arch:armv8.2"]
            else:
                compile_args = self.base_compile_args_msvc
            link_args = []

        elif self.compiler_type == "mingw32":
            if self.architecture == "x86_64":
                compile_args = self.base_compile_args_unix + [
                    "-lgomp",
                    "-lpthread",
                    "-march=x86-64-v3",
                ]
            elif self.architecture == "arm64":
                compile_args = self.base_compile_args_unix + ["-march=armv8-a+simd"]
            else:
                compile_args = self.base_compile_args_unix
            link_args = ["-lgomp", "-lpthread"]

        elif self.compiler_type == "unix":
            if self.platform == "macos":
                if self.architecture == "arm64":
                    compile_args = self.base_compile_args_macos + ["-march=armv8-a+simd"]
                else:
                    compile_args = self.base_compile_args_macos
                link_args = ["-lomp"]
            else:
                # Linux and other Unix systems
                if self.architecture == "x86_64":
                    compile_args = self.base_compile_args_unix + ["-march=x86-64-v3"]
                elif self.architecture == "arm64":
                    compile_args = self.base_compile_args_unix + ["-march=armv8-a+simd"]
                else:
                    compile_args = self.base_compile_args_unix
                link_args = self.base_link_args_unix

        return compile_args, link_args


# Legacy compiler options for backward compatibility
copt = {
    "msvc": ["/openmp", "/Ox"],
    "mingw32": ["-O3", "-w", "-fopenmp", "-lgomp", "-lpthread"],
    "unix": ["-O3", "-w", "-fopenmp"],
}
lopt = {"mingw32": ["-lgomp", "-lpthread"], "unix": ["-lgomp", "-fopenmp"]}

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
                if compiler == "unix" and platform.system() == "Darwin":
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
                if compiler in ["msvc", "mingw32"]:
                    ext.extra_compile_args = copt[compiler]
                    ext.extra_link_args = lopt.get(compiler)
                if compiler in ["unix"]:
                    if platform.system() == "Darwin":
                        ext.extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
                        ext.extra_link_args.append("-lomp")
                        for path in macos_includes:
                            if os.path.exists(path):
                                ext.include_dirs.append(path)
                        for path in macos_libs:
                            if os.path.exists(path):
                                ext.library_dirs.append(path)
                    else:
                        ext.extra_compile_args = copt[compiler]
                        ext.extra_link_args = lopt[compiler]

        build_ext.build_extensions(self)


extensions = [
    Extension("itzi.flow", sources=["src/itzi/flow.pyx"]),
    Extension("itzi.rastermetrics", sources=["src/itzi/rastermetrics.pyx"]),
]
setup(
    ext_modules=cythonize(extensions, nthreads=4),
    cmdclass={"build_ext": build_ext_compiler_check},
)
