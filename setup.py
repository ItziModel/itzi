#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext


# Set arguments according to compiler
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
        compiler = self.compiler.compiler_type
        print(f"{compiler=}")
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
                    # Add homebrew library directories
                    for path in macos_libs:
                        if os.path.exists(path):
                            ext.library_dirs.append(path)
                else:
                    ext.extra_compile_args = copt[compiler]
                    ext.extra_link_args = lopt[compiler]
        build_ext.build_extensions(self)


# Cythonize only if c file is not present
use_cython = not os.path.isfile("src/itzi/flow.c")
file_ext = "pyx" if use_cython else "c"
extensions = [Extension("itzi.flow", sources=[f"src/itzi/flow.{file_ext}"])]
if use_cython:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)


setup(ext_modules=extensions, cmdclass={"build_ext": build_ext_compiler_check})
