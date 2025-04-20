#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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


class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        print("compiler: {}".format(compiler))
        if compiler in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[compiler]
        if compiler in lopt:
            for e in self.extensions:
                e.extra_link_args = lopt[compiler]
        build_ext.build_extensions(self)


# Cythonize only if c file is not present
use_cython = not os.path.isfile("src/itzi/flow.c")
file_ext = "pyx" if use_cython else "c"
extensions = [Extension("itzi.flow", sources=[f"src/itzi/flow.{file_ext}"])]
if use_cython:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)


setup(ext_modules=extensions, cmdclass={"build_ext": build_ext_compiler_check})
