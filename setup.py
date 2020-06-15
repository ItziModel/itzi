#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import io
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.dist import Distribution
from setuptools.command.build_ext import build_ext
try:
    import numpy as np
except ImportError:
    sys.exit("Error: NumPy not found")


def get_version():
    """read version number from file"""
    ROOT = os.path.dirname(__file__)
    F_VERSION = os.path.join(ROOT, 'itzi', 'data', 'VERSION')
    with io.open(F_VERSION, 'r') as f:
        return f.readline().strip()


def get_long_description():
    with io.open('README.rst', 'r',  encoding='utf-8') as f:
        long_description = f.read()
    idx = max(0, long_description.find(u"Itzï is"))
    return long_description[idx:]


ENTRY_POINTS = {'console_scripts': ['itzi=itzi.itzi:main', ], }


CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
               "Operating System :: OS Independent",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Programming Language :: Python :: 3.7",
               "Programming Language :: Python :: 3.8",
               "Topic :: Scientific/Engineering"]


DESCR = "A 2D flood model using GRASS GIS as a back-end"


REQUIRES = ['pyinstrument', 'grass-session', 'pyswmm']


# Set arguments according to compiler
copt =  {'msvc': ['/openmp', '/Ox'],
         'mingw32' : ['-O3', '-w', '-fopenmp', '-lgomp', '-lpthread'],
         'unix' : ['-O3', '-w', '-fopenmp']
         }
lopt =  {'mingw32' : ['-lgomp', '-lpthread'],
         'unix' : ['-lgomp']
         }

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


ext_flow = Extension('itzi.flow', sources=['itzi/flow.c'],
                     include_dirs=[np.get_include()])

metadata = dict(name='itzi',
                version=get_version(),
                description=DESCR,
                long_description=get_long_description(),
                url='http://www.itzi.org',
                author='Laurent Courty',
                author_email='laurent@courty.me',
                license='GPLv2',
                classifiers=CLASSIFIERS,
                keywords='science engineering hydrology',
                packages=find_packages(),
                install_requires=REQUIRES,
                include_package_data=True,
                entry_points=ENTRY_POINTS,
                ext_modules=[ext_flow],
                cmdclass={'build_ext': build_ext_compiler_check},
                )


setup(**metadata)
