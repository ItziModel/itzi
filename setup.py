#!/usr/bin/env python
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
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 2.7",
               "Topic :: Scientific/Engineering"]


DESCR = "A 2D superficial flow simulation model using GRASS GIS as a back-end"


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
        if copt.has_key(compiler):
           for e in self.extensions:
               e.extra_compile_args = copt[compiler]
        if lopt.has_key(compiler):
            for e in self.extensions:
                e.extra_link_args = lopt[compiler]
        build_ext.build_extensions(self)


FLOW = Extension('flow', sources=['itzi/flow.c'],
                 include_dirs=[np.get_include()])


metadata = dict(name='itzi',
                version=get_version(),
                description=DESCR,
                long_description=get_long_description(),
                url='http://itzi.org',
                author='Laurent Courty',
                author_email='lrntct@gmail.com',
                license='GPLv2',
                classifiers=CLASSIFIERS,
                keywords='science engineering hydrology',
                packages=find_packages(),
                requires=['numpy', 'pyinstrument'],
                install_requires=['pyinstrument'],
                include_package_data=True,
                entry_points=ENTRY_POINTS,
                ext_modules=[FLOW,],
                cmdclass={'build_ext': build_ext_compiler_check},
                )


setup(**metadata)
