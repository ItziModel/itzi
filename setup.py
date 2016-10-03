#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import io
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.dist import Distribution
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


FLOW = Extension('itzi/flow', sources=['itzi/flow.c'],
                 extra_compile_args=['-fopenmp', '-O3'],
                 extra_link_args=['-lgomp'],
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
                )


setup(**metadata)
