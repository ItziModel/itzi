from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np

ext_modules = [Extension("flow", ["flow.pyx"])]

setup(
  name = 't.sim.flood',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],         # <---- New line
  ext_modules = ext_modules
)
