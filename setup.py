import os

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
try:
    import numpy as np
    import cython_gsl
except ImportError:
    print("Please install numpy and cythongsl.")

# Dealing with Cython
# Dealing with Cython
USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension('STMPG_cscript', ['STMPG_cscript.pyx'],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=[np.get_include(), cython_gsl.get_include()],),
]


setup(
	name = "stmpg",
    ext_modules=cythonize(extensions)
    
)
