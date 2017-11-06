#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
The heavy-lifting is here in cython.

Draws from Allen Riddell's LDA library https://github.com/ariddell/lda
"""

from datetime import (datetime, timedelta)
import numpy as np
from libc.math cimport fabs
from cython.operator cimport (preincrement, predecrement)
from cython_gsl cimport (gsl_sf_lngamma as lngamma, gsl_sf_exp as exp,
                         gsl_sf_log as ln, gsl_rng, gsl_rng_mt19937,
                         gsl_rng_alloc, gsl_rng_set,
                         gsl_rng_uniform, gsl_rng_uniform_int,
                         gsl_ran_gaussian as gaussian)
from pypolyagamma import PyPolyaGamma
import scipy as sp


cdef double[:, ::1] Psi_try = np.identity(3)
print("NumPy identity: %s" % Psi_try.shape)
