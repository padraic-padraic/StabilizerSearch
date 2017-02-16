cimport chaar_random
import numpy as np

def get_so2():
    cdef long double **mat = chaar_random.random_so2()
    return np.matrix([[mat[0][0], mat[0][1]],
                      [mat[1][0], mat[1][1]]], dtype=np.complex_)

def get_su2():
    cdef long double complex **mat = chaar_random.random_su2();
    return np.matrix([[mat[0][0], mat[0][1]],
                      [mat[1][0], mat[1][1]]], dtype=np.complex_)
