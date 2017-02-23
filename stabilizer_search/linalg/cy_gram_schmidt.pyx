#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t

cdef gs_prj(np.ndarray[DTYPE_t, ndim=2] base, 
            np.ndarray[DTYPE_t, ndim=2] target):
    if np.allclose(base.H*base, 0):
        return 0*target
    cdef DTYPE_t prefactor = np.asscalar(base.H*target)
    prefactor /= np.asscalar(base.H*base)
    return prefactor*base

cdef gram_schmidt(list vectors, int dimension, int n):
    cdef np.ndarray[DTYPE_t, ndim=2] U
    U = np.zeros([dimension, n], dtype=np.complex128)
    U = np.matrix(U) #Still works because of that sweet sweet polymorphism
    cdef int i, j
    for i in range(n):
        U[:,i] = vectors[i]
        for j in range(i):
            U[:,i] -= gs_prj(U[:,j], vectors[i])
    return [U[:,i] for i in range(n)]

cpdef ortho_projector(list vectors):
    cdef int dimension = vectors[0].size
    cdef int n = len(vectors)
    cdef list ortho_vectors = gram_schmidt(vectors, dimension, n)
    cdef np.ndarray[DTYPE_t, ndim=2] A = np.matrix(np.zeros([dimension, n],
                                                            dtype=np.complex128))
    cdef int i
    for i in range(n):
        A[:,i] = ortho_vectors[i]
    return A*A.H