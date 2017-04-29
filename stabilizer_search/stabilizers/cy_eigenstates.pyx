#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

from libc.math cimport pow as fpow

DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t


cpdef find_projector(list generating_set):
    cdef int n_qubits = len(generating_set)
    cdef np.ndarray[DTYPE_t, ndim=2] _id = np.identity(pow(2, n_qubits), dtype=np.complex128)
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.identity(pow(2, n_qubits), dtype=np.complex128)
    for _g in generating_set:
        res = res * (_id+_g)
    cdef double norm = fpow(2, n_qubits)
    return res/norm


cpdef find_eigenstate(np.ndarray[DTYPE_t, ndim=2] projector):
    cdef np.ndarray[DTYPE_t, ndim=1] eigs
    cdef double complex _eig
    cdef np.ndarray[DTYPE_t, ndim=2] vecs
    eigs, vecs = np.linalg.eig(projector)
    cdef unsigned int _n
    cdef np.ndarray[DTYPE_t, ndim=2] state
    cdef np.ndarray[np.float64_t, ndim=2] r, im
    cdef double real=1
    cdef DTYPE_t comp=1.
    for _n, _eig in enumerate(eigs):
        if np.allclose(_eig, comp) or np.allclose(_eig, real):
            state = (vecs[:,_n])
            r = np.real(state)
            im = np.imag(state)
            r[np.isclose(r, 0.)] = 0
            im[np.isclose(im, 0.)] = 0
            state = r+1j*im
            state = state / np.linalg.norm(state, 2)
            return state
    return None


def cy_find_eigenstates(list generating_sets, real_only=False):
    cdef np.ndarray[DTYPE_t, ndim=2] x
    states = [find_eigenstate(x) for x in map(find_projector, generating_sets)]
    if real_only:
        return list(filter(lambda x: np.allclose(np.imag(x), 0.), states))
    return states