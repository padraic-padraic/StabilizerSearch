#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

import array
from cpython cimport array

from ..mat import X, Y, Z

DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t
cdef np.ndarray([DTYPE_t, ndim=2]) I = np.identity(2, dtype=np.complex128)

cdef test_real(short[:] bits, int n_qubits):
    cdef short[:] x = bits[:n_qubits]
    cdef short[:] z = bits[n_qubits:]
    cdef int i, sum=0
    for i in range(n_qubits):
        if x[i]+z[1]:
            _sum +=1
    return _sum%2==0

cdef bits_to_pauli(short[:] bits, int n_qubits):
    cdef short[:] x = bits[:n_qubits]
    cdef short[:] z = bits[n_qubits:]
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] pauli, out
    if x[0]==0 and z[0]==0:
        out = I
    elif x[0]==0 and z[0]==1:
        out = Z
    elif x[0]==1 and z[0]==0:
        out = X
    else:
        out = Y
    for i in range(1, n_qubits):
        if x[0]==0 and z[0]==0:
            pauli = I
        elif x[0]==0 and z[0]==1:
            pauli = Z
        elif x[0]==1 and z[0]==0:
            pauli = X
        else:
            pauli = Y
        out = np.kron(out, pauli)
    return out

cdef random_pauli(int n_qubits, int real_only):
    cdef int n, max_n = pow(2, 2*n_qubits)
    while True:
        n = numpy.random.randint(max_n)
        b = bin(n)[2:]
        b = 0*(2(n_qubits-len(b))) + b
        cdef array.array bits = array.array('h', [int(i) for i in b])
        cdef short[:] bits_view = bits
        if real_only == 1:
            if test_real(bits_view, n_qubits) == 1:
                break
        else:
            break
    cdef np.ndarray[DTYPE_t, ndim=2] pauli = bits_to_pauli(bits_view, n_qubits)
    cdef double r = np.random.random()
    if r > 0.5:
        return -1*pauli
    return pauli

cdef random_walk(int n_qubits, np.ndarray[DTYPE_t, ndim=2] target_state,
                 int chi, int beta, int beta_max, int beta_diff,
                 int walk_steps):
    pass