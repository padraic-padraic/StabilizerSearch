#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

import array
from cpython cimport array

from libc.math cimport exp
from copy import deepcopy

from ..stabilizers import get_stabilizer_states

from ..mat import X, Y, Z

DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t

cpdef get_projector(list states):
    cdef unsigned int hilbert_dim, n_vecs, index
    n_vecs = len(states)
    hilbert_dim = states[0].size()
    cdef np.ndarray[DTYPE_t, ndim=2] vec_matrix, out_q, out_r
    vec_matrix = np.zeros((hilbert_dim, n_vecs), dtype=np.complex128)
    for index in range(n_vecs):
        vec_matrix[:,index] = states[index]
    out_q, out_r = np.linalg.qr(vec_matrix, mode='complete')
    return out_q

cdef test_real(array.array bits, int n_qubits):
    cdef short[:] bits_view = bits
    cdef short[:] x = bits_view[:n_qubits]
    cdef short[:] z = bits_view[n_qubits:]
    cdef int i, sum=0
    for i in range(n_qubits):
        if (x[i]+z[i]==2):
            _sum +=1
    return _sum%2==0

cdef bits_to_pauli(array.array bits, int n_qubits):
    cdef short[:] bits_view = bits
    cdef short[:] x = bits_view[:n_qubits]
    cdef short[:] z = bits_view[n_qubits:]
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] pauli, out
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.identity(2, dtype=np.complex128)
    if x[0]==0 and z[0]==0:
        out = I
    elif x[0]==0 and z[0]==1:
        out = Z
    elif x[0]==1 and z[0]==0:
        out = X
    else:
        out = Y
    for i in range(1, n_qubits):
        if x[i]==0 and z[i]==0:
            pauli = I
        elif x[i]==0 and z[i]==1:
            pauli = Z
        elif x[i]==1 and z[i]==0:
            pauli = X
        else:
            pauli = Y
        out = np.kron(out, pauli)
    return out

cdef random_pauli(int n_qubits):
    cdef int n, max_n = pow(2, 2*n_qubits)
    cdef array.array bits
    while True:
        n = np.random.randint(1, max_n)
        b = bin(n)[2:]
        b = '0'*(2*n_qubits-len(b)) + b
        bits = array.array('h', [int(i) for i in b])
    cdef np.ndarray[DTYPE_t, ndim=2] pauli = bits_to_pauli(bits, n_qubits)
    cdef double r = np.random.random()
    if r > 0.5:
        return -1*pauli
    return pauli

cdef random_walk(int n_qubits, np.ndarray[DTYPE_t, ndim=2] target_state,
                 int chi, double beta, int beta_max, double beta_diff,
                 int walk_steps, is_state):
    cdef list stabilizers = get_stabilizer_states(n_qubits, chi)
    cdef np.ndarray[DTYPE_t, ndim=2] projector, new_projector, new_state
    cdef double distance, new_distance, new_norm, p_accept
    cdef int counter
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.identity(pow(2, n_qubits), dtype=np.complex128)
    projector = get_projector(stabilizers)
    if is_state:
        distance = 1- np.linalg.norm(projector*target_state, 2)
    else:
        distance = np.linalg.norm(projector-target_state)
    while beta <= beta_max:
        for counter in range(walk_steps):
            if np.allclose(distance, 0.):
                return True, chi, stabilizers
            while True:
                move = random_pauli(n_qubits)
                move_target = np.random.randint(chi)
                new_state = (I+move)*stabilizers[move_target]
                new_norm = np.linalg.norm(new_state, 2)
                new_state = new_state / new_norm
                if not np.allclose(new_norm, 0.):
                    break
            new_projector = get_projector([s if n != move_target else new_state 
                                             for n, s in enumerate(stabilizers)])
            if is_state:
                new_distance = 1-np.linalg.norm(new_projector*target_state, 2)
            else:
                new_distance = np.linalg.norm(new_projector-target_state)
            if new_distance < distance:
                stabilizers[move_target] = deepcopy(new_state)
                distance = new_distance
            else:
                p_accept = exp(-1*beta*(new_distance-distance))
                if np.random.random() < p_accept:
                    stabilizers[move_target] = deepcopy(new_state)
                    distance = new_distance
        beta += beta_diff
    return False, chi, stabilizers


def cy_do_random_walk(n_qubits, target_state, chi, **kwargs):
    if target_state.shape[1]==1:
        is_state = True
    else:
        is_state = False
    beta = kwargs.pop('beta_init', 1)
    beta_max = kwargs.pop('beta_max', 4000)
    anneal_steps = kwargs.pop('steps', 100)
    beta_diff = (beta_max-beta)/anneal_steps
    walk_steps = kwargs.pop('M', 1000)
    res = random_walk(n_qubits, target_state, chi, beta, beta_max, beta_diff, 
                      walk_steps, is_state)
    return res
