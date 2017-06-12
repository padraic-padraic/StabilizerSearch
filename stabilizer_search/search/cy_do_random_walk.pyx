#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

from libc.math cimport exp
from copy import deepcopy

from ..mat import X, Y, Z
from ..stabilizers import get_stabilizer_states
from ..stabilizers.utils import array_to_pauli


DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t


cpdef get_projector(list states):
    cdef unsigned int hilbert_dim, n_vecs, index1, index2
    n_vecs = len(states)
    hilbert_dim = states[0].size
    cdef np.ndarray[DTYPE_t, ndim=2] vec_matrix, out_q, out_r
    vec_matrix = np.zeros((hilbert_dim, n_vecs), dtype=np.complex128)
    for index1 in range(n_vecs):
        for index2 in range(hilbert_dim):
            vec_matrix[index2,index1] = states[index1][index2]
    out_q, out_r = np.linalg.qr(vec_matrix, mode='complete')
    return out_q


cdef random_pauli(int n_qubits):
    cdef unsigned int index
    cdef np.ndarray[np.uint8_t, ndim=1] bits
    bits = np.zeros(2*n_qubits, dtype=np.uint8)
    bitstring = bin(np.random.randint(1, pow(2, 2*n_qubits)))[2:].zfill(2*n_qubits)
    for index in range(2*n_qubits):
        if bitstring[index]=='0':
            bits[index] = 0
        else:
            bits[index] = 1
    return array_to_pauli(bits.view(dtype=np.bool))


cdef trace_distance(np.ndarray[DTYPE_t, ndim=2] a, 
                           np.ndarray[DTYPE_t, ndim=2] b):
    cdef np.ndarray[np.float64_t, ndim=1] eigvals
    eigvals = np.absolute(np.linalg.eigvals(a-b))
    return 0.5*np.sum(eigvals)


cdef random_walk(int n_qubits, np.ndarray[DTYPE_t, ndim=2] target,
                 int chi, double beta, int beta_max, double beta_diff,
                 int walk_steps, bint is_state, bint real_only):
    cdef list stabilizers = get_stabilizer_states(n_qubits, chi, real_only=True)
    cdef np.ndarray[DTYPE_t, ndim=2] projector, new_projector, new_state
    cdef double distance, new_distance, new_norm, p_accept
    cdef int counter
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.identity(pow(2, n_qubits), dtype=np.complex128)
    projector = get_projector(stabilizers)
    if is_state:
        distance = 1 - np.linalg.norm(projector*target, 2)
    else:
        distance = trace_distance(projector, target)
    while beta <= beta_max:
        print("Anneal Progress : {}%".format((beta-1)/beta_diff))
        for counter in range(walk_steps):
            if np.allclose(distance, 0.):
                return True, chi, stabilizers
            while True:
                move = random_pauli(n_qubits)
                if np.any(np.imag(move)):
                    continue
                move_target = np.random.randint(chi)
                new_state = (I+move)*stabilizers[move_target]
                new_norm = np.linalg.norm(new_state, 2)
                new_state = new_state / new_norm
                if not np.allclose(new_norm, 0.):
                    break
            new_projector = get_projector([s if n != move_target else new_state 
                                             for n, s in enumerate(stabilizers)])
            if is_state:
                new_distance = 1-np.linalg.norm(new_projector*target, 2)
            else:
                new_distance = np.linalg.norm(new_projector-target)
            # print(distance, new_distance)
            if new_distance < distance:
                stabilizers[move_target] = deepcopy(new_state)
                distance = new_distance
            else:
                p_accept = exp(-1*beta*(new_distance-distance))
                if np.random.random() < p_accept:
                    stabilizers[move_target] = deepcopy(new_state)
                    distance = new_distance
        beta += beta_diff
    print('Final distance was {}'.format(distance))
    return False, chi, stabilizers


def cy_do_random_walk(n_qubits, target, chi, **kwargs):
    if target.shape[1]==1:
        is_state = True
    else:
        is_state = False
    beta = kwargs.pop('beta_init', 1)
    beta_max = kwargs.pop('beta_max', 4000)
    anneal_steps = kwargs.pop('steps', 100)
    beta_diff = (beta_max-beta)/anneal_steps
    walk_steps = kwargs.pop('M', 1000)
    real_only = kwargs.pop('real_only', False)
    res = random_walk(n_qubits, target, chi, beta, beta_max, beta_diff, 
                      walk_steps, is_state, real_only)
    return res
