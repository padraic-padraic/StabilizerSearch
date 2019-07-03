from functools import reduce
from math import asin, sqrt
import operator as op

from numba import jit, njit

import numpy as np

__all__ = ['get_projector', 'subspace_distance', 'symmetric_projector',
           'tensor']


@njit
def tensor(*args):
    matrices = list(args)
    out = matrices.pop(0)
    while matrices:
        out = np.kron(out, matrices.pop(0))
    return out


@njit(parallel=True)
def get_projector(vectors):
    dim = vectors[0].size
    for i in range(len(vectors)):
        vectors[i] = vectors[i].reshape(dim, 1)
    vec_mat = np.matrix(np.zeros((dim, len(vectors)), dtype=np.complex_))
    for i in range(len(vectors)):
        vec_mat[:, i] = vectors[i]
    q, r = np.linalg.qr(vec_mat)
    projector = q*q.H
    return projector


@njit
def subspace_distance(a, b):
    norm = np.linalg.norm(np.transpose(b) - (
        (np.transpose(b) @ a) @ np.transpose(a)
    ), 2)
    return asin(norm)


@njit
def projection_distance(state, projector):
    distance = 1. - np.linalg.norm(projector*state, 2)
    return distance


ZERO = np.matrix([[1], [0]], dtype=np.complex_)
ONE = np.matrix([[0], [1]], dtype=np.complex_)


@njit
def ncr(n, r):
    r = min(r, n-r)
    if r == 0:
        return 1
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom

@njit
def int_to_bits(_int, n_qubits):
    out = np.zeros((n_qubits, 1))
    for i in range(n_qubits):
        out[n_qubits - (i+1)] = (_int >> i) & 1
    return out

@njit
def symmetric_projector(n_qubits):
    dim = pow(2, n_qubits)
    ints = [int_to_bits(i, n_qubits) for i in range(dim)]
    vectors = np.matrix(np.zeros((dim, n_qubits+1), dtype=np.complex_))
    for i in range(n_qubits+1):
        vec_placeholder = np.zeros((dim, 1), dtype=np.complex_)
        for element in ints:
            if np.sum(element) == i:
                el = tensor(*[ZERO if bit == 0 else ONE
                              for bit in element])
                vec_placeholder += el
        vec_placeholder /= np.linalg.norm(vec_placeholder, 2)
        vec_placeholder *= sqrt(ncr(n_qubits, i))
        vectors[:, i] = vec_placeholder
    q, r = np.linalg.qr(vectors)
    projector = q*q.H
    return projector

@njit
def np_inc_in_list(candidate, existing):
    for arrray in existing:
        if np.all(np.equal(candidate, existing)):
            return True
    return False
