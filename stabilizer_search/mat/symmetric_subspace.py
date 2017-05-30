from functools import reduce
from itertools import permutations
from math import sqrt

import numpy as np
import operator as op

from . import tensor

ZERO = np.matrix([[1],[0]], dtype=np.complex_)
ONE = np.matrix([[0],[1]], dtype=np.complex_)

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom

def symmetric_projector(n_qubits):
    dim = pow(2, n_qubits)
    ints = list(range(dim))
    vectors = np.matrix(np.zeros((dim,n_qubits+1), dtype=np.complex_))
    for i in range(n_qubits+1):
        vec_placeholder = np.zeros((dim,1), dtype=np.complex_)
        for element in filter(lambda x: bin(x).count("1")==i, ints):
            el = tensor(*[ZERO if l=='0' else ONE 
                          for l in bin(element)[2:].zfill(n_qubits)])
            vec_placeholder += el
        vec_placeholder /= np.linalg.norm(vec_placeholder, 2)
        vec_placeholder *= sqrt(ncr(n_qubits, i))
        vectors[:,i] = vec_placeholder
    projector, r = np.linalg.qr(vectors, mode='complete')
    return projector
