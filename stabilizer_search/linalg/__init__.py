# from ..clib.c_stabilizers import c_get_projector
from .gram_schmidt import ortho_projector as gs_projector
from .symmetric_subspace import symmetric_projector

import numpy as np

def calculate_overlap(states, target):
    tot = 0
    for state in states:
        olp = np.sum(state.H*target)
        tot += np.abs(olp.conj()*olp)
    return tot

def get_projector(vectors):
    dim = vectors[0].size
    for i in range(len(vectors)):
        vectors[i] = vectors[i].reshape(dim,1)
    vec_mat = np.matrix(np.zeros((dim, len(vectors)), dtype=np.complex_))
    for i in range(len(vectors)):
        vec_mat[:,i] = vectors[i]
    q, r = np.linalg.qr(vec_mat)
    projector = q*q.H
    return projector

def subspace_distance(a, b):
    norm = np.linalg.norm(np.transpose(b)-np.matmul(
                          np.matmul(np.transpose(b), a),
                          np.transpose(a)), 2)
    return asin(norm)
