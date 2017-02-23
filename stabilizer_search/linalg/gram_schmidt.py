"""Submodule that provides useful linear algebra routines."""


from scipy.linalg import lu

import numpy as np


def check_lin_independence(vectors):
    """Gram-Schmidt only applies if vectors are linearly independent.
    We expect this to be tha case given Stab States for a mutually unbiased 
    basis but it's worth checking anyway."""
    if len(vectors) == 1:
        return True
    M = np.zeros([len(vectors), len(vectors[0])], dtype=np.complex_)
    for i in range(len(vectors)):
        M[i] = vectors[i].T
    pl, u = lu(M, permute_l=True)
    if any([np.count_nonzero(M[i]) == 0 for i in range(len(vectors))]):
        return False #M must be full rank for linear independence
    return True


def gs_prj(base, target):
    if np.allclose(np.sum((base.H*base)), 0):
        return 0*target
    return np.sum((base.H*target)) / np.sum((base.H*base)) * base


def gram_schmidt(vectors):
    dim = vectors[0].size
    n = len(vectors)
    # if not check_lin_independence(vectors):
    #     return None
    V = np.matrix(np.zeros([dim,n], dtype=np.complex_))
    U = np.matrix(np.zeros([dim,n], dtype=np.complex_))
    for i in range(len(vectors)):
        V[:,i] = vectors[i]
    for i in range(len(vectors)):
        U[:,i] = V[:,i]
        for j in range(i):
            U[:,i] -= gs_prj(U[:,j], V[:,i])
    for i in range(len(vectors)):
        norm = np.linalg.norm(np.matrix(U[:,i]), 2)
        if np.allclose(norm, 0):
            U[:,i] *= 0
        else:    
            U[:,i] /= norm
    return [np.matrix(U[:,i]) for i in range(len(vectors))]


def ortho_projector(vectors):
    dim  = len(vectors[0])
    ortho_vecs = gram_schmidt(vectors)
    A = np.matrix(np.zeros([dim, len(ortho_vecs)], dtype=np.complex_))
    for i in range(len(ortho_vecs)):
        A[:,i] = ortho_vecs[i]
    P = A*A.H
    return P
