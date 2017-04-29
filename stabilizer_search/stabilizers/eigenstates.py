"""Module which defines methods for converting the QObj Pauli Operators that 
generate each stabilizer group, building a projector and finding the associated 
+1 eigenstate."""

from math import pow as fpow
from numpy import allclose, imag, isclose, real
from numpy.linalg import eig, norm

from ..mat import qeye

def find_projector(generating_set):
    n_qubits = len(generating_set)
    _id = qeye(pow(2, n_qubits))
    res = qeye(pow(2, n_qubits))
    for _g in generating_set:
        res = res * (_id+_g)
    return res/fpow(2, n_qubits)

def find_eigenstate(projector):
    eigs, vecs = eig(projector)
    for _n, _eig in enumerate(eigs):
        if allclose(_eig, complex(1)) or allclose(_eig, 1.):
            state = (vecs[:,_n])
            r = real(state)
            im = imag(state)
            r[isclose(r, 0.)] = 0
            im[isclose(im, 0.)] = 0
            state = r+1j*im
            state = state / norm(state, 2)
            return state
    return None

def py_find_eigenstates(generating_sets, real_only=False):
    """ """
    states = [find_eigenstate(x) for x in map(find_projector, generating_sets)]
    if real_only:
        return list(filter(lambda x: allclose(imag(x), 0.), states))
    return states
