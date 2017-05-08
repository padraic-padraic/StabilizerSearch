#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

from .utils import array_to_pauli, get_sign_strings, n_stabilizer_states

from libc.math cimport pow as fpow

DTYPE=np.complex128
ctypedef np.complex128_t DTYPE_t
boolish = np.int8
ctypedef np.int8_t bool_t

cdef np.ndarray[DTYPE_t, ndim=1] get_eigenstate(list paulis):
    cdef unsigned int  dim = paulis[0].shape[0], i
    cdef np.ndarray[DTYPE_t, ndim=1] eigenvalues, state
    cdef np.ndarray[DTYPE_t, ndim=2] projector, identity, eigenvectors
    identity = np.identity(dim, dtype=np.complex128)
    projector = np.identity(dim, dtype=np.complex128)
    for p in paulis:
        projector *= (0.5*(identity+p))
    eigenvalues, eigenvectors = np.linalg.eig(projector)
    for i in range(len(eigenvalues)):
        if np.allclose(np.abs(eigenvalues[i]), 1.):
            state = eigenvectors[:,i]
            return state


cpdef cy_get_eigenstates(list positive_groups, unsigned int n_states):
    cdef unsigned int nqubits = len(positive_groups[0]), i, j
    cdef np.ndarray[bool_t, ndim=1, cast=True] p_string
    cdef list phase_strings, pauli_workspace, states, phase_workspace
    states = []
    pauli_workspace = [0]*nqubits
    phase_workspace = [0]*nqubits
    phase_strings = get_sign_strings(nqubits, n_states)
    if n_states == n_stabilizer_states(nqubits):
        #Extending the groups mode. 
        for i in range(len(positive_groups)):
            for j in range(nqubits):
                pauli_workspace[j] = array_to_pauli(positive_groups[i][j])
            states.append(get_eigenstate(pauli_workspace))
            for p_string in phase_strings:
                for j in range(nqubits):
                    if p_string[i]==0:
                        phase_workspace[j] = pauli_workspace[j]
                    else:
                        pauli_workspace[j] = -1*pauli_workspace[j]
                states.append(get_eigenstate(phase_workspace))
    else:
        #Replacing the groups mode
        for i in range(n_states):
            for j in range(nqubits):
                pauli_workspace[j] = array_to_pauli(positive_groups[i][j])
                if phase_strings[i][j] == 1:
                    pauli_workspace[j] *= -1
            states.append(get_eigenstate(pauli_workspace))
    return states
