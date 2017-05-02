#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

cimport cy_generators

import numpy as np
cimport numpy as np

complex = np.complex128
ctypedef np.complex128_t complex_t
boolish = np.int8
ctypedef np.int8_t bool_t

import itertools

cdef np.ndarray[bool_t, ndim=1] num_to_pauli(unsigned int nqubits, unsigned int num)
    bitstring = bin(0)[2:].zfill(2*nQubits)
    cdef bits = np.ndarray(bool_t, ndim=1)
    bits = np.zeros(2*nQubits)
    unsigned int i
    for i in range(2*nQubits):
        bits[i] = bitstring[i]=='1'
    return bits

cdef class StabilizerMatrix:
    cdef public unsigned int nQubits
    cdef public np.ndarray[complex_t, ndim=2] sMatrix
    
    def __init__(self, list generators, nQubits=None):
        cdef unsigned int i
        if nQubits is None:
            nQubits = len(generators)
        sMatrix = np.zeros(nQubits, 2*nQubits)
        for i in range(nQubits):
            self.sMatrix[i] = generators[i]
        self.__to_canonical_form()
    
    cdef isXY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit]==1):
            return True
        return False

    cdef isZY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit+self.nQubits]==1):
            return True
        return False

    cdef bint linearly_independent(self):
        cdef unsigned int i
        for i in range(self.nQubits):
            if np.all(self.sMatrix[i]==0):
                return False
        return True

    cdef void __to_canonical_form(self):
        cdef unsigned int i = 0, j, k, m
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isXY(k,j):
                    self.sMatrix[[i,k]] = self.sMatrix[[k,i]]
                    for m in range(self.nQubits):
                        if m != i and self.isZY(m, j):
                            self.sMatrix[m] += self.sMatrix[i]
                            self.sMatrix[m] = np.mod(self.sMatrix[m], 2)
                    i+=1
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isZY(k, j):
                    self.sMatrix[[i,k]] = self.sMatrix[[k,i]]
                    for m in range(self.nQubits):
                        if m!= i and self.isZY(m, j):
                            self.sMatrix[m]+= self.sMatrix[i]
                            self.sMatrix[m] = np.mod(self.sMatrix[m], 2)
                    i++

cpdef get_positive_groups(unsigned int nQubits):
    cdef list paulis []
    cdef list groups
    cdef unsigned int i
    for i in range(pow(2,nQubits)):
        paulis.append(SymplecticPauli(i))
    for combination in itertools.combinations(paulis, 2):
        if paulis_commute(combination):
            candidate = StabilizerMatrix(combination)
            if candidate.linearly_independent():
                if not candidate in groups:
                    groups.append(candidate.sMatrix.view(dtype=np.bool))
    return groups