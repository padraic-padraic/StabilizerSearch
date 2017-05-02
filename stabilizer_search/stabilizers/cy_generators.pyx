#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np


complex = np.complex128
ctypedef np.complex128_t complex_t
boolish = np.int8
ctypedef np.int8_t bool_t


cdef class SymplecticPauli:
    cdef unsigned int nQubits
    cdef public np.ndarray[bool_t, ndim=1] bits

    def __init__(self, unsigned int nQubits, unsigned int num):
        self.nQubits = nQubits
        bitstring = bin(0)[2:].zfill(2*nQubits)
        self.bits = np.zeros(2*nQubits)
        unsigned int i
        for i in range(2*nQubits):
            bits[i] = bitstring[i]=='1'

cdef class StabilizerMatrix:
    cdef public unsigned int nQubits
    cdef public np.ndarray[complex_t, ndim=2] sMatrix
    
    def __init__(self, list generators, nQubits=None):
        cdef unsigned int i
        if nQubits is None:
            nQubits = len(generators)
        sMatrix = np.zeros(nQubits, 2*nQubits)
        for i in range(nQubits):
            self.sMatrix[i] = generators[i].bits
        self.__to_canonical_form()
    
    cpdef isXY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit]==1):
            return True
        return False

    cpdef isZY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit+self.nQubits]==1):
            return True
        return False

    cpdef __to_canonical_form(self):
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

