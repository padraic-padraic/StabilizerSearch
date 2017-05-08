#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

# cimport cy_generators

import numpy as np
cimport numpy as np

import itertools


DTYPE = np.int8
ctypedef np.int8_t DTYPE_t

cdef int falsey = 0
cdef int truey = 1
cdef int base = 2

cdef np.ndarray[DTYPE_t, ndim=1] num_to_pauli(unsigned int nqubits, 
                                             unsigned int num):
    bitstring = bin(0)[2:].zfill(2*nqubits)
    cdef np.ndarray[DTYPE_t, ndim=1] bits
    bits = np.zeros(2*nqubits, dtype=np.int8)
    cdef unsigned int i
    for i in range(2*nqubits):
        bits[i] = bitstring[i]=='1'
    return bits

cdef bint commutes(p1, p2):
    cdef unsigned int i, tot=0
    eqs = (p1==np.roll(p2,p1.nQubits))
    for i in range(2*p1.nQubits):
        if eqs[i]:
            tot += p1[i]
    return (tot % 2 == 0)


cpdef paulis_commute(list paulis):
    for p1, p2 in itertools.combinations(paulis):
        if not commutes(p1, p2):
            return False
    return True


cdef class StabilizerMatrix:
    cdef public unsigned int nQubits
    cdef public int [:,:] sMatrix
    def __init__(self, list generators, nQubits=None):
        cdef unsigned int i
        if nQubits is None:
            nQubits = len(generators)
        self.nQubits = nQubits
        self.sMatrix = np.zeros((nQubits, 2*nQubits), dtype=np.int8)
        for i in range(nQubits):
            self.sMatrix[i] = generators[i]
        self.__to_canonical_form()
    
    def __richcmp__(StabilizerMatrix x, StabilizerMatrix y, int op):
        if op != 2 or op != 3:
            raise NotImplementedError('Order comparisons are meaningless for StabilizerMatrices')
        if x.nQubits != y.nQubits:
            return False
        cdef unsigned int i, j
        for i in range(x.nQubits):
            for j in range(2*x.nQubits):
                if x.StabilizerMatrix[i,j] != y.sMatrix[i,j]:
                    return False
        return True


    cdef isXY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit]==truey):
            return True
        return False


    cdef isZ(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit]==falsey and 
            self.sMatrix[row, qubit+self.nQubits]==truey):
            return True
        return False

    cdef isZY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit+self.nQubits]==truey):
            return True
        return False

    cdef bint linearly_independent(self):
        cdef unsigned int i, j, tot
        for i in range(self.nQubits):
            tot = 0
            for j in range(2*self.nQubits):
                tot += self.sMatrix[i,j]
            if tot == 0:
                return False
        return True

    cdef void rowMult(self, unsigned int i, unsigned int k):
        cdef unsigned int index
        for index in range(self.nQubits):
            self.sMatrix[i,index] = (self.sMatrix[i, index] + 
                                    self.sMatrix[k, index])

    cdef void rowSwap(self, unsigned int i, unsigned int k):
        cdef unsigned int transpose_scratch
        transpose_scratch = np.zeros(2*self.nQubits, dtype=np.int8)
        cdef unsigned int index
        for index in range(2*self.nQubits):
            transpose_scratch = self.sMatrix[k, index]
            self.sMatrix[k, index] = self.sMatrix[i, index]
            self.sMatrix[i, index] = transpose_scratch

    cdef void rowMod(self, unsigned int i):
        cdef unsigned int index
        for index in range(2*self.nQubits):
            self.sMatrix[i, index] = (self.sMatrix[i, index] % base)

    cdef void __to_canonical_form(self):
        transpose_scratch = np.zeros(self.nQubits, dtype=np.int8)
        cdef unsigned int i = 0, j, k, m
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isXY(k,j):
                    self.rowSwap(i,k)
                    for m in range(self.nQubits):
                        if m != i and self.isZY(m, j):
                            self.rowMult(m, i)
                            self.rowMod(m)
                    i+=1
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isZ(k, j):
                    transpose_scratch = self.sMatrix[i,k]
                    self.sMatrix[i,k] = self.sMatrix[k,i]
                    self.sMatrix[k,i] = transpose_scratch
                    for m in range(self.nQubits):
                        if m!= i and self.isZY(m, j):
                            self.rowMult(m, i)
                            self.rowMod(m)
                    i+=1


cpdef get_positive_groups(unsigned int nQubits):
    cdef list paulis
    cdef list groups
    paulis = []
    groups = []
    cdef unsigned int i
    for i in range(pow(2,nQubits)):
        paulis.append(num_to_pauli(nQubits, i))
    for combination in itertools.combinations(paulis, 2):
        if paulis_commute(combination):
            candidate = StabilizerMatrix(combination)
            if candidate.linearly_independent():
                if not candidate in groups:
                    groups.append(candidate.sMatrix.view(dtype=np.bool))
    return groups