#!python
#cython: boundscheck=False, nonecheck=False, wraparound=False

# cimport cy_generators

import numpy as np
cimport numpy as np

import itertools

from .utils import n_stabilizer_states, array_to_string
from .py_generators import test_commutivity

DTYPE = np.int8
ctypedef np.int8_t DTYPE_t

cdef char falsey = 0
cdef char truey = 1
cdef char base = 2

cdef np.ndarray[DTYPE_t, ndim=1] num_to_pauli(unsigned int nqubits, 
                                             unsigned int num):
    bitstring = bin(num)[2:].zfill(2*nqubits)
    cdef np.ndarray[DTYPE_t, ndim=1] bits
    bits = np.zeros(2*nqubits, dtype=np.int8)
    cdef unsigned int i
    for i in range(2*nqubits):
        bits[i] = (bitstring[i]=='1')
    return bits

cdef bint commutes(p1, p2):
    cdef np.ndarray[DTYPE_t, ndim=1] bit_array
    cdef unsigned int i, tot=0, nQubits = len(p1)//2
    bit_array = np.asarray(p1)&np.roll(np.asarray(p2), nQubits)
    tot = np.sum(bit_array)
    return (tot==falsey)


cpdef paulis_commute(tuple paulis):
    cdef bint res
    for p1, p2 in itertools.combinations(paulis, 2):
        res = commutes(p1, p2)
        if not res:
            return False
    return True


cdef class StabilizerMatrix:

    cdef public unsigned int nQubits
    cdef public char [:,:] sMatrix


    def __init__(self, tuple generators, nQubits=None):
        cdef unsigned int i, j
        if nQubits is None:
            nQubits = len(generators)
        self.nQubits = nQubits
        self.sMatrix = np.zeros((nQubits, 2*nQubits), dtype=np.int8)
        for i in range(nQubits):
            for j in range(2*nQubits):
                self.sMatrix[i, j] = generators[i][j]
        self.__to_canonical_form()
    

    def __richcmp__(StabilizerMatrix x, StabilizerMatrix y, int op):
        if op != 2 and op != 3:
            raise NotImplementedError('Order comparisons are meaningless for StabilizerMatrices')
        if x.nQubits != y.nQubits:
            return False
        cdef unsigned int i, j
        for i in range(x.nQubits):
            for j in range(2*x.nQubits):
                if x.sMatrix[i,j] != y.sMatrix[i,j]:
                    if (op==3):
                        return True
                    return False
        if (op==3):
            return False
        return True


    cpdef bint isIdentity(self, unsigned int row):
        cdef unsigned int i
        for i in range(2*self.nQubits):
            if self.sMatrix[row, i] != falsey:
                return False
        return True


    cpdef bint isXY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit]==truey):
            return True
        return False


    cpdef bint isZ(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit]==falsey and 
            self.sMatrix[row, qubit+self.nQubits]==truey):
            return True
        return False


    cpdef bint isZY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit+self.nQubits]==truey):
            return True
        return False


    cpdef bint linearly_independent(self):
        cdef unsigned int i
        i = np.linalg.matrix_rank(self.sMatrix)
        if i < self.nQubits:
            return False
        return True


    cpdef void rowMult(self, unsigned int i, unsigned int k):
        cdef unsigned int index
        cdef char scratch
        for index in range(self.nQubits):
            scratch = self.sMatrix[i, index]^self.sMatrix[k, index]
            self.sMatrix[i,index] = scratch


    cpdef void rowSwap(self, unsigned int i, unsigned int k):
        cdef unsigned int transpose_scratch
        cdef unsigned int index
        cdef char scratch
        for index in range(2*self.nQubits):
            transpose_scratch = self.sMatrix[k, index]
            self.sMatrix[k, index] = self.sMatrix[i, index]
            self.sMatrix[i, index] = transpose_scratch


    cpdef void __to_canonical_form(self):
        cdef unsigned int i = 0, j, k, m, index
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isXY(k,j):
                    self.rowSwap(i,k)
                    for m in range(self.nQubits):
                        if m != i and self.isXY(m, j):
                            self.rowMult(m,i)
                            # for index in range(2*self.nQubits):
                            #     scratch = self.sMatrix[i,index]^self.sMatrix[m, index]
                            #     self.sMatrix[m, index] = scratch
                    i+=1
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isZ(k, j):
                    self.rowSwap(i, k)
                    for m in range(self.nQubits):
                        if m!= i and self.isZY(m, j):
                            # for index in range(2*self.nQubits):
                            #     scratch = self.sMatrix[i,index]^self.sMatrix[m, index]
                            #     self.sMatrix[m, index] = scratch
                            self.rowMult(m,i)
                    i+=1
        return


cpdef get_positive_groups(unsigned int nQubits, unsigned int nStates):
    cdef unsigned int target
    if nStates == n_stabilizer_states(nQubits):
        target = nStates / pow(2, nQubits)
    else: #TODO: Handle the case where nstates > n_positive but < all
        target = nStates
    cdef list paulis
    cdef list groups
    paulis = []
    groups = []
    cdef unsigned int i
    cdef bint ctest
    for i in range(1, pow(2, 2*nQubits)):
        paulis.append(num_to_pauli(nQubits, i))
    for combination in itertools.combinations(paulis, nQubits):
        ctest =  paulis_commute(combination)
        if not ctest:
            continue
        candidate = StabilizerMatrix(combination, nQubits)
        if candidate.linearly_independent():
            if not candidate in groups:
                groups.append(candidate)
        # if len(groups)==target:
        #     break
    return [np.asarray(g.sMatrix) for g in groups]
