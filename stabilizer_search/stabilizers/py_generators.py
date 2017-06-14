"""Module which finds unique generating sets for pauli Stabilier groups using
python code and the bitarray module."""


from itertools import combinations
from random import shuffle
from .utils import *

import numpy as np


__all__ = ['get_positive_stabilizer_groups']


def xnor(a,b):
    return (a&b)^(~a&~b)


def xor(a,b):
      return (a|b)&~(a&b)


class BinarySubspace(object):
    """Set-like class for bitarray objects to generate a closed subspace."""
    def __init__(self, *data):
        self.order = 0
        self._items = []
        self.generators = []
        for val in data:
            # if not isinstance(val, bitarray):
            if not isinstance(val, np.ndarray):
                raise ValueError('This class works for numpy arrays only!')
                # raise ValueError('This class works for bitarrays only!')
            self.add(val)
    
    def __contains__(self, it):
        for _el in self._items:
            # if all(xnor(_el, it)):
            if np.array_equal(_el, it):
                return True
        return False

    def __iter__(self):
        for item in self._items:
            yield item

    def _generate(self, obj):
        for item in self._items:
            # new = item^obj
            new = xor(item, obj)
            if new in self:
                continue
            else:
                self.order +=1
                self._items.append(new)
                self._generate(new)
        return

    def __eq__(self, other):
        return all([_el in other for _el in self._items])

    def add(self, obj):
        for _el in self._items:
            if np.all(xnor(obj, _el)):
                return self
        self.order +=1
        self.generators.append(obj)
        self._items.append(obj)
        self._generate(obj)
        return self


class StabilizerMatrix(object):

    def __init__(self, generators):
        self.n_qubits = len(generators)
        self.sMatrix = np.zeros((self.n_qubits, 2*self.n_qubits), dtype=np.bool)
        for n, g in enumerate(generators):
            self.sMatrix[n] = g
        try:
            self.__to_canonical_form()
        except:
            for g in generators:
                print(array_to_string(g))

    def isZ(self, row, qubit):
        if self.sMatrix[row, qubit]:
            return False
        if self.sMatrix[row, qubit+self.n_qubits]:
            return True
        return False

    def isZY(self, row, qubit):
        if self.sMatrix[row, qubit+self.n_qubits]:
            return True
        return False

    def isXY(self, row, qubit):
        if self.sMatrix[row, qubit]:
            return True
        return False

    def rowMult(self, row1, row2):
        self.sMatrix[row1] = np.logical_xor(self.sMatrix[row1], 
                                            self.sMatrix[row2])

    @property
    def linearly_independent(self):
        return np.linalg.matrix_rank(self.sMatrix) == self.n_qubits

    def __to_canonical_form(self):
        i = 0
        for j in range(self.n_qubits):
            for k in range(self.n_qubits):
                if self.isXY(k, j):
                    self.sMatrix[[i,k]] = self.sMatrix[[k,i]]
                    for m in range(self.n_qubits):
                        if m!= i and self.isXY(m, j):
                            self.rowMult(m,i)
                    i+=1
                    break
        for j in range(self.n_qubits):
            for k in range(self.n_qubits):
                if self.isZ(k,j):
                    self.sMatrix[[i,k]] = self.sMatrix[[k,i]]
                    for m in range(self.n_qubits):
                        if m!= i and self.isZY(m, j):
                            self.rowMult(m, i)
                    i+=1
                    break

    def __eq__(self, other):
        return np.array_equal(self.sMatrix, other.sMatrix)

def symplectic_inner_product(n, a, b):
    x_a, z_a = a[:n], a[n:]
    x_b, z_b = b[:n], b[n:]
    # count = (x_a&z_b).count() + (x_b&z_a).count()
    count = np.sum((x_a&z_b)) + np.sum((x_b&z_a))
    return count%2


def test_commutivity(n, bits1, bits2):
    return symplectic_inner_product(n, bits1, bits2) == 0 #1 if they anticommute, 0 if they commute


def gen_bitstrings(n):
    bitstrings = []
    for i in range(1, pow(2,2*n)): #We ignore the all 0 string as it corresponds to I^{n}
        bin_string = bin(i)[2:] #strip the 0b from the string
        bin_string = '0'*(2*n - len(bin_string)) + bin_string
        a = np.array([b == '1' for b in bin_string])
        bitstrings.append(a)
    return bitstrings


def paulis_commute(n_qubits, paulis):
    for p1, p2 in combinations(paulis, 2):
        if not test_commutivity(n_qubits, p1, p2):
            return False
    return True

def get_positive_stabilizer_groups(n_qubits, n_states, real_only=False):
    if n_states > n_stabilizer_states(n_qubits)/pow(2, n_qubits): 
        # If generating all states, we want to focus on only the all
        # positive signed operators
        target = n_stabilizer_states(n_qubits)/pow(2, n_qubits)
    else:
        #If generating less than all, we'll add signs in randomly to compenstate
        target = n_states
    bitstrings = gen_bitstrings(n_qubits)
    shuffle(bitstrings)
    subspaces = []
    generators = []
    for group in combinations(bitstrings, n_qubits):
        if not paulis_commute(n_qubits, group):
            continue
        if np.linalg.matrix_rank(np.matrix(group)) < n_qubits:
            continue
        candidate = BinarySubspace(*group)
        if len(candidate.generators) < n_qubits:
            continue
        if len(candidate._items) < pow(2,n_qubits):
            continue
        # candidate = StabilizerMatrix(group)
        # if not candidate.linearly_independent:
        #     continue
        res = tuple(i for i in sorted(candidate._items, key=bool_to_int))
        if np.any([np.all([np.allclose(_el1, _el2) for _el1, _el2 in zip(res, space)]) 
                   for space in subspaces]):
            continue
        subspaces.append(res)
        generators.append(tuple(candidate.generators))
        # if not candidate in generators:
        #     generators.append(candidate)
        if len(generators) == target:
            break
    return generators
