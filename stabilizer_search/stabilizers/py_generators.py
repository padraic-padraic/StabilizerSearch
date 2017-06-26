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


def symplectic_inner_product(n, a, b):
    x_a, z_a = a[:n], a[n:]
    x_b, z_b = b[:n], b[n:]
    # count = (x_a&z_b).count() + (x_b&z_a).count()
    count = np.sum((x_a&z_b)) + np.sum((x_b&z_a))
    return count%2


class PauliArray(object):

    def __init__(self, n_qubits, number=0):
        self.n_qubits = n_qubits
        bin_string = bin(number)[2:] #strip the 0b from the string
        bin_string = '0'*(2*n_qubits - len(bin_string)) + bin_string
        self.bits = np.array([b == '1' for b in bin_string])

    @classmethod
    def from_arr(cls, array):
        n_qubits = array.size //2
        out = cls(n_qubits)
        out.bits = array

    @classmethod
    def from_string(cls, _str):
        pauli = cls(n_qubits)
        pauli.bits = array_from_string(_str)
        return pauli
        
    def __mul__(self, other):
        if self.n_qubits != other.n_qubits:
            raise ValueError("Paulis act on different numbers of qubits.")
        out = PauliArray(self.n_qubits)
        out.bits == xor(self.bits, other.bits)

    def __imul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if self.n_qubits != other.n_qubits:
            raise ValueError("Paulis act on different numbers of qubits.")
        return np.array_equal(self.bits, out.bits)

    def __repr__(self):
        return '<Pauli Array on {} qubits: {}'.format(self.n_qubits, 
                                                      array_to_string(self.bits))

    def __str__(self):
        return array_to_string('self.bits')

    def commute(self, other):
        if self.n_qubits != other.n_qubits:
            raise ValueError("Paulis act on different numbers of qubits.")
        return symplectic_inner_product(self.n_qubits, self.bits, other.bits)==0

    def is_real(self):
        return np.sum(self.bits[:self.n_qubits] & 
                      self.bits[self.n_qubits:])%2 ==0

class BinarySubspace(object):
    """Set-like class for PauliArray objects to generate a closed subspace."""
    def __init__(self, *data):
        self.order = 0
        self._items = []
        self.generators = []
        for val in data:
            if not isinstance(val, PauliArray):
                raise ValueError('This class works for PauliArrays only!')
            self.add(val)
    
    def __contains__(self, it):
        for _el in self._items:
            if _el == it:
            # if np.array_equal(_el, it):
                return True
        return False

    def __iter__(self):
        for item in self._items:
            yield item

    def _generate(self, obj):
        for item in self._items:
            new = item*obj
            # new = xor(item, obj)
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
        if not p1.commute(p2):
        # if not test_commutivity(n_qubits, p1, p2):
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
    # bitstrings = gen_bitstrings(n_qubits)
    paulis = [PauliArray(n_qubits, i) for i in range(1, pow(2,2*n))]
    # shuffle(bitstrings)
    shuffle(paulis)
    subspaces = []
    generators = []
    for group in combinations(bitstrings, n_qubits):
        if not paulis_commute(n_qubits, group):
            continue
        if np.linalg.matrix_rank(np.matrix([g.bits for g in group])) < n_qubits:
            continue
        if real_only:
            if not is_real(group):  
                continue
        candidate = BinarySubspace(*group)
        if len(candidate.generators) < n_qubits:
            continue
        if len(candidate._items) < pow(2,n_qubits):
            continue
        res = tuple(i for i in sorted(candidate._items, key=bool_to_int))
        if np.any([np.all([_el1==_el24 for _el1, _el2 in zip(res, space)]) 
                   for space in subspaces]):
            continue
        subspaces.append(res)
        generators.append(tuple(candidate.generators))
        # if not candidate in generators:
        #     generators.append(candidate)
        if len(generators) == target:
            break
    return generators
