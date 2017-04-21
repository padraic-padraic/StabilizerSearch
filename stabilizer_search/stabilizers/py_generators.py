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
            if all(xnor(obj, _el)):
                return self
        self.order +=1
        self.generators.append(obj)
        self._items.append(obj)
        self._generate(obj)
        return self


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


def get_positive_stabilizer_groups(n_qubits, n_states):
    if n_states == n_stabilizer_states(n_qubits): 
        # If generating all states, we want to focus on only the all
        # positive signed operators
        target = n_states/pow(2, n_qubits)
    else:
        #If generating less than all, we'll add signs in randomly to compenstate
        target = n_states
    bitstrings = gen_bitstrings(n_qubits)
    shuffle(bitstrings)
    subspaces = []
    generators = []
    for group in combinations(bitstrings, n_qubits):
        groups = sorted(group, key=bool_to_int)
        if len(group) == 2:
            if not test_commutivity(n_qubits, group[0], group[1]):
                continue
        if len(group) > 2:
            if not all([test_commutivity(n_qubits, pair[0], pair[1]) 
                        for pair in combinations(group, 2)]): 
                continue
        candidate = BinarySubspace(*group)
        if len(candidate.generators) < n_qubits:
            continue
        if len(candidate._items) < pow(2,n_qubits):
            continue
        res = tuple(i for i in sorted(candidate._items, key=bool_to_int))
        if np.any([np.all([np.allclose(_el1, _el2) for _el1, _el2 in zip(res, space)]) 
                   for space in subspaces]):
            continue
        subspaces.append(res)
        generators.append(tuple(candidate.generators))
        if len(generators) == target:
            break
    return generators

def get_stabilizer_groups(n_qubits, n_states):
    positive_groups = get_positive_stabilizer_groups(n_qubits, n_states)
    extend = False
    if n_states == n_stabilizer_states(n_qubits):
        extend = True
        print("Found {} positive groups".format(len(positive_groups)))
    groups = [list(map(array_to_pauli, group)) for group in positive_groups]
    sign_strings = get_sign_strings(n_qubits, n_states)
    return add_sign_to_groups(groups, sign_strings, extend)