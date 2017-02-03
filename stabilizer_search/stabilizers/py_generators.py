"""Module which finds unique generating sets for pauli Stabilier groups using
python code and the bitarray module."""


from bitarray import bitarray
from functools import reduce
from itertools import combinations
from random import sample, randrange, random

import operator as op



__all__ = ['get_positive_stabilizer_groups']


def xnor(a,b):
    """Define the xnor operation between bitarrays"""
    return (a&b)^(~a&~b)


class BinarySubspace(object):
    """Set-like class for bitarray objects to generate a closed subspace."""
    def __init__(self, *data):
        self.order = 0
        self._items = []
        self.generators = []
        for val in data:
            if not isinstance(val, bitarray):
                raise ValueError('This class works for bitarrays only!')
            self.add(val)
    
    def __contains__(self, it):
        for _el in self._items:
            if all(xnor(_el, it)):
                return True
        return False

    def __iter__(self):
        for item in self._items:
            yield item

    def _generate(self, obj):
        for item in self._items:
            new = item^obj
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


def ncr(n, r):
    """Efficient evaluation of ncr, taken from StackOverflow
    http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python"""
    r = min(r, n-r)
    if r == 0: 
        return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom


def random_combination(iterable, r):
    """Random selection from itertools.combinations(iterable, r)
    Taken from itertools.recipes on pydoc"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple(pool[i] for i in indices)


def symplectic_inner_product(n, a, b):
    """Symplectic inner product between two Pauli operators in the binary 
    representation; equivlanent to testing commutivity (see test_commutivity)."""
    x_a, z_a = a[:n], a[n:]
    x_b, z_b = b[:n], b[n:]
    count = (x_a&z_b).count() + (x_b&z_a).count()
    return count%2


def test_commutivity(n, bits1, bits2):
    return symplectic_inner_product(n, bits1, bits2) == 0 #1 if they anticommute, 0 if they commute


def gen_bitstrings(n):
    bitstrings = []
    for i in range(1, pow(2,2*n)): #We ignore the all 0 string as it corresponds to I^{n}
        bin_string = bin(i)[2:] #strip the 0b from the string
        a = bitarray(2*n - len(bin_string))
        a.extend(bin_string)
        bitstrings.append(a)
    return bitstrings


def get_positive_stabilizer_groups(n_qubits, n_states):
    bitstrings = gen_bitstrings(n_qubits)
    subspaces = []
    generators = []
    for group in random_combination(combinations(bitstrings, n_qubits),
                                    ncr(len(bitstrings), n_qubits)):
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
        res = tuple(i for i in sorted(candidate._items))
        if not res in subspaces:
            subspaces.append(res)
            generators.append(tuple(candidate.generators))
        if len(generators) == n_states:
            break
    phase_strings = []
    for i in range(1, pow(2, n_qubits)): #2^n different phase strings exist
        base = bin(i)[2:]
        _a = bitarray(n_qubits - len(base))
        _a.extend(base)
        phase_strings.append(_a)
    for ps in phase_strings:
        for i in range(len(generators)):
            generators.append([-1*p if b else p 
                                    for p, b in zip(generators[i], ps)])
    return generators