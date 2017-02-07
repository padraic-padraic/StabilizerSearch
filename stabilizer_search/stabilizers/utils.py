"""Module that provides miscellanious functions useful when generating
stabilizer states."""


import numpy as np

from bitarray import bitarray
from math import pow
from random import random, randrange

from ..mat import qeye, X, Y, Z


I = qeye(2)


__all__ = ['n_stabilizer_states', 'array_to_pauli', 'get_sign_strings']


def n_stabilizer_states(n_qubits):
    """Calculate the number of unique Stabilizer States for a given number of
    qubits."""
    res = pow(2., n_qubits)
    for i in range(n_qubits):
        res *= (pow(2., n_qubits-i)+1)
    return res


def array_to_pauli(bits):
    n = len(bits)//2
    pauli_chain = []
    for x, z in zip(bits[:n], bits[n:]):
        if not x and not z:
            pauli_chain.append(I)
        elif x and z:
            pauli_chain.append(Y)
        elif x and not z:
            pauli_chain.append(X)
        else:
            pauli_chain.append(Z)
    result = pauli_chain.pop(0)
    while pauli_chain:
        result = np.kron(result, pauli_chain.pop(0))
    return result


def get_sign_strings(n_qubits, n_states):
    sign_strings = []
    if n_states != n_stabilizer_states(n_qubits):
        for i in range(n_states):
            if random() > (1 / pow(2, n_qubits)): # Add a phase! Randomly...
                sign_num = bin(randrange(1,pow(2,n_qubits)))[2:]
                _bits = bitarray(n_qubits-len(sign_num))
                _bits.extend(sign_num)
                sign_strings.append(_bits)
    else:
        for i in range(1, pow(2, n_qubits)): #2^n different phase strings exist
            sign_num = bin(i)[2:]
            _bits = bitarray(n_qubits - len(sign_num))
            _bits.extend(sign_num)
            sign_strings.append(_bits)
    return sign_strings


def add_sign_to_groups(groups, sign_strings):
    if len(sign_strings) != pow(2, len(sign_strings[0])):
        for _bits in sign_strings:
                    for i in range(len(groups)):
                        groups.append([-1*p if b else p 
                                            for p, b in zip(groups[i], _bits)])
    else:
        for i in range(len(groups)):
            groups[i] = [-1*p if b else p
                                   for p, b in zip(groups[i], sign_strings[i])]
    return groups
