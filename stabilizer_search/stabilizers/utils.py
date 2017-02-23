"""Module that provides miscellanious functions useful when generating
stabilizer states."""


import numpy as np

from random import random, randint

from ..mat import qeye, X, Y, Z


I = qeye(2)


__all__ = ['n_stabilizer_states', 'array_to_pauli', 'get_sign_strings',
            'bool_to_int', 'add_sign_to_groups']


def bool_to_int(bits):
    tot = 0
    for i in range(len(bits)):
        if bits[-1-i]:
            tot += pow(2,i)
    return tot

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
                sign_num = bin(randint(1, pow(2, n_qubits)))[2:]
                sign_num = '0'*(n_qubits - len(sign_num)) + sign_num
                _a = np.array([b == '1' for b in sign_num])
                sign_strings.append(_a)
            else:
                sign_strings.append(np.array([False]*n_qubits))
    else:
        for i in range(1, pow(2, n_qubits)): #2^n -1 different phase strings exist
            sign_num = bin(i)[2:]
            sign_num = '0'*(n_qubits - len(sign_num)) + sign_num
            _a = np.array([b == '1' for b in sign_num])
            sign_strings.append(_a)
    return sign_strings


def add_sign_to_groups(groups, sign_strings, extend):
    if extend:
        for i in range(len(groups)):
            for _bits in sign_strings:
                        groups.append([-1*p if b else p 
                                            for p, b in zip(groups[i], _bits)])
        print('Added sign to produce {} total groups'.format(len(groups)))
    else:
        for i in range(len(groups)):
            groups[i] = [-1*p if b else p
                                   for p, b in zip(groups[i], sign_strings[i])]
    return groups
