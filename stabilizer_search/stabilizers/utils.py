"""Module that provides miscellanious functions useful when generating
stabilizer states."""


import numpy as np

from random import random, randint

from ..mat import qeye, X, Y, Z, tensor


I = qeye(2)


__all__ = ['n_stabilizer_states', 'array_to_pauli', 'array_to_string',
           'get_sign_strings','bool_to_int', 'add_sign_to_groups',
           'states_from_file', 'states_to_file', 'gens_from_file', 'gens_to_file']


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
    return int(res)


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
    return tensor(*pauli_chain)


def array_to_string(bits):
    n = len(bits)//2
    literals =''
    for x, z in zip(bits[:n], bits[n:]):
        if not x and not z:
            literals += 'I'
        elif x and z:
            literals += 'Y'
        elif x and not z:
            literals += 'X'
        else:
            literals += 'Z'
    return literals

def pauli_from_string(_str):
    pauli_chain = []
    for literal in _str:
        if literal=='I':
            pauli_chain.append(I)
        elif literal=='X':
            pauli_chain.append(X)
        elif literal=='Y':
            pauli_chain.append(Y)
        else:
            pauli_chain.append(Z)
    return tensor(*pauli_chain)

def array_from_string(_str):
    n_qubits = len(_str)
    bits = np.array([False]*2*n_qubits)
    for n, literal in enumerate(_str):
        if literal == 'X':
            bits[n] = True
        elif literal == 'Y':
            bits[n] = True
            bits[n+n_qubits] = True
        elif literal == 'Z':
            bits[n+n_qubits] = True
    return bits

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


def group_to_file(gen_set, _f):
    _f.write('GROUP\n')
    for pauli in gen_set:
        _f.write('{}\n'.format(array_to_string(pauli)))
    _f.write('ENDGROUP\n\n')


def gens_to_file(generators, _f):
    n_qubits = len(generators[0])
    n_positive_groups = n_stabilizer_states(n_qubits) // pow(2,n_qubits)
    for i in range(n_positive_groups):
        group_to_file(generators[i], _f)


def states_to_file(states, _f):
    for state in states:
        _f.write('STATE\n')
        for _el in state:
            _f.write('({}, {})\n'.format(np.asscalar(np.real(_el)), 
                                       np.asscalar(np.imag(_el))))
        _f.write('ENDSTATE\n\n')


def gens_from_file(f, n_qubits):
    positive_generators=[]
    line = f.readline()
    while line:
        line = line.strip()
        if line=='GROUP':
            group = []
            while True:
                line = f.readline()
                line = line.strip()
                if line=='ENDGROUP':
                    positive_generators.append(group)
                    break
                group.append(array_from_string(line))
        line = f.readline()
    return positive_generators


def states_from_file(f, n_qubits):
    states = []
    line = f.readline()
    while line:
        line = line.strip()
        if line=='STATE':
            state = np.matrix(np.zeros((pow(2,n_qubits), 1), dtype=np.complex_))
            counter = 0
            while True:
                line = f.readline()
                line = line.strip()
                if line=='ENDSTATE':
                    states.append(state)
                    break
                line = line.strip('()').split(',')
                line = [l.strip() for l in line]
                state[counter] = float(line[0])+1j*float(line[1])
                counter +=1
        line = f.readline()
    return states
