"""Module that provides miscellanious functions useful when generating
stabilizer states."""

from bitarray import bitarray
from math import pow
from qutip import qeye, sigmax, sigmay, sigmaz, tensor


I = qeye(2)
X = sigmax()
y = sigmay()
z = sigmaz()


__all__ = ['n_stabilizer_states', 'bitarray_to_pauli', 'phaseify_paulis']

def n_stabilizer_states(n_qubits):
    """Calculate the number of unique Stabilizer States for a given number of
    qubits."""
    res = pow(2., n_qubits)
    for i in range(n_qubits):
        res *= (pow(2.,n_qubits-i)+1)
    return res


def bitarray_to_pauli(bits):
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
    return tensor(pauli_chain)


def phaseify_paulis(n_qubits, pauli_generators):
    # Add phase 'by hand'
    phase_strings = []
    for i in range(1, pow(2, n_qubits)): #2^n different phase strings exist
        base = bin(i)[2:]
        a = bitarray(n_qubits - len(base))
        a.extend(base)
        phase_strings.append(a)
    for ps in phase_strings:
        for i in range(len(pauli_generators)):
            pauli_generators.append([-1*p if b else p 
                                    for p, b in zip(pauli_generators[i], ps)])
    return pauli_generators