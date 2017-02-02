"""Module that provides miscellanious functions useful when generating
stabilizer states."""

from math import pow


def n_stabilizer_states(n_qubits):
    """Calculate the number of unique Stabilizer States for a given number of
    qubits."""
    res = pow(2., n_qubits)
    for i in range(n_qubits):
        res *= (pow(2.,n_qubits-i)+1)
    return res
