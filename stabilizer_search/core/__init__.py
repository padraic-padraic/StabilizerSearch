from random import seed

from .linalg import *
from .unitaries import *

seed()
SEEDED_RANDOM = True


def n_stabilizer_states(n_qubits):
    """Calculate the number of unique Stabilizer States for a given number of
    qubits."""
    res = pow(2., n_qubits)
    for i in range(n_qubits):
        res *= (pow(2., n_qubits-i)+1)
    return int(res)
