from math import exp
from six import PY2
from random import random, randint, randrange

from ._search import _Search
from ._result import _Result
from .cy_do_random_walk import cy_do_random_walk
from ..linalg import calculate_overlap, ortho_projector
from ..mat import qeye
from ..stabilizers import get_stabilizer_states
from ..stabilizers.utils import array_to_pauli

import numpy as np
import sys

def test_real(n_qubits, bits):
    x_bits, z_bits = bits[:n_qubits], bits[n_qubits:]
    _sum = 0
    for i in range(n_qubits):
        if x_bits[i]&z_bits[1]:
            _sum +=1
    return _sum%2==0


def random_pauli(n_qubits, real_only):
    while True:
        base = bin(randint(1, pow(2, 2*n_qubits)))[2:]
        bits = '0'*(2*n_qubits - len(base)) + base
        bits = np.array([b == '1' for b in bits])
        if real_only:
            if test_real(bits):
                break
        else:
            break
    pauli = array_to_pauli(bits)
    if random() > 0.5:
        return -1* pauli
    return pauli

def do_random_walk(n_qubits, target_state, chi, **kwargs):
    # print('Searching with chi={}'.format(chi))
    beta = kwargs.pop('beta_init', 1)
    beta_max = kwargs.pop('beta_max', 4000)
    anneal_steps = kwargs.pop('steps', 100)
    beta_diff = (beta_max-beta)/anneal_steps
    walk_steps = kwargs.pop('M', 1000)
    real = np.allclose(np.imag(target_state), 0.)
    stabilizers = get_stabilizer_states(n_qubits, chi, real_only=real)
    I = qeye(pow(2, n_qubits))
    projector = ortho_projector([s for s in stabilizers])
    overlap = np.linalg.norm(projector*target_state, 2)
    while beta <= beta_max:
        # print("Anneal Progress : {}%".format((beta-1)/beta_diff))
        for i in range(walk_steps):
            if np.allclose(overlap, 1.):
                return True, chi, stabilizers
            while True:
                move = random_pauli(n_qubits, real)
                move_target = randrange(chi)
                new_state = (I+move) * stabilizers[move_target]
                if not np.allclose(new_state, 0.): #Accept the move only if the resulting state is not null!
                    break
            new_state = new_state / np.linalg.norm(new_state, 2)
            new_projector = ortho_projector([s if n != move_target 
                                            else new_state for n, s in 
                                            enumerate(stabilizers)])
            new_overlap = np.linalg.norm(new_projector*target_state, 2)
            # print('New Overlap is {}'.format(new_overlap))
            if new_overlap > overlap:
                overlap = new_overlap
                stabilizers[move_target] = new_state
            else:
                p_accept = exp(-beta*(overlap - new_overlap))
                if random() < p_accept:
                    overlap = new_overlap
                    stabilizers[move_target] = new_state
        beta += beta_diff
    return False, chi, None

class RandomWalkResult(_Result):
    ostring = """
    The Random Walk method for the state {target_state} on {n_qubits} qubits,
    {success} in finding a decomposition with stabilizer rank {chi}.
    {decomposition}
    """

    def __init__(self, *args):
        args = list(args)
        self.basis = args[-1]
        args[-1] = self.parse_decomposition(args[-1])
        if PY2:
            super(RandomWalkResult, self).__init__(*args)
        else:
            super().__init__(*args)

    def parse_decomposition(self, decomposition):
        if decomposition is None:
            return ''
        else:
            return "\n".join(str(state) for state in decomposition)


class RandomWalkSearch(_Search):
    Result_Class = RandomWalkResult
    func = staticmethod(cy_do_random_walk)

    def __init__(self, *args, **kwargs):
        _f = kwargs.pop('func', None)
        if _f is not None:
            self.func = staticmethod(_f)
        if PY2:
            super(RandomWalkSearch, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

