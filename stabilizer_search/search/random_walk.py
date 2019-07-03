from math import exp
from six import PY2
from random import random, randrange

from ._search import _Search
from ._result import _Result
from ..core.linalg import get_projector, np_inc_in_list
from ..core.linalg import projection_distance, subspace_distance
from ..stabilizer import random_pauli_matrix
from ..stabilizer.seed_states import *

import numpy as np


def do_random_walk(n_qubits, stabilizer_states, target, distance_func,
                   beta_init=1, beta_max=4000, anneal_steps=100,
                   walk_steps=1000, real_only=False):
    projector = get_projector(stabilizer_states)
    distance = distance_func(target, projector)
    step_size = (beta_max-beta_init)/anneal_steps
    chi = len(stabilizer_states)
    betas = np.arange(beta_init, beta_max+step_size, step_size)
    Id = np.identity(pow(2, n_qubits), dtype=np.complex128)
    for beta in betas:
        for i in range(walk_steps):
            if np.allclose(distance, 0.):
                return True, len(stabilizer_states), stabilizer_states
            while True:
                move = (Id + random_pauli_matrix(n_qubits, real_only))
                move_target = randrange(chi)
                new_state = move @ stabilizer_states[move_target]
                if not np.any(np.nonzero(new_state)):
                    continue
                new_state = np.divide(new_state, np.linalg.norm(new_state, 2))
                if not np_inc_in_list(new_state, stabilizer_states):
                    break
            new_projector = get_projector([
                s if index != move_target else new_state
                for index, s in enumerate(stabilizer_states)
            ])
            new_distance = distance_func(target, new_projector)
            diff = new_distance - distance
            if diff < 0 or random() < exp(-1*beta*diff):
                distance = new_distance
                stabilizer_states[move_target] = new_state
    return False, chi, None


def random_walker(n_qubits, target, chi, **kwargs):
    real_only = kwargs.pop(
        'real_only',
        np.all(np.nonzero(np.imag(target))))
    seed = kwargs.pop('seed', 'random')
    if seed == 'random':
        inital_states = random_stabilizer_states(n_qubits, chi, real_only)
    elif seed == 'product':
        inital_states = random_prodct_states(n_qubits, chi, real_only)
    else:
        initial_states = random_computational_states(n_qubits, chi)
    if target.size() > target.shape[0]:
        distance_func = subspace_distance
    else:
        distance_func = projection_distance
    do_random_walk(n_qubits, initial_states, target, distance_func,
                   real_only=real_only, **kwargs)


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
    func = staticmethod(random_walker)

    def __init__(self, *args, **kwargs):
        _f = kwargs.pop('func', None)
        if _f is not None:
            self.func = staticmethod(_f)
        if PY2:
            super(RandomWalkSearch, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
