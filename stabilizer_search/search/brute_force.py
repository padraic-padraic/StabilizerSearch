from itertools import combinations
from math import factorial
from six import PY2
from random import shuffle

from ._search import _Search
from ._result import _Result
from ..core.linalg import get_projector, projection_distance, subspace_distance
from ..stabilizers import get_stabilizer_states

from numba import njit

import numpy as np


def ncr(n, r):
    return factorial(n)//factorial(r)//factorial(n-r)


def do_brute_force(n_qubits, stabilizer_states, target, distance_func,
                   chi=None, lower_bound=1, real_only=False):
    """Function which performs the brute force search for stabilizer rank.
    Takes a number of qubits and the target state as input, and returns
    success: Bool, did the method succeed?
    chi: The rank found
    basis: the resulting decomposition"""
    dims = pow(2, n_qubits)
    shuffle(stabilizer_states)
    if chi is None:
        for i in range(lower_bound, pow(2, n_qubits)):
            print('Test with {} states.'.format(i))
            for basis in combinations(stabilizer_states, i):
                projector = get_projector([b for b in basis])
                distance = distance_func(target, projector)
                if np.allclose(distance, 0.):
                    return True, i, basis
        return False, dims, None
    else:
        print('Searching brute force with chi={}'.format(chi))
        # print('Got {} combinations to test'.format(ncr(len(stabilizer_states), chi)))
        for basis in combinations(stabilizer_states, chi):
            projector = get_projector([b for b in basis])
            distance = distance_func(target, projector)
            if np.allclose(distance, 0.):
                return True, chi, basis
        return False, chi, None


def brute_force_search(n_qubits, target, **kwargs):
    real_only = kwargs.pop(
        'real_only',
        np.all(np.nonzero(np.imag(target))))
    stabilizer_states = get_stabilizer_states(
        n_qubits, real_only=real_only)
    if target.shape[1] == 1:
        distance_func = projection_distance
    else:
        distance_func = subspace_distance
    do_brute_force(n_qubits, stabilizer_states, target, distance_func,
                   real_only=real_only, **kwargs)


class BruteForceResult(_Result):
    ostring = """
    The Brute Force method for the state {target_state} on {n_qubits} qubits
    {success}.
    We found a decomposition with stabilizer rank {chi}, which looked like:
    {decomposition}.
    """

    def __init__(self, *args):
        args = list(args)
        self.basis = args[-1]
        args[-1] = self.parse_decomposition(args[-1])
        super(BruteForceResult, self).__init__(*args)

    def parse_decomposition(self, decomposition):
        """Additional method for BruceForceResult that takes the decompositions
        and converts them to strings."""
        if decomposition is None:
            return "Bubkis"
        return "\n".join(str(state) for state in decomposition)

class BruteForceSearch(_Search):
    Result_Class = BruteForceResult
    func = staticmethod(brute_force_search)

    def __init__(self, *args, **kwargs):
        super(BruteForceSearch, self).__init__(*args, **kwargs)
