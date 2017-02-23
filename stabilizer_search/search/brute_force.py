from ._search import _Search
from ._result import _Result
from ..linalg import ortho_projector
from ..mat import qeye
from ..stabilizers import get_stabilizer_states
from itertools import combinations
from math import factorial
from six import PY2
from random import shuffle

import numpy as np

def ncr(n, r):
    return factorial(n)//factorial(r)//factorial(n-r)


def do_brute_force(n_qubits, target_state, chi=None, **kwargs):
    """Function which performs the brute force search for stabilizer rank.
    Takes a number of qubits and the target state as input, and returns
    success: Bool, did the method succeed?
    chi: The rank found
    basis: the resulting decomposition"""
    dims = pow(2, n_qubits)
    stabilizers = get_stabilizer_states(n_qubits)
    shuffle(stabilizers)
    if chi is None:       
        for i in range(1, pow(2, n_qubits)):
            print('Test with {} states.'.format(i))
            for basis in combinations(stabilizers, i):
                projector = ortho_projector([b for b in basis])
                projection = np.linalg.norm(projector*target_state, 2)
                if np.allclose(projection, 1):
                    return True, i, basis
        I = qeye(pow(2, n_qubits))
        return False, dims, [I[:,i].reshape(dims, 1) for i in range(dims)]
    else:
        print('Searching brute force with chi={}'.format(chi))
        print('Got {} combinations to test'.format(ncr(len(stabilizers), chi)))
        for basis in combinations(stabilizers, chi):
            projector = ortho_projector([b for b in basis])
            projection = np.linalg.norm(projector*target_state, 2)
            if np.allclose(projection, 1):
                return True, chi, basis
        return False, chi, None


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
        if PY2:
            super(BruteForceResult, self).__init__(*args)
        else:
            super().__init__(*args)

    def parse_decomposition(self, decomposition):
        """Additional method for BruceForceResult that takes the decompositions
        and converts them to strings."""
        if decomposition is None:
            return "Bubkis"
        return "\n".join(str(state) for state in decomposition)

class BruteForceSearch(_Search):
    Result_Class = BruteForceResult
    func = staticmethod(do_brute_force)

    def __init__(self, *args, **kwargs):
        if PY2:
            super(BruteForceSearch, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
