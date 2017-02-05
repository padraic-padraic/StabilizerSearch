from . import _Search, _Result
from ..linalg import OrthoProjector
from ..stabilizer import get_stabilizer_states
from itertools import combinations
from six import PY2


def do_BruteForce(n_qubits, target_state, *args, **kwargs):
    stabilizers = get_stabilizer_states(n_qubits)
    for i in range(1, pow(2, n_qubits)):
        for basis in combinations(stabilizers, i):
            projector = OrthoProjector([b.full() for b in basis])
            projection = np.linalg.norm(prif*target_state.full(), 2)
            if np.allclose(projection, 1):
                return True, i, basis
    return False, pow(2, n), [qt.basis(pow(2,n), i) for i in range(pow(2, n))]


class BruteForceResult(_Result):
    ostring = """
    The Brute Force method for the state {target_state} on {n_qubits} qubits 
    {success}.
    We found a decomposition with stabilizer rank {chi}, which looked like:
    {decomposition}.
    """

    def __init__(self, *args):
        args[-1] = self.parse_decomposition(args[-1])
        if PY2:
            super(BruceForceResult, self).__init__(*args)
        else:
            super().__init__(*args)

    def parse_decomposition(self, decomposition):
        """Additional method for BruceForceResult that takes the decompositions
        and converts them to strings."""
        return "\n".join(str(state) for state in decomposition)

class BruteForceSearch(_Search):
    Result_Class = BruteForceResult
    func = do_BruteForce

    def __init__(self, target_state, n_qubits*args, **kwargs):
        if PY2:
            super(BruteForceSearch, self).__init__(*args, *kwargs)
        super().__init__(*args, **kwargs)
