from . import _Search, _Result
from ..lingalg import ortho_projector
from ..stabilizer import get_stabilizer_states
from six import PY2


def do_RandomWalk(*args, **kwargs):
    pass


class RandomWalkResult(_Result):
    ostring = """
    The Random Walk method for the state {target_state} on {n_qubits} qubits,
    {success} in finding a decomposition with stabilizer rank {chi}.
    {decomposition}
    """

    def __init__(self, *args):
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
    func = do_RandomWalk

    def __init__(self, *args, **kwargs):
        if PY2:
            super(RandomWalkSearch, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
