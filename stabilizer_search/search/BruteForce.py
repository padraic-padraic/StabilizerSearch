from . import _Search, _Result


def do_BruteForce(n_qubits, target_state, *args, **kwargs):
    pass


class BruteForceResult(_Result):
    ostring = """
    The Brute Force method for the state {target_state} on {n_qubits} qubits 
    {success}.
    We found a decomposition with stabilizer rank {chi}, which looked like:
    {decomposition}.
    """

    def __init__(self, *args):
        args[-1] = self.parse_decomposition(args[-1])
        super().__init__(*args)

    def parse_decomposition(self, decomposition):
        """Additional method for BruceForceResult that takes the decompositions
        and converts them to strings."""
        return "\n".join(str(state) for state in decomposition)

class BruteForceSearch(_Search):
    Result_Class = BruteForceResult
    func = do_BruteForce

    def __init__(self, target_state, n_qubits*args, **kwargs):
        super().__init__(*args, **kwargs)
