"""The StabilizerSearch.search module defines a pair of base classes, 
_Search and _Result, that are used to find and return the Stabilizer Rank
decomposition of a given state.

The submodules each define a strategy for finding the Stabilizer Rank."""


class _Result(object):
    ostring = None
    def __init__(self, target_state, n_qubits, success, decomposition):
        self.target_state = target_state
        self.n_qubits = n_qubits
        self.success = success
        self.decomposition = decomposition

    def __str__(self):
        if self.ostring is None:
            return """This is the base class. You should not ever be doing this.
                   Why are you doing this. 😠"""
        return self.ostring.format(target_state=self.target_state,
                                   n_qubits=self.n_qubits,
                                   success=self.success,
                                   decomposition=self.decomposition)


class _Search(object):
    Result_Class = _Result
    def __init__(self, target_state, n_qubits, success, decomposition):
        self.target_state = target_state
        self.n_qubits = n_qubits
        self.success = success
        self.decomposition = decomposition

    def __call__(self):
        return self.Result_Class(self.target_state, self.n_qubits, self.success, 
                            self.decomposition)


        