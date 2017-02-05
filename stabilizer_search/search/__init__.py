"""The StabilizerSearch.search module defines a pair of base classes, 
_Search and _Result, that are used to find and return the Stabilizer Rank
decomposition of a given state.

The submodules each define a strategy for finding the Stabilizer Rank."""


class _Result(object):
    ostring = None
    def __init__(self, target_state_string, n_qubits, success, decomposition):
        self.target_state = target_state_string
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


def dummy_func(n_qubits, target_state, *args, **kwargs):
    """Dummy method for the search class."""
    print("This should be overridden yo.")
    return True, ['No', 'Result']


class _Search(object):
    """Callable base class for the Stabilizer Decomposition searches. This
    class is overridden by specifying a derived instance of _Result, and a
    function that takes the same arguemnts as search.dummy_func, returning if
    the method succeeded and the resulting decomposition."""
    Result_Class = _Result
    func = dummy_func

    def __init__(self, target_state, target_state_string,
                 n_qubits, *args, **kwargs):
        self.target_state = target_state
        self.target_state_string = target_state_string
        self.n_qubits = n_qubits
        self.f_args = args
        self.f_kwargs = kwargs

    def __call__(self):
        success, decomposition = self.func(self.n_qubits, self.target_state,
                                           *self.f_args, **self.f_kwargs)
        return self.Result_Class(self.target_state_string, self.n_qubits,
                                 success, decomposition)


        