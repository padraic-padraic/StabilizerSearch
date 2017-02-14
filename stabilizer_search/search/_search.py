from ._result import _Result

def dummy_func(n_qubits, target_state, *args, **kwargs):
    """Dummy method for the search class."""
    print("This should be overridden yo.")
    return True, 0, ['No', 'Result']


class _Search(object):
    """Callable base class for the Stabilizer Decomposition searches. This
    class is overridden by specifying a derived instance of _Result, and a
    function that takes the same arguemnts as search.dummy_func, returning if
    the method succeeded, and the resulting stabilizer rank
    and decomposition."""
    Result_Class = _Result
    func = staticmethod(dummy_func)

    def __init__(self, target_state, target_state_string,
                 n_qubits, *args, **kwargs):
        self.target_state = target_state
        self.target_state_string = target_state_string
        self.n_qubits = n_qubits
        self.f_args = args
        self.f_kwargs = kwargs

    def __call__(self):
        success, chi, decomposition = self.func(self.n_qubits, self.target_state,
                                           *self.f_args, **self.f_kwargs)
        return self.Result_Class(self.target_state_string, self.n_qubits,
                                 chi, success, decomposition)
