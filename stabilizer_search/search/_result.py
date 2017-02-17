class _Result(object):
    """Base class to handle pretty-printing of results from a stabilizer rank
    search. Inhertied classes should over-write ostring with a format string
    which takes the arguments:
    target_state: A string description to the target state.
    n_qubits: The number of qubits in the search
    chi: The number returned by the search
    success: Did the method converge/find a smaller stabilizer rank.
    decomposition: A string description of the decomposition found. """
    ostring = None
    def __init__(self, target_state_string, n_qubits, chi,
                 success, decomposition):
        self.target_state = target_state_string
        self.n_qubits = n_qubits
        self.success = success
        self.success_string = 'succeeded' if success else 'did not succeed'
        self.chi = chi
        self.decomposition = decomposition

    def __str__(self):
        if self.ostring is None:
            return """This is the base class. You should not ever be doing this.
                   Why are you doing this. """
        return self.ostring.format(target_state=self.target_state,
                                   n_qubits=self.n_qubits,
                                   success=self.success_string,
                                   chi = self.chi,
                                   decomposition=self.decomposition)