"""Module for obtaining Stabilizer States as QObj instances. The main method 
provided by this module is `get_stabilizer_states`, which is also available in 
the package namespace.

Potential future plans would allow a more general representation than QObj 
states. Alternatively, we could look to implement a wrapper function capable of 
transforming between a QObj and the raw vector."""

from .utils import n_stabilizer_states

def try_load_states(n_qubits, n_states=None):
    pass


def try_load_groups(n_qubits, n_states):
    pass


def get_stabilizer_states(n_qubits, n_states=None, **kwargs):
    """Method for returning a set of stabilizer states. It takes the following 
    arguments:
    Positional:
    n_qubits: The number of qubits our stabilizer states will be built out of.
    n_states (Optional): Number of stabilizer states we require. If not specified,
    defaults to all Stabilier states.
    Keyword:
    use_cached: Boolean, defaults to True and looks in the package or working dir
    for serialised states or generators.
    generator_backend: Function which searches for the stabilizer generators
    eigenstate_backend: Function which takes sets of stabilizer generators and
    builds the corresponding eigenstates.
    """
    use_cached = kwargs.get('use_cached', True)
    generator_func = kwargs.get('generator_backend')
    eigenstate_func = kwargs.get('eigenstate_backend')


