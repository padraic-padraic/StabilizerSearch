"""Module for obtaining Stabilizer States as Numpy arrays. The main method 
provided by this module is `get_stabilizer_states`, which is also available in 
the package namespace.

Potential future plans would allow a more general representation than Numpy 
states. Alternatively, we could look to implement a wrapper function capable of 
transforming between a Numpy matrix and other representations."""

from random import sample

from ..clib.c_stabilizers import c_get_stabilizer_groups, c_get_eigenstates
from .eigenstates import py_find_eigenstates
from .py_generators import get_positive_stabilizer_groups as py_get_groups
from .py_generators import PauliArray
from .utils import *

import os.path as path


__all__ = ['get_stabilizer_states']


def group_to_file(gen_set, _f):
    _f.write('GROUP\n')
    for pauli in gen_set:
        _f.write('{}\n'.format(array_to_string(pauli)))
    _f.write('ENDGROUP\n\n')


def gens_to_file(generators, _f):
    n_qubits = len(generators[0])
    n_positive_groups = n_stabilizer_states(n_qubits) // pow(2,n_qubits)
    for i in range(n_positive_groups):
        group_to_file(generators[i], _f)


def states_to_file(states, _f):
    for state in states:
        _f.write('STATE\n')
        for _el in state:
            _f.write('({}, {})\n'.format(np.asscalar(np.real(_el)), 
                                       np.asscalar(np.imag(_el))))
        _f.write('ENDSTATE\n\n')


def gens_from_file(f, n_qubits):
    positive_generators=[]
    line = f.readline()
    while line:
        line = line.strip()
        if line=='GROUP':
            group = []
            while True:
                line = f.readline()
                line = line.strip()
                if line=='ENDGROUP':
                    positive_generators.append(group)
                    break
                group.append(PauliArray.from_string(line))
        line = f.readline()
    return positive_generators


def states_from_file(f, n_qubits):
    states = []
    line = f.readline()
    while line:
        line = line.strip()
        if line=='STATE':
            state = np.matrix(np.zeros((pow(2,n_qubits), 1), dtype=np.complex_))
            counter = 0
            while True:
                line = f.readline()
                line = line.strip()
                if line=='ENDSTATE':
                    states.append(state)
                    break
                line = line.strip('()').split(',')
                line = [l.strip() for l in line]
                state[counter] = float(line[0])+1j*float(line[1])
                counter +=1
        line = f.readline()
    return states


APP_DIR = path.abspath(__file__)
STATE_STRING = '{}.states'
GROUP_STRING = '{}.generators'
LOAD_METHODS = {'states':states_from_file,
               'generators':gens_from_file}
WRITE_METHODS = {'states': states_to_file,
                 'generators': gens_to_file} 

def try_load(_type, format_string, n_qubits, n_states, real_only=False):
    loader = LOAD_METHODS.get(_type, None)
    if loader is None:
        raise KeyError("Don't know how to load: "+_type)
    f_string = format_string.format(n_qubits)
    package_path = path.join(APP_DIR, 'data', f_string)
    rel_path = path.join('./', f_string)
    if path.exists(package_path):
        with open(package_path, 'r') as _f:
            items = loader(_f, n_qubits)
    elif path.exists(rel_path):
        with open(rel_path, 'r') as _f:
            items = loader(_f, n_qubits)
    else:
        items = None
    if items is not None:
        if real_only:
            items = [i for i in items if is_real(i)]
            if n_states != n_stabilizer_states(n_qubits) and len(items) < n_states:
                if type=='states':
                    raise RuntimeError('There are insufficient real states on {} qubits'.format(n_qubits))
                else:
                    if pow(2,n_qubits)*len(items) < n_states:
                        raise RuntimeError('There are insufficient real states on {} qubits'.format(n_qubits))
        if n_states!= n_stabilizer_states(n_qubits):
            if _type!='groups' or n_states < n_stabilizer_states(n_qubits)//pow(2,n_qubits):
                return sample(items, n_states)
    return items


def save_to_file(_type, items, format_string, n_qubits):
    writer = WRITE_METHODS.get(_type, None)
    if writer is None:
        raise KeyError("Don't know how to store " + _type)
    with open(path.join('./', format_string.format(n_qubits)), 'w') as f:
        writer(items, f)
    return


def get_stabilizer_states(n_qubits, n_states=None, **kwargs):
    """Method for returning a set of stabilizer states. It takes the following 
    arguments:
    Positional:
      n_qubits: The number of qubits our stabilizer states will be built out of.
      n_states (Optional): Number of stabilizer states we require. If not 
      specified, defaults to all Stabilier states.
    Keyword:
      use_cached: Boolean, defaults to True and looks in the package or working 
      dir for serialised states or generators.
      generator_backend: Function which searches for the stabilizer generators
      eigenstate_backend: Function which takes sets of stabilizer generators and
      builds the corresponding eigenstates.
      real_only: Return only real-valued stabilizer states
    """
    use_cached = kwargs.pop('use_cached', True)
    generator_func = kwargs.pop('generator_backend', py_get_groups)
    eigenstate_func = kwargs.pop('eigenstate_backend', py_find_eigenstates)
    real_only = kwargs.pop('real_only', False)
    stabilizer_states = None
    if n_states is None:
        get_all = True
        n_states = n_stabilizer_states(n_qubits)
    else:
        get_all = (n_states==n_stabilizer_states(n_qubits))
    if use_cached:
        stabilizer_states = try_load('states',STATE_STRING, n_qubits, n_states,
                                     real_only)
        if stabilizer_states is None:
            groups = try_load('generators', GROUP_STRING, n_qubits, n_states,
                              real_only)
            if groups is not None:
                stabilizer_states = eigenstate_func(groups, n_states)
                if get_all:
                    save_to_file('states', stabilizer_states, STATE_STRING,
                                 n_qubits)
    if stabilizer_states is None:
        generators = generator_func(n_qubits, n_states, real_only)
        stabilizer_states = eigenstate_func(generators, n_states)
        if use_cached and get_all and not real_only:
            save_to_file('generators', generators, GROUP_STRING, n_qubits)
            save_to_file('states', stabilizer_states, STATE_STRING, n_qubits)
    return stabilizer_states
