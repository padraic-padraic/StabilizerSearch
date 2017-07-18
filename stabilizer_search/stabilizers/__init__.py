"""Module for obtaining Stabilizer States as Numpy arrays. The main method
provided by this module is `get_stabilizer_states`, which is also available in
the package namespace.

Potential future plans would allow a more general representation than Numpy
states. Alternatively, we could look to implement a wrapper function capable of
transforming between a Numpy matrix and other representations."""

from random import sample

from ..clib.c_stabilizers import c_get_stabilizer_groups, c_get_eigenstates
from ..clib.c_stabilizers import SymplecticPauli, StabilizerMatrix
from .eigenstates import py_find_eigenstates
from .py_generators import get_positive_stabilizer_groups as py_get_groups
from .py_generators import GeneratorSet, PauliArray
from .seed_states import random_computational_states, random_product_states
from .utils import *

import os.path as path
import numpy as np


__all__ = ['get_stabilizer_states']


def gens_to_file(generators, _f):
    n_qubits = len(generators[0])
    n_positive_groups = n_stabilizer_states(n_qubits) // pow(2,n_qubits)
    for i in range(n_positive_groups):
        _f.write('GROUP\n')
        for pauli in generators[i]:
            _f.write('{}\n'.format(str(pauli)))
        _f.write('ENDGROUP\n\n')


def states_to_file(states, _f):
    for state in states:
        _f.write('STATE\n')
        for _el in state:
            _f.write('({}, {})\n'.format(np.asscalar(np.real(_el)),
                                       np.asscalar(np.imag(_el))))
        _f.write('ENDSTATE\n\n')


def gens_from_file(f, PauliClass=SymplecticPauli, GroupClass=GeneratorSet):
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
                group.append(PauliClass.from_string(line))
        line = f.readline()
    return positive_generators


def states_from_file(f, n_qubits):
    states = []
    line = f.readline()
    while line:
        line = line.strip()
        if line=='STATE':
            state = np.array(np.zeros((pow(2,n_qubits)), dtype=np.complex_))
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
WRITE_METHODS = {'generators': gens_to_file,
                 'states': states_to_file}
METHODS = {'c':{'generators':c_get_stabilizer_groups,
                'eigenstates':c_get_eigenstates},
           'python':{'generators':py_get_groups,
                     'eigenstates':py_find_eigenstates}
          }
CLASSES = {'c':{'paulis':SymplecticPauli,
                'groups':StabilizerMatrix},
           'python':{'paulis':PauliArray,
                    'groups':GeneratorSet},
          }
SEEDED_STATES = {'computational':random_computational_states,
                 'product': random_product_states
                 }

def get_file_path(format_string, n_qubits):
    f_string = format_string.format(n_qubits)
    _path = path.join(APP_DIR, 'data', f_string)
    if not path.exists(_path):
        _path = path.join('./', f_string)
        if not path.exists(_path):
            return None
    return _path


def load_groups(n_qubits, n_states, real_only=False,
                PauliClass=SymplecticPauli, GroupClass=GeneratorSet):
    f_path = get_file_path(GROUP_STRING, n_qubits)
    if f_path is not None:
        with open(f_path, 'r') as f:
            groups = gens_from_file(f, PauliClass, GroupClass)
        if real_only:
            groups = [g for g in groups if is_real(i)]
            if len(groups) < n_states:
                if pow(2, n_qubits)*len(groups) < n_states:
                    raise RuntimeError('There are insufficient real states on {} qubits'.format(n_qubits))
        if n_states < n_stabilizer_states(n_qubits)//pow(2,n_qubits):
            return sample(groups, n_states)
    else:
        return None


def load_states(n_qubits, n_states, real_only=False):
    f_path = get_file_path(STATE_STRING, n_qubits)
    if f_path is not None:
        with open(f_path, 'r') as f:
            states = states_from_file(f, n_qubits)
        if real_only:
            states = [s for s in states if is_real(s)]
            if len(states) < n_states:
                raise AttributeError('There are insufficient real states on {} qubits'.format(n_qubits))
        if n_states < len(states):
            return sample(states, n_states)
        else:
            return states
    else:
        return None


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
    generator_func = kwargs.pop('generator_backend', None)
    eigenstate_func = kwargs.pop('eigenstate_backend', None)
    real_only = kwargs.pop('real_only', False)
    seed = kwargs.pop('seed', None)
    if seed is not None:
        if n_states > pow(2,n_qubits):
            raise NotImplementedError("Seeded states are only used in finding Stabilizer Rank decompositions, and generating more than 2^n states is not implemented.")
        return SEEDED_STATES[seed](n_qubits, n_states, real_only=real_only)
    PauliClass = kwargs.pop('pauli_class', None)
    GroupClass = kwargs.pop('group_class', None)
    backend = kwargs.pop('backend', 'c')
    verbose = kwargs.pop('verbose', None)
    if generator_func is None:
        generator_func = METHODS[backend]['generators']
    if eigenstate_func is None:
        eigenstate_func = METHODS[backend]['eigenstates']
    if PauliClass is None:
        PauliClass = CLASSES[backend]['paulis']
    if GroupClass is None:
        GroupClass = CLASSES[backend]['groups']
    if verbose and backend != 'c':
        raise NotImplementedError('Only the c backend knows about verbose generation.')
    stabilizer_states = None
    if n_states is None:
        get_all = True
        n_states = n_stabilizer_states(n_qubits)
    else:
        get_all = (n_states==n_stabilizer_states(n_qubits))
    if use_cached:
        stabilizer_states = load_states(n_qubits, n_states, real_only)
        if stabilizer_states is None:
            groups = load_groups(n_qubits, n_states, real_only,
                                 PauliClass, GroupClass)
            if groups is not None:
                stabilizer_states = eigenstate_func(groups, n_states)
                if get_all:
                    save_to_file('states', stabilizer_states, STATE_STRING,
                                 n_qubits)
    if stabilizer_states is None:
        if not verbose:
            generators = generator_func(n_qubits, n_states, real_only)
            stabilizer_states = eigenstate_func(generators, n_states)
        else:
            generators = generator_func(n_qubits, n_states, real_only, verbose)
            stabilizer_states = eigenstate_func(generators, n_states, verbose)
        if use_cached and get_all and not real_only:
            save_to_file('generators', generators, GROUP_STRING, n_qubits)
            save_to_file('states', stabilizer_states, STATE_STRING, n_qubits)
    return stabilizer_states
