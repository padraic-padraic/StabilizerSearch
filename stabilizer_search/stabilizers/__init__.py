from .smatrix import *
from ..core import n_stabilizer_states, SEEDED_RANDOM

from math import ceil, pow
from os.path import exists as path_exists, join as path_join, realpath
from random import shuffle, seed

DIR = os.path.split(realpath(__FILE__))

if not SEEDED_RANDOM:
	seed()
	SEEDED_RANDOM=True


@jit
def get_blocks_and_lines(file_path, delimiter):
	file = open(file_path, 'r')
	lines = file.readlines()
	lines = [l.strip() for l in lines]
	block_starts = np.array([i for i,l in enumerate(lines) if l==delimiter])
	return (block_starts, lines)

@jit(nopython=True)
def parse_states(block_starts, lines, n_qubits, n_states=None, real_only=False):
	block_dim = pow(2,n_qubits)+1
	shuffle(block_starts)
	states = []
	append_state = states.append
	n_found = 0
	for s in block_starts:
		state = np.array([float(val[0])+1j*float(val[1]) for val in
							[line.strip('()').split(', ') for line in 
							                              lines[s:s+block_dim]]
						])
		append_state(state)
		n_found +=1
		if n_states is not None and n_found >= n_states:
			break
	if n_states is not None and n_found < n_states:
		raise ValueError("Unable to load sufficient stabilizer states.")
	return states


@jit
def parse_groups(block_starts, lines, n_qubits, n_groups=None, real_only=False):
	block_dim = pow(2,n_qubits)+1
	shuffle(block_starts)
	groups = []
	append_group = groups.append
	n_found = 0
	for s in block_starts:
		group = StabilizerMatrix([PauliBytes(l) for l in lines[s:s+block_dim]])
		append_group(group)
		n_found += 1
		if n_groups is not None and n_found>=n_groups:
			breal
	if n_groups is not None and n_found < n_groups:
		raise ValueError("Unable to load sufficient stabilizer generators.")
	return groups

def load_states(n_qubits, n_states=None, real_only=False):
	file_path = path_join(DIR,'data','{}.states'.format(n_qubits))
	if not path_exists(file_path):
		return None #Generate...? Return None is probably the simplest.
	block_starts, lines = get_blocks_and_lines(file_path, 'STATE')
	states = parse_states(block_starts, lines, n_qubits, n_states, real_only)
	return states

def load_groups(n_qubits, n_groups=None, real_only=False):
	file_path = path_join(DIR,'data','{}.generators'.format(n_qubits))
	if not path_exists(file_path):
		return None
	block_starts, lines = get_blocks_and_lines(file_path, 'GROUP')
	groups = parse_groups(block_starts, lines, n_qubits, n_groups, real_only)
	return groups

def get_stabilizer_states(n_qubits, n_states=None,
	                      real_only=False, use_cached=True):
	if n_states is None:
		n_states = n_stabilizer_states(n_qubits)
	states = None
	if use_cached:
		states = load_states(n_qubits, n_states, real_only)
	if states is None:
		n_groups = ceil(n_states/pow(2,n_qubits))
		groups = None
		if use_cached:
			groups = load_groups(n_qubits, n_groups, real_only)
		if groups is None:
			generate_groups(n_qubits, n_groups, real_only)
		states = get_states_from_groups(groups, n_qubits)
	return states
