from stabilizer_cpp cimport *

from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

DTYPE = np.int8
ctypedef np.int8_t DTYPE_t

cdef list SM_to_array(const StabilizerMatrix& m):
	cdef np.ndarray[DTYPE_t, ndim=1] out
	cdef unsigned int gen_counter, bit_counter, offset, nqubits
	cdef list out_group = []
	nqubits = m.NQubits()
	for gen_counter in range(nqubits):
		out = np.zeros(2*nqubits, dtype=np.bool_)
		num = m.Generators()[gen_counter].toUlong()
		bits = bin(num)[2:]
		offset = 2*nqubits-len(bits)
		for bit_counter in range(offset, 2*nqubits):
			if bits[bit_counter]=='1':
				out[bit_counter] = 1
		out_group.append(out.view(dtype=np.bool))
	return out_group


cdef list SMatrices_to_numpy(vector[StabilizerMatrix] generators):
	cdef list np_groups = []
	cdef unsigned int counter = 0
	for matrix in generators:
		np_groups.append(SM_to_array(matrix))
	return np_groups


def c_get_positive_groups(unsigned int nqubits, unsigned int nstates):
	cdef vector[StabilizerMatrix] positive_groups = getStabilizerGroups(nqubits, 
																     nstates)
	cdef list out_groups =  SMatrices_to_numpy(positive_groups)
	return out_groups