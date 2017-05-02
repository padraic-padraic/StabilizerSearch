from libcpp.vector cimport vector 

cdef extern from "lib/utils.h":
	unsigned int nStabilizers(unsigned int nQubits)


cdef extern from "lib/SymplecticPauli.h":
	cdef cppclass SymplecticPauli:
		unsigned long toUlong()


cdef extern from "lib/StabilizerMatrix.h":
	cdef cppclass StabilizerMatrix:
		const unsigned int NQubits()
		vector[SymplecticPauli] Generators()

cdef extern from "lib/generation.h":
	# VectorList getStabilizerStates(unsigned int nQubits, unsigned int nStates)
	vector[StabilizerMatrix] getStabilizerGroups(unsigned int nQubits, unsigned int nStates)
	