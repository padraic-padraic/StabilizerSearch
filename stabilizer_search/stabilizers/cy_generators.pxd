boolish = np.int8
ctypedef np.int8_t bool_t

cdef class StabilizerMatrix:
    cdef public unsigned int nQubits
    cdef public np.ndarray[complex_t, ndim=2] sMatrix
    
    cdef bint isXY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit]==1):
            return True
        return False

    cdef bint isZY(self, unsigned int row, unsigned int qubit):
        if (self.sMatrix[row, qubit+self.nQubits]==1):
            return True
        return False

    cdef bint linearly_independent(self):
        cdef unsigned int i
        for i in range(self.nQubits):
            if np.all(self.sMatrix[i]==0):
                return False
        return True

    cdef void __to_canonical_form(self):
        cdef unsigned int i = 0, j, k, m
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isXY(k,j):
                    self.sMatrix[[i,k]] = self.sMatrix[[k,i]]
                    for m in range(self.nQubits):
                        if m != i and self.isZY(m, j):
                            self.sMatrix[m] += self.sMatrix[i]
                            self.sMatrix[m] = np.mod(self.sMatrix[m], 2)
                    i+=1
        for j in range(self.nQubits):
            for k in range(i, self.nQubits):
                if self.isZY(k, j):
                    self.sMatrix[[i,k]] = self.sMatrix[[k,i]]
                    for m in range(self.nQubits):
                        if m!= i and self.isZY(m, j):
                            self.sMatrix[m]+= self.sMatrix[i]
                            self.sMatrix[m] = np.mod(self.sMatrix[m], 2)
                    i++
