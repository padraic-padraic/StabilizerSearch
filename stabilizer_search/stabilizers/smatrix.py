from ..core.unitaries import Id, X, Y, Z, tensor, qeye

from numba import vectorize, uint8, uint64, guvectorize, jit

import numpy as np

CHAR_WEIGHTS = bytearray([
              0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2,
              2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3,
              2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4,
              4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
              2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
              4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
              4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6,
              6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
              2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3,
              3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
              4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
              4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
              4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5,
              5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7,
              6, 7, 7, 8])

CHAR_PARITIES = bytearray([
               0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
               0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
               0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
               0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
               0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
               0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
               0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0,
               0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1,
               0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
               1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
               0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
               0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
               0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
               1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
               0, 1, 1, 0])

__all__ = ['PauliBytes','StabilizerMatrix']

P_LITERALS = ['I','Z','X','Y']

P_MATRICES = [Id, Z, X, Y]

@vectorize(["uint8(uint8,uint8)"])
def get_literal_index(x_bit,z_bit):
    return (2 * x_bit) + z_bit

@guvectorize([(uint8[:],uint8[:],uint64,uint8[:])],'(n),(n),()->(n)')
def gu_get_literals(x_bytes, z_bytes, n_qubits, res):
  for i in range(n_qubits):
    index = i//8
    shift = 7-(i%8)
    res[i] = 2*((self.x_bytes[index] >> shift) & 1) + ((self.z_bytes[index]>>shift) & one)

@vectorize(["uint8(uint8)"])
def get_weight(char_byte):
    return CHAR_WEIGHTS[char_byte]

@vectorize(["uint8(uint8)"])
def get_parity(char_byte):
    return CHAR_PARITIES[char_byte]

@guvectorize([(uint8[:],uint8[:])],'(n)->()')
def parity_reduction(bytes, res):
    res = np.uint8(0)
    for b in bytes:
        res ^= CHAR_PARITIES[b]

@guvectorize([(uint8[:],uint64[:])],'(n)->()')
def weight_reduction(bytes, res):
    for b in bytes:
        res += CHAR_WEIGHTS[b]

def find_first_set(barr):
    for j in range(barr.size):
        if barr[j]:
            return bin(barr[j])[2:][::-1].find('1')
      else:
            continue
    return None

class PauliBytes(object):

    def __init__(self, n_qubits, bin_string, phase=0):
        self.n_qubits = n_qubits
        self.phase = phase
        self.__matrix = None
        self.__ready_mat = False
        self.__mat = None
        print(type(bin_string))
        print(isinstance(bin_string,list))
        if isinstance(bin_string,str):
            if len(bin_string) != 2*n_qubits:
                bin_string = bin_string.zfill(2*n_qubits)
            self.xbytes = np.packbits(np.array([True if b=='1' else False
                                      for b in bin_string[:n_qubits] ]))
            self.zbytes = np.packbits(np.array([True if b=='1' else False
                                      for b in bin_string[n_qubits:] ]))
        elif isinstance(bin_string,list):
            gap = 2*n_qubits - len(bin_string)
            if gap>0:
                bin_string = [0]*(gap) + bin_string
            self.xbytes = np.packbits(np.array([True if b==1 else False
                                     for b in bin_string[:n_qubits] ]))
            self.zbytes = np.packbits(np.array([True if b==1 else False
                                     for b in bin_string[n_qubits:] ]))
        else:
            raise TypeError("PauliBytes needs a binary string, or else a binary list.")

    @classmethod
    def from_pstring(cls, pauli_string, phase=0):
      x_bits = []
      z_bits = []
      n_qubits = len(pauli_string)
      x_bits = [1 if p=='X' or p=='Y' else 0 for p in pauli_string]
      z_bits = [1 if p=='Z' or p=='Y' else 0 for p in pauli_string]
      pauli = cls(n_qubits, x_bits+z_bits, phase)
      return pauli

    @classmethod
    def from_pauli(cls, pauli):
      cls = PauliBytes(pauli.n_qubits, '0')
      cls.bytes = np.array(pauli.xbytes)

    @property
    def bit_string(self):
      bstring = ''.join([str(i) for i in 
                         np.unpackbits(self.xbytes)[:self.n_qubits]])
      bstring += ''.join([str(i) for i in
                         np.unpackbits(self.zbytes)[:self.n_qubits]])
      return self._bit_string

    def __str__(self):
        x_string = np.unpackbits(self.xbytes)[:self.n_qubits]
        z_string = np.unpackbits(self.zbytes)[:self.n_qubits]
        indices = get_literal_index(self.x_string, self.z_string)
        p_string = ''.join([P_LITERALS[indices[i]] for i in range(self.n_qubits)])
        return p_string

    @property
    def weight(self):
        res = np.zeros(1,dtype=np.uint64)
        weight_reduction(np.bitwise_or(self.xbytes, self.zbytes),res)
        return res[0]

    @property
    def mat(self):
      if not self.__ready_mat:
        indices = np.zeros([1,self.n_qubits],dtype=np.uint64)
        gu_get_literals(self.xbytes, self.zbytes, self.n_qubits, indices)
        self.__mat = tensor([P_MATRICES[i] for i in indices])
        self.__ready_mat = True
      return self.__mat

    def __mul__(self, other):
        self.__ready_mat = False
        out = PauliBytes(self.n_qubits, ''.zfill(2*self.n_qubits), self.phase)
        out.xbytes = np.bitwise_xor(self.xbytes, other.xbytes)
        out.zbytes = np.bitwise_xor(self.zbytes, other.zbytes)
        out.phase = np.sum(get_parity(np.bitwise_and(self.zbytes,other.xbytes)))
        out.phase = out.phase%2

    def __imul__(self, other):
        self.__ready_mat = False
        self.phase += np.sum(get_parity(
                   np.bitwise_and(self.zbytes,other.xbytes)))
        self.phase = self.phase % 2
        self.xbytes = np.bitwise_xor(self.xbytes,other.xbytes)
        self.zbytes = np.bitwise_xor(self.zbytes,other.zbytes)

    def conjugation(self, other):
        self = other * self;

    @jit
    def is_XY(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        return ((self.xbytes[char_ind] >> char_shift) & 1)==1 

    @jit
    def is_ZY(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        return ((self.zbytes[char_ind] >> char_shift) & 1)==1

    @jit
    def is_Z(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        z_bit = (self.zbytes[char_ind] >> char_shift) & 1
        x_bit = (self.xbytes[char_ind] >> char_shift) & 1
        return (z_bit == 1) and (x_bit == 0)

    @jit
    def is_real(self):
        res = np.zeros(1,dtype=np.uint8)
        parity_reduction(np.bitwise_and(self.xbytes,self.zbytes), res)
        return res[0] == 0


class StabilizerMatrix(object):

    def __init__(self, paulis, phase=0):
        self.n_qubits = paulis[0].n_qubits
        self.phase = phase
        self.paulis = paulis

    def to_canonical_form(self):
        i = 0
        for j in range(self.n_qubits):
            for k in range(i,self.n_qubits):
                if self.paulis[k].is_XY(j):
                    p_copy = self.paulis[i]
                    self.paulis[i] = self.paulis[k]
                    self.paulis[k] = p_copy
                    for m in range(self.n_qubits):
                        if m==i:
                            continue
                        if(self.paulis[m].is_XY(j)):
                            self.paulis[m] = (self.paulis[i]*self.paulis[m])
                    i+=1
                    break
        for j in range(self.n_qubits):
            for k in range(i, self.n_qubits):
                if self.paulis[k].is_Z(j):
                    p_copy = self.paulis[k]
                    self.paulis[k] = self.paulis[i]
                    self.paulis[i] = self.paulis[k]
                    for m in range(self.n_qubits):
                        if m==i:
                            continue
                        if(self.paulis[m].is_ZY(j)):
                            self.paulis[m] = (self.paulis[i]*self.paulis[m])
                    i+=1
                    break
        return

    def get_projector(self):
        eye = qeye(pow(2,self.n_qubits))
        proj = 0.5 * (eye + self.paulis[0])
        for i in range(1,len(self.paulis)):
          proj = proj * (0.5 * (eye + self.paulis[i]))
        return proj

    def get_stabilizer_state(self):
        projector = self.get_projector()
        vals, vecs = np.linalg.eigh(projector)
        vals = np.abs(vals)
        for i in range(vals):
          if np.allclose(vals,1.):
            return np.matrix(vecs[:,i],dtype=np.complex_).reshape(dim, 1)
        return None
