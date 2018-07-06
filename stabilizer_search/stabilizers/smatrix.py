from itertools import combinations
from numba import vectorize, uint8, uint64, guvectorize, jit
from random import seed, shuffle

import numpy as np

from ..core import n_stabilizer_states, tensor, Id, X, Y, Z, qeye, SEEDED_RANDOM

if not SEEDED_RANDOM:
    seed()
    SEEDED_RANDOM = True

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

__all__ = ['PauliBytes', 'StabilizerMatrix', 'get_states_from_groups', 'generate_groups']

P_LITERALS = ['I', 'Z', 'X', 'Y']

P_MATRICES = [Id, Z, X, Y]

ONE = np.uint8(1)

@vectorize(["uint8(uint8, uint8)"])
def get_literal_index(x_bit,z_bit):
    return (2 * x_bit) + z_bit

@guvectorize([(uint8[:], uint8[:], uint64, uint8[:])], '(n),(n),()->(n)')
def gu_get_literals(x_bytes, z_bytes, n_qubits, res):
    for i in range(n_qubits):
        index = i//8
        shift = 7-(i%8)
        res[i] = 2*((x_bytes[index] >> shift) & 1) + ((z_bytes[index]>>shift) & ONE)

@vectorize(["uint8(uint8)"])
def get_weight(char_byte):
    return CHAR_WEIGHTS[char_byte]

@vectorize(["uint8(uint8)"])
def get_parity(char_byte):
    return CHAR_PARITIES[char_byte]

@guvectorize([(uint8[:], uint8[:])], '(n)->()')
def parity_reduction(bytes_array, res):
    res = np.uint8(0)
    for b in bytes_array:
        res ^= CHAR_PARITIES[b]

@guvectorize([(uint8[:], uint64[:])], '(n)->()')
def weight_reduction(bytes_array, res):
    for b in bytes_array:
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
        print(isinstance(bin_string, list))
        if isinstance(bin_string, str):
            if len(bin_string) != 2*n_qubits:
                bin_string = bin_string.zfill(2*n_qubits)
            self.xbytes = np.packbits(np.array([True if b == '1' else False
                                      for b in bin_string[:n_qubits] ]))
            self.zbytes = np.packbits(np.array([True if b == '1' else False
                                      for b in bin_string[n_qubits:] ]))
        elif isinstance(bin_string, (list, np.ndarray)):
            gap = 2 * n_qubits - len(bin_string)
            if gap > 0:
                bin_string = [0] * (gap) + bin_string
            self.xbytes = np.packbits(np.array([True if b == 1 else False
                                     for b in bin_string[:n_qubits] ]))
            self.zbytes = np.packbits(np.array([True if b == 1 else False
                                     for b in bin_string[n_qubits:] ]))
        else:
            raise TypeError("PauliBytes needs a binary string, or else a binary list.")

    @classmethod
    def from_pstring(cls, pauli_string, phase=0):
      x_bits = []
      z_bits = []
      n_qubits = len(pauli_string)
      x_bits = [1 if p == 'X' or p == 'Y' else 0 for p in pauli_string]
      z_bits = [1 if p == 'Z' or p == 'Y' else 0 for p in pauli_string]
      pauli = cls(n_qubits, x_bits+z_bits, phase)
      return pauli

    @classmethod
    def from_pauli(cls, pauli):
      p = cls(pauli.n_qubits, '0')
      p.xbytes = np.array(pauli.xbytes)
      p.zbytes = np.array(pauli.zbytes)
      return p

    @classmethod
    def random(cls, n_qubits, phase=0):
        return cls(np.around(np.random.random(2*n_qubits)), phase)

    @property
    def bit_string(self):
      bstring = ''.join([str(i) for i in 
                         np.unpackbits(self.xbytes)[:self.n_qubits]])
      bstring += ''.join([str(i) for i in
                         np.unpackbits(self.zbytes)[:self.n_qubits]])
      return bstring

    def __str__(self):
        x_string = np.unpackbits(self.xbytes)[:self.n_qubits]
        z_string = np.unpackbits(self.zbytes)[:self.n_qubits]
        indices = get_literal_index(x_string, z_string)
        p_string = ''.join([P_LITERALS[indices[i]] for i in range(self.n_qubits)])
        return p_string

    def __eq__(self, other):
        if self.n_qubits != other.n_qubits:
            return False
        if not np.array_equal(self.xbytes, other.xbytes):
            return False
        if not np.array_equal(self.zbytes, other.zbytes):
            return False
        return True

    def __neq__(self, other):
        return (not self.__eq__(other))

    @property
    def weight(self):
        res = np.zeros(1, dtype=np.uint64)
        weight_reduction(np.bitwise_or(self.xbytes, self.zbytes), res)
        return res[0]

    @property
    def mat(self):
        if not self.__ready_mat:
            indices = np.zeros([1, self.n_qubits], dtype=np.uint64)
            gu_get_literals(self.xbytes, self.zbytes, self.n_qubits, indices)
            self.__mat = tensor([P_MATRICES[i] for i in indices])
            self.__ready_mat = True
        return self.__mat

    def __mul__(self, other):
        self.__ready_mat = False
        out = PauliBytes(self.n_qubits, ''.zfill(2*self.n_qubits), self.phase)
        out.xbytes = np.bitwise_xor(self.xbytes, other.xbytes)
        out.zbytes = np.bitwise_xor(self.zbytes, other.zbytes)
        out.phase = np.sum(get_parity(np.bitwise_and(self.zbytes, other.xbytes)))
        out.phase = out.phase%2

    def __imul__(self, other):
        self.__ready_mat = False
        self.phase += np.sum(get_parity(
                   np.bitwise_and(self.zbytes, other.xbytes)))
        self.phase = self.phase % 2
        self.xbytes = np.bitwise_xor(self.xbytes, other.xbytes)
        self.zbytes = np.bitwise_xor(self.zbytes, other.zbytes)

    def conjugation(self, other):
        self = other * self
        y_count = np.zeros(1, dtype=np.uint8)
        parity_reduction(np.bitwise_and(self.xbytes, self.zbytes), y_count)
        if y_count == 1:
            self.phase ^= 1

    @jit
    def is_XY(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        return ((self.xbytes[char_ind] >> char_shift) & 1) == 1 

    @jit
    def is_ZY(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        return ((self.zbytes[char_ind] >> char_shift) & 1) == 1

    @jit
    def is_Z(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        z_bit = (self.zbytes[char_ind] >> char_shift) & 1
        x_bit = (self.xbytes[char_ind] >> char_shift) & 1
        return (z_bit == 1) and (x_bit == 0)

    @jit
    def is_real(self):
        res = np.zeros(1, dtype=np.uint8)
        parity_reduction(np.bitwise_and(self.xbytes, self.zbytes), res)
        return res[0] == 0


class StabilizerMatrix(object):

    def __init__(self, paulis, phase=0):
        self.n_qubits = paulis[0].n_qubits
        self.phase = phase
        self.paulis = paulis

    def to_canonical_form(self):
        i = 0
        for j in range(self.n_qubits):
            for k in range(i, self.n_qubits):
                if self.paulis[k].is_XY(j):
                    p_copy = self.paulis[i]
                    self.paulis[i] = self.paulis[k]
                    self.paulis[k] = p_copy
                    for m in range(self.n_qubits):
                        if m == i:
                            continue
                        if self.paulis[m].is_XY(j):
                            self.paulis[m] = (self.paulis[i]*self.paulis[m])
                    i += 1
                    break
        for j in range(self.n_qubits):
            for k in range(i, self.n_qubits):
                if self.paulis[k].is_Z(j):
                    p_copy = self.paulis[k]
                    self.paulis[k] = self.paulis[i]
                    self.paulis[i] = self.paulis[k]
                    for m in range(self.n_qubits):
                        if m == i:
                            continue
                        if self.paulis[m].is_ZY(j):
                            self.paulis[m] = (self.paulis[i]*self.paulis[m])
                    i += 1
                    break

    def __eq__(self, other):
        if other.n_qubits != self.n_qubits:
            return False
        if len(self.paulis) != len(other.paulis):
            return False
        for i in range(len(self.paulis)):
            if self.paulis[i] != other.paulis[i]:
                return False
        return True

    def __neq__(self, other):
        return (not self.__eq__(other))

    def is_real(self):
        if any(not(p.is_real()) for p in self.paulis):
            return False
        else:
            return True

    @classmethod
    def random(cls, n_qubits):
        return cls([PauliBytes.random(n_qubits) for i in range(n_qubits)])

    def set_phase(self, phase):
        phase_string = None
        if isinstance(phase, int):
            if phase >= pow(2, len(self.paulis)):
                raise IndexError("Phase num is larger than the group.")
            phase_string = bin(phase)[2:].zfill(len(self.paulis))
        elif isinstance(phase, str):
            if len(phase) > len(self.paulis):
                raise IndexError("Phase string is larger than the group.")
            phase_string = phase.zfill(len(self.paulis))
        if phase_string:
            for i, bit in enumerate(phase_string):
                if bit == '1':
                    self.paulis.phase = 1
                else:
                    self.paulis.phase = 0
        elif isinstance(phase, np.ndarray):
            if phase.size() > len(self.paulis):
                raise IndexError("Phase string is larger than the group.")
            for i, bit in enumerate(phase):
                if bit == '1':
                    self.paulis[i].phase = 1
                else:
                    self.paulis[i].phase = 0
        else:
            raise TypeError("set_phase expects an integer, binary string, or numpy array")

    def get_projector(self):
        eye = qeye(pow(2, self.n_qubits))
        proj = 0.5 * (eye + self.paulis[0])
        for i in range(1, len(self.paulis)):
            proj = proj * (0.5 * (eye + self.paulis[i]))
        return proj

    def get_stabilizer_state(self):
        dim = pow(2, self.n_qubits)
        projector = self.get_projector()
        vals, vecs = np.linalg.eigh(projector)
        vals = np.abs(vals)
        for i in range(vals):
            if np.allclose(vals[i],1.):
                vec = np.matrix(vecs[:,i],dtype=np.complex_).reshape(dim, 1)
                vec = vec / (np.linalg.norm(vec, 2))
                return vec
        return None

@jit
def get_states_from_groups(groups, n_states):
    n_qubits = groups[0].n_qubits
    phases = pow(2, n_qubits)
    states = []
    append_state = states.append
    n_found = 0
    for p in range(phases):
        p_string = bin(p)[2:].zfill(n_qubits)
        for g in groups:
            g.set_phase(p_string)
            append_state(g.get_stabilizer_state())
            n_found += 1
            if n_found >= n_states:
                break
    if n_found < n_states:
        raise ValueError("Unable to generate sufficient stabilizer states.")
    return states

@jit
def generate_groups(n_qubits, n_groups=None, real_only=False):
    dim = pow(2, n_qubits)
    paulis = [PauliBytes(n_qubits, [True if b == '1' else False 
                                       for b in bin(x)[2:].zfill(2 * n_qubits)])
                            for x in range(1, dim)]
    shuffle(paulis)
    groups = []
    group_count = 0
    if n_groups is None:
        n_groups = n_stabilizer_states(n_qubits) / dim
    append_group = groups.append
    for pauli_set in combinations(paulis, n_qubits):
        candidate = StabilizerMatrix([p for p in pauli_set])
        candidate.to_canonical_form()
        candidate.set_phase(0)
        if real_only:
            if not candidate.is_real():
                continue
        if candidate not in groups:
            append_group(candidate)
            group_count += 1
        if group_count >= n_groups:
            break
    if group_count < n_groups:
        raise ValueError("Couldn't find enough stabilizer groups.")
    return groups
