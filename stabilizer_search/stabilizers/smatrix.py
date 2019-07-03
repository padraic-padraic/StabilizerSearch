from ..core import n_stabilizer_states
from ..core.linalg import int_to_bits, tensor
from ..core.unitaries import Id, X, Y, Z, qeye

from itertools import combinations
from random import shuffle

from numba import guvectorize, jit, jitclass, njit, vectorize
from numba import boolean, complex128, uint8, uint64

import numpy as np

CHAR_WEIGHTS = np.array([
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
              6, 7, 7, 8], dtype=np.uint8)

CHAR_PARITIES = np.array([
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
               0, 1, 1, 0], dtype=np.uint8)

__all__ = ['PauliBytes', 'StabilizerMatrix']

P_LITERALS = ['I', 'Z', 'X', 'Y']

P_MATRICES = [Id, Z, X, Y]

ZERO = np.uint8(0)
ONE = np.uint8(1)


@vectorize(["uint8(uint8,uint8)"])
def get_literal_index(x_bit, z_bit):
    return (2 * x_bit) + z_bit


@guvectorize([(uint8[:], uint8[:], uint64, uint8[:])], '(n),(n),()->(n)')
def gu_get_literals(x_bytes, z_bytes, n_qubits, res):
    for i in range(n_qubits):
        index = i//8
        shift = 7 - (i % 8)
        res[i] = 2 * ((x_bytes[index] >> shift) & ONE)
        res[i] = res[i] + ((z_bytes[index] >> shift) & ONE)


@vectorize(["uint8(uint8)"])
def get_weight(char_byte):
    return CHAR_WEIGHTS[char_byte]


@vectorize(["uint8(uint8)"])
def get_parity(char_byte):
    return CHAR_PARITIES[char_byte]


def find_first_set(barr):
    for j in range(barr.size):
        if barr[j]:
            return bin(barr[j])[2:][::-1].find('1')
        else:
            continue
    return None


pb_spec = [
    ('n_qubits', uint64),
    ('phase', uint8),
    ('__ready_mat', boolean),
    ('__mat', complex128[:])
]


@jitclass(pb_spec)
class PauliBytes(object):
    def __init__(self, n_qubits, bin_string, phase=0):
        self.n_qubits = n_qubits
        self.phase = phase
        self.__ready_mat = False
        self.__mat = None
        print(type(bin_string))
        print(isinstance(bin_string, list))
        if isinstance(bin_string, str):
            if len(bin_string) != 2*n_qubits:
                bin_string = bin_string.zfill(2*n_qubits)
            self.xbytes = np.packbits(np.array([True if b == '1' else False
                                      for b in bin_string[:n_qubits]]))
            self.zbytes = np.packbits(np.array([
                b == '1' for b in bin_string[n_qubits:]]))
        elif type(bin_string) in [list, np.ndarray]:
            gap = 2*n_qubits - len(bin_string)
            if gap > 0:
                bin_string = [0]*(gap) + bin_string
            self.xbytes = np.packbits(np.array([
                b == 1 for b in bin_string[:n_qubits]]))
            self.zbytes = np.packbits(np.array([
                b == 1 for b in bin_string[n_qubits:]]))
        else:
            raise TypeError("PauliBytes needs a binary string, "
                            "or else a binary list.")

    @property
    def bit_string(self):
        bstring = ''.join([
            str(i) for i in np.unpackbits(self.xbytes)[:self.n_qubits]])
        bstring += ''.join([
            str(i) for i in np.unpackbits(self.zbytes)[:self.n_qubits]])
        return bstring

    def to_string(self):
        indices = np.zeros([1, self.n_qubits], dtype=np.uint64)
        gu_get_literals(self.x_bytes, self.z_bytes, self.n_qubits, indices)
        p_string = ''.join([
            P_LITERALS[indices[i]] for i in range(self.n_qubits)])
        return p_string

    @property
    def weight(self):
        res = np.sum(get_weight(np.bitwise_or(self.xbytes, self.zbytes)))
        return res

    @property
    def mat(self):
        if not self.__ready_mat:
            indices = np.zeros([1, self.n_qubits], dtype=np.uint64)
            gu_get_literals(self.xbytes, self.zbytes, self.n_qubits, indices)
            self.__mat = tensor([P_MATRICES[i] for i in indices])
            self.__ready_mat = True
        return self.__mat

    def left_mul(self, other):
        self.__ready_mat = False
        out = PauliBytes(self.n_qubits, ''.zfill(2*self.n_qubits), self.phase)
        out.xbytes = np.bitwise_xor(self.xbytes, other.xbytes)
        out.zbytes = np.bitwise_xor(self.zbytes, other.zbytes)
        out.phase = np.sum(
            get_parity(np.bitwise_and(self.zbytes, other.xbytes)))
        out.phase = out.phase % 2

    def left_inplace_mul(self, other):
        self.__ready_mat = False
        self.phase += np.sum(
            get_parity(np.bitwise_and(self.zbytes, other.xbytes)))
        self.phase = self.phase % 2
        self.xbytes = np.bitwise_xor(self.xbytes, other.xbytes)
        self.zbytes = np.bitwise_xor(self.zbytes, other.zbytes)

    def is_XY(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        return ((self.xbytes[char_ind] >> char_shift) & 1) == 1

    def is_ZY(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        return ((self.zbytes[char_ind] >> char_shift) & 1) == 1

    def is_Z(self, qubit):
        char_ind = qubit // 8
        char_shift = qubit % 8
        z_bit = (self.zbytes[char_ind] >> char_shift) & 1
        x_bit = (self.xbytes[char_ind] >> char_shift) & 1
        return (z_bit == 1) and (x_bit == 0)

    def is_real(self):
        res = np.zeros(1, dtype=np.uint8)
        parity_reduction(np.bitwise_and(self.xbytes, self.zbytes), res)
        return res[0] == 0

    def is_zero(self):
        return np.any(np.bitwise_or(self.xbytes, self.zbytes) == 0)


def pauli_from_string(pauli_string, phase=0):
    x_bits = []
    z_bits = []
    n_qubits = len(pauli_string)
    x_bits = [1 if p == 'X' or p == 'Y' else 0 for p in pauli_string]
    z_bits = [1 if p == 'Z' or p == 'Y' else 0 for p in pauli_string]
    pauli = PauliBytes(n_qubits, x_bits+z_bits, phase)
    return pauli


def pauli_from_pauli(pauli):
    p = PauliBytes(pauli.n_qubits, '0')
    p.xbytes = np.array(pauli.xbytes)
    p.zbytes = np.array(pauli.zbytes)
    return p


def random_pauli_matrix(n_qubits, real_only=False, phase=0):
    return PauliBytes(np.around(np.random.random(2*n_qubits)), phase)


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
                        if(self.paulis[m].is_XY(j)):
                            self.paulis[m] = (
                                self.paulis[i].let_mul(self.paulis[m])
                            )
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
                        if(self.paulis[m].is_ZY(j)):
                            self.paulis[m] = (
                                self.paulis[i].left_mul(self.paulis[m])
                            )
                    i += 1
                    break

    def is_real(self):
        return all(p.is_real() for p in self.paulis)

    def is_full_rank(self):
        return any([p.is_zero() for p in self.paulis])

    def set_phase(self, phase):
        phase_string = None
        if isinstance(phase, int):
            if phase >= pow(2, self.n_qubits):
                raise IndexError("Phase num is larger than the group.")
            phase_string = bin(phase)[2:].zfill(self.n_qubits)
        elif isinstance(phase, str):
            if len(phase) > self.n_qubits:
                raise IndexError("Phase string is larger than the group.")
            phase_string = phase.zfill(self.n_qubits)
        if phase_string is not None:
            for i, bit in enumerate(phase_string):
                if bit == '1':
                    self.paulis[i].phase = 1
                else:
                    self.paulis[i].phase = 0
        elif isinstance(phase, np.ndarray):
            if np.any(np.logical_and(phase != 1, phase != 0)):
                raise ValueError("Invalid phase array, entries should be 0 or "
                                 "1.")
            if phase.size() > self.n_qubits:
                raise IndexError("Phase string is larger than the group.")
            for i, bit in enumerate(phase):
                self.paulis[i] = 1
        else:
            raise TypeError("set_phase expects an integer, binary string"
                            ", or numpy array")

    def get_projector(self):
        eye = qeye(pow(2, self.n_qubits))
        proj = 0.5 * (eye + self.paulis[0])
        for i in range(1, len(self.paulis)):
            proj = proj * (0.5 * (eye + self.paulis[i]))
        return proj

    def get_stabilizer_state(self):
        projector = self.get_projector()
        vals, vecs = np.linalg.eigh(projector)
        vals = np.abs(vals)
        dim = pow(2, self.n_qubits)
        for i in range(vals):
            if np.allclose(vals, 1.):
                vec = np.matrix(vecs[:, i], dtype=np.complex_).reshape(dim, 1)
                vec = vec / (np.linalg.norm(vec, 2))
                return vec
        return None


@njit
def random_smatrix(n_qubits):
    while True:
        matrix = StabilizerMatrix(
            [PauliBytes.random(n_qubits) for i in range(n_qubits)]
        )
        matrix.to_canonical_form()


@njit
def get_states_from_groups(groups, n_states):
    n_qubits = groups[0].n_qubits
    phases = pow(2, n_qubits)
    states = []
    n_found = 0
    for p in range(phases):
        p_string = bin(p)[2:].zfill(n_qubits)
        for g in groups:
            g.set_phase(p_string)
            states.append(g.get_stabilizer_state())
            n_found += 1
            if n_found >= n_states:
                break
    if n_found < n_states:
        raise ValueError("Unable to generate sufficient stabilizer states.")
    return states


@njit
def generate_groups(n_qubits, n_groups=None):
    dim = pow(2, n_qubits)
    b_strings = [
        int_to_bits(x) for x in range(1, dim)
    ]
    shuffle(b_strings)
    groups = []
    group_count = 0
    if n_groups is None:
        n_groups = n_stabilizer_states(n_qubits)/dim
    append_group = groups.append
    for strings in combinations(b_strings, n_qubits):
        candidate = StabilizerMatrix([PauliBytes(s) for s in strings])
        candidate.to_canonical_form()
        candidate.set_phase(0)
        if candidate not in groups:
            append_group(candidate)
            group_count += 1
        if group_count == n_groups:
            break
    return groups
