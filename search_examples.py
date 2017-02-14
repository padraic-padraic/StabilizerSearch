from math import sqrt
from stabilizer_search import BruteForceSearch, RandomWalkSearch
from stabilizer_search.mat import X, T
from unittest.mock import patch

import numpy as np

plus = np.array([[1/sqrt(2)], [1/sqrt(2)]], dtype=np.complex_)
H = T*plus
target_state = np.kron(H,H)
target_string = 'Two-fold T State'
n_qubits = 2

BF = BruteForceSearch(target_state, target_string, n_qubits)
RW = RandomWalkSearch(target_state, target_string, n_qubits, 2)

# res = BF()
# print(res)
# print('\n --- \n')
res = RW()
print(res)