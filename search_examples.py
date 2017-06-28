from math import sqrt
from stabilizer_search import BruteForceSearch, RandomWalkSearch
from stabilizer_search.search.random_walk import do_random_walk
from stabilizer_search.mat import X, T
from stabilizer_search.mat.haar_random import get_su2
from stabilizer_search.mat import tensor

import numpy as np
import timeit

plus = np.array([[1/sqrt(2)], [1/sqrt(2)]], dtype=np.complex_)
H = T*plus
target_state = tensor(H,H)
target_string = 'Two-fold T State'
n_qubits = 2

BF = BruteForceSearch(target_state, target_string, n_qubits)
RW = RandomWalkSearch(target_state, target_string, n_qubits, 2, use_cached=False)

# res = BF()
# print(res)
# print('\n --- \n')
def test():
    #res = BF()
    res = RW()
    # print(res)


if __name__ == '__main__':
    test()
    # print(timeit.timeit("test()", "from __main__ import test", number=20))
