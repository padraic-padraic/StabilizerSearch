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
RW = RandomWalkSearch(target_state, target_string, n_qubits, 2,
                      use_cached=False)
RW_cseed = RandomWalkSearch(target_state, target_state, n_qubits, 2,
                            use_cached=False, seed='computational')
RW_pseed = RandomWalkSearch(target_state, target_state, n_qubits, 2,
                            use_cached=False, seed='product')


# res = BF()
# print(res)
# print('\n --- \n')
def test():
    #res = BF()
    res = RW()
    # print(res)

def test_c():
    res = RW_cseed()

def test_p():
    res = RW_pseed()

if __name__ == '__main__':
    t1 = timeit.repeat("test()", "from __main__ import test", number=5, repeat=3)
    t2 = timeit.repeat("test_c()", "from __main__ import test_c", number=5, repeat=3)
    t3 = timeit.repeat("test_p()", "from __main__ import test_p", number=5, repeat=3)
    print(t1)
    print(t2)
    print(t3)
