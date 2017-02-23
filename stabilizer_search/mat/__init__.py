"""Module to hold useful matrices"""

from cmath import exp as cexp
from math import pi, sqrt

from .haar_random import get_so2, get_su2
from .py_haar_random import pyget_su2


import numpy as np


__all__=['qeye', 'X', 'Y', 'Z', 'S', 'H', 'T']


def qeye(n):
    return np.eye(n, dtype=np.complex_)


X = np.matrix([[0,1], [1,0]], dtype=np.complex_)


Y = np.matrix([[0,-1j], [1j,0]], dtype=np.complex_)


Z = np.matrix([[1,0], [0,-1]], dtype=np.complex_)


H = (1/sqrt(2.))*np.matrix([[1, 1],[1, -1]], dtype=np.complex_)


S = np.matrix([[1, 0], [0, 1j]], dtype=np.complex_)


T = np.matrix([[1, 0], [0, cexp(1j*pi/4)]], dtype=np.complex_)

def tensor(*args):
    matrices = list(args)
    out = matrices.pop(0)
    while matrices:
        out = np.kron(out, matrices.pop(0))
    return out