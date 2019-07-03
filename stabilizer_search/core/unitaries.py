"""Module to hold useful matrices"""

from cmath import exp as cexp
from math import asin, cos, pi, sin, sqrt
from random import random

import numpy as np


def qeye(n):
    return np.eye(n, dtype=np.complex_)


Id = np.eye(2, dtype=np.complex_)

X = np.matrix([[0, 1], [1, 0]], dtype=np.complex_)

Y = np.matrix([[0, -1j], [1j, 0]], dtype=np.complex_)

Z = np.matrix([[1, 0], [0, -1]], dtype=np.complex_)

H = (1/sqrt(2.))*np.matrix([[1, 1], [1, -1]], dtype=np.complex_)

S = np.matrix([[1, 0], [0, 1j]], dtype=np.complex_)

T = np.matrix([[1, 0], [0, cexp(1j*pi/4)]], dtype=np.complex_)

ROOT_T = np.matrix([[1, 0], [0, cexp(1j*pi/8)]], dtype=np.complex_)


ROOT_ROOT_T = np.matrix([[1, 0], [0, cexp(1j*pi/16)]], dtype=np.complex_)


CS = np.matrix([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1j]], dtype=np.complex_)


CCZ = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, -1]],
                dtype=np.complex_)


def hrandom_su2():
    phi = asin(sqrt(random()))
    psi = random()*2*pi
    chi = random()*2*pi
    return np.matrix([[cexp(1j*psi)*cos(phi), cexp(1j*chi)*sin(phi)],
                      [-1*cexp(-1j*chi)*sin(phi), cexp(-1j*psi)*cos(phi)]],
                     dtype=np.complex_)
