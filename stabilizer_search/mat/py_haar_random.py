from cmath import exp as cexp
from math import asin, cos, pi, sin, sqrt
from random import random

import numpy as np

def pyget_su2():
    phi = asin(sqrt(random()))
    psi = random()*2*pi
    chi = random()*2*pi
    return np.matrix([[cexp(1j*psi)*cos(phi), cexp(1j*chi)*sin(phi)],
                      [-1*cexp(-1j*chi)*sin(phi), cexp(-1j*psi)*cos(phi)]],
                     dtype=np.complex_)