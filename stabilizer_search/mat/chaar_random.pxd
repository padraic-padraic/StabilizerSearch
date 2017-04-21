cdef extern from "complex.h":
    pass

cdef extern from "haarrandom.h":
    long double ** random_so2();
    long double complex ** random_su2();
