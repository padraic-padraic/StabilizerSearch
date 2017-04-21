import numpy as np
cimport numpy as np

cdef gs_prj(np.ndarray[np.complex128_t, ndim=2], np.ndarray[np.complex128_t, ndim=2])

cdef gram_schmidt(list, int, int)

cpdef ortho_projector(list)
