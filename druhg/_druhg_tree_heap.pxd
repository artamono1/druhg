# cython: language_level=3

import numpy as np
cimport numpy as np


cdef class FloatIntMinHeap:
    cdef:
        np.ndarray keys_arr
        np.ndarray vals_arr
        np.double_t[::1] keys
        np.intp_t[::1] vals
        np.intp_t size
        np.intp_t capacity

    cdef void _grow(self, np.intp_t need) except *
    cdef void _siftdown(self, np.intp_t startpos, np.intp_t pos) noexcept nogil
    cdef void _siftup(self, np.intp_t pos) noexcept nogil
    cdef void push(self, np.double_t key, np.intp_t val) except *
    cdef void pop(self) except *
    cdef void heapify(self) noexcept nogil
    cdef void extend_from(self, FloatIntMinHeap other) except *
