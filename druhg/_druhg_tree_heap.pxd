# cython: language_level=3

cimport numpy as np


cdef class FloatIntMinHeap:
    cdef:
        np.double_t *keys
        np.intp_t *vals
        np.intp_t size
        np.intp_t capacity

    cdef void _grow(self, np.intp_t need) except *
    cdef void _siftdown(self, np.intp_t startpos, np.intp_t pos) noexcept nogil
    cdef void _siftup(self, np.intp_t pos) noexcept nogil
    cdef void push(self, np.double_t key, np.intp_t val) except *
    cdef void pop(self) noexcept nogil
    cdef void replace_root(self, np.double_t key, np.intp_t val) noexcept nogil
    cdef void heapify(self) noexcept nogil
    cdef void extend_from(self, FloatIntMinHeap other) except *


cdef class IntIntMinHeap:
    cdef:
        np.intp_t *keys
        np.intp_t *vals
        np.intp_t size
        np.intp_t capacity

    cdef void _grow(self, np.intp_t need) except *
    cdef void _siftdown(self, np.intp_t startpos, np.intp_t pos) noexcept nogil
    cdef void _siftup(self, np.intp_t pos) noexcept nogil
    cdef void push(self, np.intp_t key, np.intp_t val) except *
    cdef void append_unsorted(self, np.intp_t key, np.intp_t val) except *
    cdef void pop(self) noexcept nogil
    cdef void heapify(self) noexcept nogil
