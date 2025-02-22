# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Union find structure is filled in the tree part and reused in labeling/motion parts.
# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np


cdef class UnionFind (object):
    cdef:
        np.intp_t p_size
        np.ndarray parent_arr
        np.intp_t *parent
        np.ndarray fast_arr
        # np.intp_t *fast
        np.intp_t[:] fast

        np.intp_t next_label

    cdef np.intp_t get_offset(self)

    cdef np.intp_t nullify(self)

    cdef np.intp_t mark_up(self, np.intp_t n)
    cdef np.intp_t is_same_parent(self, np.intp_t p, np.intp_t on)
    cdef np.intp_t union(self, np.intp_t n, np.intp_t on, np.intp_t p, np.intp_t op)
