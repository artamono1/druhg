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

def allocate_unionfind_pair(np.intp_t N):
    buffer_parents = np.zeros(2 * N, dtype=np.intp) # is used to store all the connections and to pass between phases
    buffer_fast = np.zeros(N, dtype=np.intp) # is used only in the first phase for the fast access
    return buffer_parents, buffer_fast

cdef class UnionFind (object):

    def __init__(self, np.intp_t N, buffer_parents, buffer_fast=None):
        self.p_size = N
        self.next_label = N + 1

        self.parent_arr = buffer_parents
        self.parent = NULL

        self.fast_arr = buffer_fast
        # self.fast = NULL

        if buffer_parents is None:
            print('ERROR: buffer was not provided')
            return
        elif len(self.parent_arr) < 2*N:
            print('ERROR: parent_arr is too small', len(self.parent_arr), 2*N)
            return
        else:
            self.parent = (<np.intp_t *> self.parent_arr.data)

        if buffer_fast is None:
            return
        elif len(self.fast_arr) < N:
            print('ERROR: fast_arr is too small', len(self.fast_arr), N)
            return
        else:
            # self.fast = (<np.intp_t *> self.fast_arr.data)
            self.fast : np.intp_t[:] = self.fast_arr

    cdef np.intp_t get_offset(self):
        return self.p_size + 1

    cdef np.intp_t nullify(self):
        self.parent_arr[:2 * self.p_size] = 0
        i = self.p_size
        while i!=0:
            i -= 1
            self.fast[i] = i

    cdef np.intp_t mark_up(self, np.intp_t n):
        cdef np.intp_t p

        p = self.fast[n]
        while self.parent[p] != 0:
            assert p != self.parent[p]
            p = self.parent[p]

        self.fast[n] = p
        return p

    cdef np.intp_t is_same_parent(self, np.intp_t p, np.intp_t on):
        cdef np.intp_t op
        op = self.fast[on]

        if p == op:
            return 1
        return p == self.mark_up(on)

    cdef np.intp_t union(self, np.intp_t n, np.intp_t on, np.intp_t p, np.intp_t op):
        pp = self.next_label
        self.fast[n] = self.fast[on] = pp
        self.parent[p] = self.parent[op] = pp

        self.next_label += 1
        return pp
