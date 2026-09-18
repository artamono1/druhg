# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Typed (float key, int payload) min-heap for MST targeting bulks.
# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np


cdef inline bint _fi_less(np.double_t ka, np.intp_t va,
                          np.double_t kb, np.intp_t vb) nogil:
    # Same order as Python heapq on (key, val) tuples.
    return ka < kb or (ka == kb and va < vb)


cdef class FloatIntMinHeap:
    """Typed binary min-heap of (float key, int payload) pairs."""

    def __cinit__(self, np.intp_t capacity=8):
        if capacity < 1:
            capacity = 1
        self.capacity = capacity
        self.size = 0
        self.keys_arr = np.empty(capacity, dtype=np.float64)
        self.vals_arr = np.empty(capacity, dtype=np.intp)
        self.keys = self.keys_arr
        self.vals = self.vals_arr

    cdef void _grow(self, np.intp_t need) except *:
        cdef np.intp_t new_cap, n
        cdef np.ndarray keys_new, vals_new

        new_cap = self.capacity << 1
        if new_cap < need:
            new_cap = need
        keys_new = np.empty(new_cap, dtype=np.float64)
        vals_new = np.empty(new_cap, dtype=np.intp)
        n = self.size
        if n:
            keys_new[:n] = self.keys_arr[:n]
            vals_new[:n] = self.vals_arr[:n]
        self.keys_arr = keys_new
        self.vals_arr = vals_new
        self.keys = self.keys_arr
        self.vals = self.vals_arr
        self.capacity = new_cap

    cdef void _siftdown(self, np.intp_t startpos, np.intp_t pos) noexcept nogil:
        # Bubble newitem at pos up toward startpos (Python heapq._siftdown).
        cdef np.intp_t parentpos
        cdef np.double_t new_key, parent_key
        cdef np.intp_t new_val, parent_val

        new_key = self.keys[pos]
        new_val = self.vals[pos]
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent_key = self.keys[parentpos]
            parent_val = self.vals[parentpos]
            if not _fi_less(new_key, new_val, parent_key, parent_val):
                break
            self.keys[pos] = parent_key
            self.vals[pos] = parent_val
            pos = parentpos
        self.keys[pos] = new_key
        self.vals[pos] = new_val

    cdef void _siftup(self, np.intp_t pos) noexcept nogil:
        # Bubble hole at pos down, then siftdown (Python heapq._siftup).
        cdef np.intp_t endpos, startpos, childpos, rightpos
        cdef np.double_t new_key, child_key, right_key
        cdef np.intp_t new_val, child_val, right_val

        endpos = self.size
        startpos = pos
        new_key = self.keys[pos]
        new_val = self.vals[pos]
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            child_key = self.keys[childpos]
            child_val = self.vals[childpos]
            if rightpos < endpos:
                right_key = self.keys[rightpos]
                right_val = self.vals[rightpos]
                if not _fi_less(child_key, child_val, right_key, right_val):
                    childpos = rightpos
                    child_key = right_key
                    child_val = right_val
            self.keys[pos] = child_key
            self.vals[pos] = child_val
            pos = childpos
            childpos = 2 * pos + 1
        self.keys[pos] = new_key
        self.vals[pos] = new_val
        self._siftdown(startpos, pos)

    cdef void push(self, np.double_t key, np.intp_t val) except *:
        cdef np.intp_t pos

        pos = self.size
        if pos >= self.capacity:
            self._grow(pos + 1)
        self.keys[pos] = key
        self.vals[pos] = val
        self.size = pos + 1
        self._siftdown(0, pos)

    cdef void pop(self) except *:
        cdef np.intp_t last

        last = self.size - 1
        self.size = last
        if last > 0:
            self.keys[0] = self.keys[last]
            self.vals[0] = self.vals[last]
            self._siftup(0)

    cdef void heapify(self) noexcept nogil:
        cdef np.intp_t i

        i = self.size >> 1
        while i > 0:
            i -= 1
            self._siftup(i)

    cdef void extend_from(self, FloatIntMinHeap other) except *:
        cdef np.intp_t i, n, need

        n = other.size
        if n == 0:
            return
        need = self.size + n
        if need > self.capacity:
            self._grow(need)
        for i in range(n):
            self.keys[self.size + i] = other.keys[i]
            self.vals[self.size + i] = other.vals[i]
        self.size = need
        self.heapify()
