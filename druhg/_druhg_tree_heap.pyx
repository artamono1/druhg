# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Typed (float key, int payload) min-heap for MST targeting bulks.
# Author: AI generated
# License: 3-clause BSD

cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy


cdef inline bint _fi_less(np.double_t ka, np.intp_t va,
                          np.double_t kb, np.intp_t vb) nogil:
    # Same order as Python heapq on (key, val) tuples.
    return ka < kb or (ka == kb and va < vb)


cdef class FloatIntMinHeap:
    """Typed binary min-heap of (float key, int payload) pairs."""

    def __cinit__(self, np.intp_t capacity=8):
        if capacity < 1:
            capacity = 1
        self.size = 0
        self.capacity = 0
        self.keys = NULL
        self.vals = NULL
        self.keys = <np.double_t *> PyMem_Malloc(capacity * sizeof(np.double_t))
        self.vals = <np.intp_t *> PyMem_Malloc(capacity * sizeof(np.intp_t))
        if self.keys == NULL or self.vals == NULL:
            raise MemoryError()
        self.capacity = capacity

    def __dealloc__(self):
        if self.keys != NULL:
            PyMem_Free(self.keys)
            self.keys = NULL
        if self.vals != NULL:
            PyMem_Free(self.vals)
            self.vals = NULL

    cdef void _grow(self, np.intp_t need) except *:
        cdef np.intp_t new_cap
        cdef np.double_t *new_keys
        cdef np.intp_t *new_vals

        new_cap = self.capacity << 1
        if new_cap < need:
            new_cap = need
        new_keys = <np.double_t *> PyMem_Realloc(self.keys, new_cap * sizeof(np.double_t))
        if new_keys == NULL:
            raise MemoryError()
        self.keys = new_keys
        new_vals = <np.intp_t *> PyMem_Realloc(self.vals, new_cap * sizeof(np.intp_t))
        if new_vals == NULL:
            raise MemoryError()
        self.vals = new_vals
        self.capacity = new_cap

    cdef void _siftdown(self, np.intp_t startpos, np.intp_t pos) noexcept nogil:
        # Bubble newitem at pos up toward startpos (Python heapq._siftdown).
        cdef np.intp_t parentpos
        cdef np.double_t new_key, parent_key
        cdef np.intp_t new_val, parent_val
        cdef np.double_t *keys
        cdef np.intp_t *vals

        keys = self.keys
        vals = self.vals
        new_key = keys[pos]
        new_val = vals[pos]
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent_key = keys[parentpos]
            parent_val = vals[parentpos]
            if not _fi_less(new_key, new_val, parent_key, parent_val):
                break
            keys[pos] = parent_key
            vals[pos] = parent_val
            pos = parentpos
        keys[pos] = new_key
        vals[pos] = new_val

    cdef void _siftup(self, np.intp_t pos) noexcept nogil:
        # Bubble hole at pos down, then siftdown (Python heapq._siftup).
        cdef np.intp_t endpos, startpos, childpos, rightpos
        cdef np.double_t new_key, child_key, right_key
        cdef np.intp_t new_val, child_val, right_val
        cdef np.double_t *keys
        cdef np.intp_t *vals

        keys = self.keys
        vals = self.vals
        endpos = self.size
        startpos = pos
        new_key = keys[pos]
        new_val = vals[pos]
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            child_key = keys[childpos]
            child_val = vals[childpos]
            if rightpos < endpos:
                right_key = keys[rightpos]
                right_val = vals[rightpos]
                if not _fi_less(child_key, child_val, right_key, right_val):
                    childpos = rightpos
                    child_key = right_key
                    child_val = right_val
            keys[pos] = child_key
            vals[pos] = child_val
            pos = childpos
            childpos = 2 * pos + 1
        keys[pos] = new_key
        vals[pos] = new_val
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

    cdef void append_unsorted(self, np.double_t key, np.intp_t val) except *:
        cdef np.intp_t pos

        pos = self.size
        if pos >= self.capacity:
            self._grow(pos + 1)
        self.keys[pos] = key
        self.vals[pos] = val
        self.size = pos + 1

    cdef void pop(self) noexcept nogil:
        cdef np.intp_t last

        last = self.size - 1
        self.size = last
        if last > 0:
            self.keys[0] = self.keys[last]
            self.vals[0] = self.vals[last]
            self._siftup(0)

    cdef void replace_root(self, np.double_t key, np.intp_t val) noexcept nogil:
        # Same payload set as pop()+push(), one sift instead of two.
        self.keys[0] = key
        self.vals[0] = val
        if self.size > 1:
            self._siftup(0)

    cdef void heapify(self) noexcept nogil:
        cdef np.intp_t i

        i = self.size >> 1
        while i > 0:
            i -= 1
            self._siftup(i)

    cdef void extend_from(self, FloatIntMinHeap other) except *:
        # Union-by-size often appends a tiny donor onto a large heap.
        # Floyd heapify of the combined array is O(|large|) and becomes
        # quadratic if size-1 bulks keep merging into a growing bulk.
        # Inserting |small| items is O(|small| log |large|); keep Floyd
        # only when the two heaps are close in size.
        cdef np.intp_t i, n, need, pos
        cdef np.double_t *okeys
        cdef np.intp_t *ovals
        cdef np.double_t *tmp_keys
        cdef np.intp_t *tmp_vals

        n = other.size
        if n == 0:
            return

        if self.size == 0:
            # Donor is already a valid heap; swap buffers instead of copying.
            tmp_keys = self.keys
            tmp_vals = self.vals
            self.keys = other.keys
            self.vals = other.vals
            other.keys = tmp_keys
            other.vals = tmp_vals
            self.size = n
            need = self.capacity
            self.capacity = other.capacity
            other.capacity = need
            other.size = 0
            return

        need = self.size + n
        if need > self.capacity:
            self._grow(need)

        okeys = other.keys
        ovals = other.vals
        if n > 1 and n * 8 >= need:
            memcpy(self.keys + self.size, okeys, n * sizeof(np.double_t))
            memcpy(self.vals + self.size, ovals, n * sizeof(np.intp_t))
            self.size = need
            self.heapify()
            return

        for i in range(n):
            pos = self.size
            self.keys[pos] = okeys[i]
            self.vals[pos] = ovals[i]
            self.size = pos + 1
            self._siftdown(0, pos)
