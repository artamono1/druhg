import numpy as np
cimport numpy as np

cdef struct Relation:
    np.double_t reciprocity
    np.intp_t endpoint
    np.intp_t max_rank
    np.intp_t skip_first
