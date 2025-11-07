import numpy as np
cimport numpy as np

cdef struct Relation:
    np.double_t reciprocity
    np.intp_t point
    np.intp_t endpoint
    np.double_t min_rank
    np.double_t max_rank
    np.intp_t rank_reached
