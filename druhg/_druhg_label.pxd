# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind
