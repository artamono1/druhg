# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Labels the nodes using buffers from previous DRUHG's results.
# Also provides tools for label manipulations, such as:
# * Treats small clusters as outliers on-demand
# * Breaks big clusters on-demand
# * Glues outliers to the nearest clusters on-demand
#
# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

from ._druhg_group cimport group_is_cluster, set_precision
from ._druhg_group import Group
from ._druhg_group cimport Group

def allocate_buffer_labels(np.intp_t size):
    return np.empty(size, dtype=np.intp)

cdef getIndex(UnionFind U, np.intp_t p):
    return p - U.p_size - 1

cdef void fixem(np.ndarray edges_arr, np.intp_t num_edges, np.ndarray result):
    cdef:
        np.intp_t p, a, b, dontstop
        set new_results, links
        list new_path, restart

    new_results = set()
    new_path = []
    restart = []
    for p in range(0, num_edges):
        a, b = edges_arr[2*p], edges_arr[2*p + 1]
        if result[a] < 0 and result[b] < 0:
            new_results.update([a,b])
            new_path.append((a,b))
            continue
        elif result[b] < 0:
            a,b = b,a
        elif result[a] >= 0:
            continue
        res = result[b]
        result[a] = res
        if a in new_results:
            links = set(a)
            dontstop = 1
            while dontstop:
                dontstop = 0
                for path in list(new_path):
                    a, b = path
                    if a in links or b in links:
                        result[a] = result[b] = res
                        # print ('new_r', a, b, res)
                        links.update([a,b])
                        new_path.remove(path)
                        dontstop = 1

    return

cdef mark_labels(UnionFind U, dict clusters, np.ndarray ret_labels):
    cdef np.intp_t i, p, pp, label
    i = U.p_size
    while i:
        i -= 1
        p = U.parent[i]
        label = -1
        while p != 0:
            pp = getIndex(U,p)
            if pp in clusters:
                label = pp
                # break # can't break - the dict contains subclusters
            p = U.parent[p]
        ret_labels[i] = label
    return ret_labels

cdef emerge_clusters(UnionFind U, np.ndarray values_arr, np.ndarray group_arr,
                     list exclude = None, np.intp_t limit1 = 0, np.intp_t limit2 = 0,
                     np.ndarray ranks_arr = None):
    cdef:
        np.intp_t p, i, cluster_size, loop_size, \
            has_ranks
        np.double_t v, limit = 0.
        dict ret_clusters = {}
    has_ranks = 1
    if ranks_arr is None:
        has_ranks = 0
        ranks_arr = np.zeros(1)

    p_group = Group(group_arr[0]) # helper

    loop_size1, loop_size2 = U.p_size, U.p_size * 2
    for u in range(loop_size1):
        p = U.parent[u]
        if p == 0:
            continue
        assert p >= U.p_size
        p = getIndex(U, p)
        assert p < U.p_size
        v = values_arr[p]
        # first ever node connection
        p_group.assume_data(group_arr[p])
        p_group.add_1(v)

    for u in range(loop_size1, loop_size2):
        p = U.parent[u]
        if p == 0:
            continue
        p = getIndex(U, p)
        p_group.assume_data(group_arr[p])
        v = values_arr[p]

        i = getIndex(U, u)

        if not group_is_cluster(group_arr[i], &limit, v):
            p_group.aggregate(group_arr[i])
            continue

        p_group.add_cluster(limit, group_arr[i])
        cluster_size = group_arr[i][0]
        if limit1 < cluster_size < limit2 \
                and i not in exclude: # эта структура будет включать в себя подкластеры
            # TODO: should we include subclusters?
            ret_clusters[i] = (i, cluster_size, limit, group_arr[i][2], v, ranks_arr[has_ranks * p])

    return ret_clusters

cpdef np.ndarray label(np.ndarray values_arr, np.ndarray uf_arr, int size,
                       np.ndarray group_arr, np.ndarray ret_labels,
                       np.ndarray ranks_arr=None,
                       list exclude = None, np.intp_t limit1 = 0, np.intp_t limit2 = 0,
                       np.intp_t fix_outliers = 0, edgepairs_arr=None,
                       precision=0.0000001):
    """Returns cluster labels and clusters densities.
    
    Uses the results of DRUHG MST-tree algorithm(edges and values).
    Marks data-points with corresponding parent index of a cluster.
    Exclude list breaks passed clusters by their parent index.
    The parameters `limit1` and 'limit2' allows the clustering to declare noise points.
    Outliers-noise marked by -1.

    Parameters
    ----------
    edges_arr : ndarray
        Edgepair nodes of mst.
    
    values_arr : ndarray
        Edge values two for each edge.

    ranks_arr : ndarray
        Edge ranks one for each edge.
        
    size : int
        Amount of nodes.
        
    exclude : list
        Clusters with parent-index from this list will not be formed. 
    
    limit1 : int, optional (default=sqrt(size))
        Clusters that are smaller than this limit treated as noise. 
        Use 1 to find True outliers.
        
    limit2 : int, optional (default=size)
        Clusters with size OVER this limit treated as noise. 
        Use it to break down big clusters.
 
    fix_outliers: int, optional (default=0)
        All outliers will be assigned to the nearest cluster. Need to pass mst(edgepairs).
    
    edgepairs_arr: array, optional (default=None)
        Used with fix_outliers.

    Returns
    -------

    labels : array [size]
       An array of cluster labels, one per data-point. Unclustered points get
       the label -1.
       
    clusters : dictionary
        A dictionary: keys - labels, values - tuples (distance, rank).   
       
    """

    cdef:
        dict ret_clusters
        int i

    if limit1 <= 0:
        limit1 = int(np.ceil(np.sqrt(size)))
        print ('label: default value for limit1 is used, clusters below '+str(limit1)+' are considered as noise.')

    if limit2 <= 0:
        limit2 = size
        print ('label: default value for limit2 is used, clusters above '+str(limit2)+' will not be formed.')

    # this is only relevant if distances between datapoints are super small
    if precision is None:
        precision = 0.0000001
    set_precision(precision)

    if size == 0:
        size = int((len(uf_arr)+1)/2)

    if not exclude:
        exclude = []

    if ret_labels is not None and len(ret_labels) < size:
        print('ERROR: labels buffer is too small', len(ret_labels), size)
        return

    group_arr[:size].fill(0)

    U = UnionFind(size, uf_arr)
    ret_clusters = emerge_clusters(U, values_arr, group_arr,
                   exclude, limit1, limit2,
                   ranks_arr)

    ret_labels = mark_labels(U, ret_clusters, ret_labels)

    if fix_outliers != 0 and len(np.unique(ret_labels))>1 and edgepairs_arr is not None:
        fixem(edgepairs_arr, size, ret_labels) # only possible through edgepairs

    return ret_labels


cdef np.ndarray pretty(np.ndarray labels_arr):
    """ Relabels to pretty positive integers. 
    """
    cdef np.intp_t i, p, label, max_label
    cdef np.ndarray[np.intp_t, ndim=1] result_arr
    cdef dict converter
    cdef np.intp_t* result

    result_arr = -1*np.ones(len(labels_arr), dtype=np.intp)
    result = (<np.intp_t *> result_arr.data)

    converter = {-1: -1}
    max_label = 0
    i = len(labels_arr)
    while i:
        i -= 1
        p = labels_arr[i]
        if p in converter:
            label = converter[p]
        else:
            label = max_label
            converter[p] = max_label
            max_label += 1
        result[i] = label

    return result_arr
