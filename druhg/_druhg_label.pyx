# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Labels the nodes using buffers from previous DRUHG's results.
# Also provides tools for label manipulations, such as:
# * Treats small clusters as outliers
# * (on-demand) Breaks big clusters
# * (on-demand) Glues outliers to the nearest clusters
#
# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

from ._druhg_group cimport set_precision
from ._druhg_group import Group
from ._druhg_group cimport Group

def allocate_buffer_labels(np.intp_t size):
    return np.empty(size, dtype=np.intp)

cdef class Clusterizer(object):
    cdef UnionFind _U
    cdef np.ndarray _values_arr
    cdef np.ndarray _data_arr # for motion only
    cdef np.ndarray group_arr
    cdef np.ndarray ret_sizes
    cdef np.ndarray ret_clusters

    def __init__(self, np.ndarray _uf_arr, int _size, np.ndarray _values_arr, np.ndarray _data_arr,
                np.ndarray buf_ret_clusters,
                np.ndarray buf_ret_sizes,
                np.ndarray buf_group_arr):
        """ Uses the results of DRUHG MST-tree algorithm(unionfind structure and values).
            Emerge clusters and sizes arrays for later labeling.

            Parameters
            ----------
            _uf_arr : ndarray
                Unionfind structure from first phase.

            _size : int
                Amount of nodes.

            _values_arr : ndarray
                Edge values two for each edge.

            _data_arr : ndarray, nullable
                Points coordinates (motion only).

            buf_ret_clusters : ndarray
                Buffer related to 2nd half of UF. If positive then parent clusters it.

            buf_ret_sizes : ndarray
                Buffer related to 2nd half of UF. Node size (points, not edges).

            buf_group_arr : ndarray
                Buffer related to 2nd half of UF.

        """

        if _size is None or _size == 0:
            _size = int((len(_uf_arr) + 1) / 2)
        self._U = UnionFind(_size, _uf_arr)

        self._data_arr = _data_arr
        self._values_arr = _values_arr

        # TODO: check the allocations and the size
        # if ret_labels is not None and len(ret_labels) < self._U.p_size:
        #     print('ERROR: labels buffer is too small', len(ret_labels), self._U.p_size)
        #     return

        self.group_arr = buf_group_arr
        self.ret_sizes = buf_ret_sizes
        self.ret_clusters = buf_ret_clusters


    cpdef emerge(self, precision=0.0000001, run_motion = False):
        cdef:
            np.intp_t p, i, cluster_size, loop_size, \
                offset = self._U.get_offset()
            np.double_t v, limit = 0.

        set_precision(precision)

        # if self._data_arr is None and run_motion:
        #     print('ERROR: data values are missing')
        #     return

        self.group_arr[:self._U.p_size].fill(0)
        self.ret_clusters[:self._U.p_size].fill(0)
        self.ret_sizes[:self._U.p_size].fill(0)

        # helpers
        x_group = Group(np.zeros_like(self.group_arr[:1])[0])
        y_group = Group(np.zeros_like(self.group_arr[:1])[0])
        p_group = Group(np.zeros_like(self.group_arr[:1])[0])
        outlier_group = Group(np.zeros_like(self.group_arr[:1])[0])
        outlier_group_cluster = 0
        outlier_group.cook_outlier(0)

        loop_size1, loop_size2 = self._U.p_size, self._U.p_size * 2

        for u in range(loop_size1):
            p = self._U.parent[u]
            if p == 0:  # in case a point doesnt have any connection
                continue
            assert p >= self._U.p_size
            p = p - offset
            assert 0 <= p <= self._U.p_size
            v = self._values_arr[p]
            # r = self.ranks_arr[has_ranks * p]

            # first ever node connection
            p_group.assume_data(self.group_arr[p], self.ret_sizes[p], self.ret_clusters[p])
            p_group.child(u, p_group.points() == 0)

            if run_motion:
                p_group.mtn_add_1_coords(self._data_arr[u], v)

            p_group.add_1_autocluster(v)
            self.ret_sizes[p] = p_group.points()
            self.ret_clusters[p] = p_group.uniq_edges()

        for u in range(loop_size1 + 1, loop_size2):
            i = u - offset
            assert 0 <= i <= self._U.p_size

            if self.ret_sizes[i] == 0:
                break
            x_group.assume_data(self.group_arr[i], self.ret_sizes[i], self.ret_clusters[i])

            p = self._U.parent[u]
            if p == 0:
                continue
            p = p - offset
            assert 0 <= p <= self._U.p_size

            p_group.assume_data(self.group_arr[p], self.ret_sizes[p], self.ret_clusters[p])

            has_child, is_outlier, j = p_group.child(i, False)
            if not has_child:  # save to process later as pair
                continue

            v = self._values_arr[p]

            if is_outlier:
                y_group.assume_data(outlier_group.data, 1, 0)  # плохо сделано, надо отдельный метод
                y_is_cluster = True
            else:
                y_group.assume_data(self.group_arr[j],
                                    self.ret_sizes[j],
                                    self.ret_clusters[j])  # можем таргетить дефолта? Когда координаты не нужны - 100%
                y_is_cluster = False

            common_coef = 1. / (x_group.uniq_edges() + y_group.uniq_edges() - 1.)
            # print('common_coef', common_coef)
            x_is_cluster = x_group.will_cluster(v, common_coef * y_group.points())
            if not is_outlier:
                y_is_cluster = y_group.will_cluster(v, common_coef * x_group.points())

            p_group.aggregate(v, x_is_cluster, x_group, y_is_cluster, y_group)

            self.ret_sizes[p] = p_group.points()
            self.ret_clusters[p] = p_group.uniq_edges()

            assert x_group.uniq_edges() <= 0
            # non-clusters are negative
            self.ret_clusters[i] = -x_group.uniq_edges() if x_is_cluster else x_group.uniq_edges()
            if not is_outlier:
                assert y_group.uniq_edges() <= 0
                self.ret_clusters[j] = -y_group.uniq_edges() if y_is_cluster else y_group.uniq_edges()

            # if run_motion:
            #     # outlier coords already in p_group
            #     p_group.mtn_aggregate(v, x_is_cluster, x_group.data, y_is_cluster, y_group.data)
            #     x_group.mtn_mark_cluster(x_is_cluster)
            #     y_group.mtn_mark_cluster(y_is_cluster)

        if run_motion:
            return self.group_arr
        return self.ret_clusters, self.ret_sizes

    cdef void _fixem(self, np.ndarray edges_arr, np.intp_t num_edges, np.ndarray result):
        cdef:
            np.intp_t p, a, b, dontstop
            set new_results, links
            list new_path, restart

        new_results = set()
        new_path = []
        restart = []
        for p in range(0, num_edges):
            a, b = edges_arr[2 * p], edges_arr[2 * p + 1]
            if result[a] < 0 and result[b] < 0:
                new_results.update([a, b])
                new_path.append((a, b))
                continue
            elif result[b] < 0:
                a, b = b, a
            elif result[a] >= 0:
                continue
            res = result[b]
            result[a] = res
            if a in new_results:
                links = set([a])
                dontstop = 1
                while dontstop:
                    dontstop = 0
                    for path in list(new_path):
                        a, b = path
                        if a in links or b in links:
                            result[a] = result[b] = res
                            links.update([a, b])
                            new_path.remove(path)
                            dontstop = 1

        return

    cdef _mark_labels(self, ret_labels,
                     list exclude = None,
                      np.intp_t limitL = 0, np.intp_t limitH = 0,
                     ):
        cdef np.intp_t i, p, pp, label, offset

        offset = self._U.get_offset()

        i = self._U.p_size
        while i:
            i -= 1
            p = self._U.parent[i]
            label = -1
            while p != 0:
                pp = p - offset
                cluster_size = self.ret_sizes[pp]
                if cluster_size > limitH:
                    break
                if self.ret_clusters[pp] > 0 and cluster_size >= limitL and pp not in exclude:
                    label = pp
                p = self._U.parent[p]
            ret_labels[i] = label
        return ret_labels

    cpdef np.ndarray label(self, np.ndarray ret_labels,
                                list exclude=None, size_range=None,
                                np.intp_t fix_outliers=0, edgepairs_arr=None,
                                precision=0.0000001):
        """Returns cluster labels and clusters densities.
    
        Marks data-points with corresponding parent index of a cluster.
        Exclude list breaks passed clusters by their parent index.
        `size_range` breaks clusters outside it's range.
        Outliers-noise marked by -1.
    
        Parameters
        ----------
    
        ret_labels : ndarray
            The result. -1 are outliers.    
    
        exclude : list
            Clusters with parent-index from this list will not be formed. 
    
        size_range : list, optional (default=[1,size])
            Clusters that are smaller or over than the range treated as noise. 
            Pass None to find True outliers.
    
        fix_outliers: bool, optional (default=False)
            All outliers will be assigned to the nearest cluster. Need to pass mst(edgepairs).
    
        edgepairs_arr: array, optional (default=None)+
            Used with fix_outliers.
            
        precision: double, optional
            Relevant for small distances
    
        Returns
        -------
    
        labels : array [size]
           An array of cluster labels, one per data-point. Unclustered points get
           the label -1.
    
        metalabels : dictionary, on-demand
            A dictionary: keys - labels, values - tuples (distance, rank).   
    
        """
        cdef:
            int i

        if size_range is None:
            limitL, limitH = 0, 0
        else:
            limitL, limitH = size_range[0], size_range[1]
        if limitL < 0 or limitL > self._U.p_size:
            print ('label: size_range[0] is ignored. Cannot use '+str(limitL))
            limitL = 0
        if limitL < 1:
            limitL = int(limitL * self._U.p_size)

        if limitH <= 0 or limitH > self._U.p_size + 1:
            print ('label: size_range[1] is ignored. Cannot use ' + str(limitH))
            limitH = self._U.p_size
        if limitH <= 1:
            limitH = int(limitH * self._U.p_size + 1)

        if not exclude:
            exclude = []

        if ret_labels is not None and len(ret_labels) < self._U.p_size:
            print('ERROR: labels buffer is too small', len(ret_labels), self._U.p_size)
            return

        ret_labels = self._mark_labels(ret_labels,
                                 exclude, limitL, limitH)

        if fix_outliers == 1 and len(np.unique(ret_labels)) > 1:
            if edgepairs_arr is not None:
                self._fixem(edgepairs_arr, self._U.p_size - 1, ret_labels)
            else:
                print('To fix_outliers pass edgepairs', edgepairs_arr)

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
