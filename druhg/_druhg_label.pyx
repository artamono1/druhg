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
import logging

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

from ._druhg_group import Group
from ._druhg_group cimport Group
from ._druhg_group cimport GroupNode
from ._druhg_group cimport set_precision

def allocate_buffer_labels(np.intp_t size):
    return np.empty(size, dtype=np.intp)

cdef class Clusterizer:
    def __init__(self, np.ndarray _uf_arr, int _size, np.ndarray _values_arr, object _data_arr,
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

        self._data_arr = _data_arr if isinstance(_data_arr, np.ndarray) else None
        self._values_arr = _values_arr

        # TODO: check the allocations and the size
        self.group_arr = buf_group_arr
        self.ret_sizes = buf_ret_sizes
        self.ret_clusters = buf_ret_clusters

    cpdef emerge(self, precision=0.0000001, run_motion = False):
        if precision is not None and precision > 0:
            set_precision(precision)

        self.ret_clusters[:self._U.p_size].fill(0)
        self.ret_sizes[:self._U.p_size].fill(0)

        return self.emerge_still()

    cdef emerge_still(self):
        cdef:
            np.intp_t p, u, i, j,
            x_size, y_size,

            loop_size1 = self._U.p_size,
            loop_size2 = self._U.p_size * 2,
            offset = self._U.get_offset()

            np.intp_t x_is_cluster, y_is_cluster
            GroupNode x_node, y_node

            np.double_t v, limit = 0.

        data_dict = {}
        outlier = GroupNode(0)  # stub, values are ignored

        logger = logging.getLogger(__package__)

        logger.info('Labeling: start')
        # first ever connection of every point
        for u in range(loop_size1):
            p = self._U.parent[u]
            if p == 0:  # point has no connection
                continue
            assert p >= self._U.p_size
            p = p - offset
            assert 0 <= p <= self._U.p_size
            v = self._values_arr[p]

            y_size = self.ret_sizes[p]
            self.ret_sizes[p] = y_size + 1

            logger.debug('%s %s %s', p, v, y_size!=0)

            if y_size != 0:  # same as y_size == 1
                self.ret_clusters[p] = 1
                data_dict[p] = GroupNode(sum_reciprocals=1. / v if v != 0. else 0.)

        logger.info('Labeling: %s nodes allocated', len(data_dict))

        # dealing with complex parents
        for u in range(loop_size1 + 1, loop_size2):
            i = u - offset
            assert 0 <= i <= self._U.p_size

            p = self._U.parent[u]
            if p == 0:
                if self.ret_sizes[i] == 0:
                    break
                continue
            p = p - offset
            assert 0 <= p <= self._U.p_size
            logger.debug('u %s %s',u, p)

            x_size = self.ret_sizes[i]
            y_size = self.ret_sizes[p]
            self.ret_sizes[p] = y_size + x_size

            logger.debug('%s y_size ', y_size)
            if y_size == 0: # first time passing, second time processing
                self.ret_clusters[p] = i # temp using future clustering place
                continue

            v = self._values_arr[p]
            logger.debug('v %s', v)
            
            # x_group cannot be an outlier by construction
            x_node = data_dict[i]
            del data_dict[i]
            x_is_cluster = self.ret_clusters[i]

            _y_is_cluster = 0  # outlier has no edges
            y_is_cluster = -1  # outlier auto-clusterize
            if y_size == 1:
                y_node = outlier  # stub, values are ignored
            else:
                j = self.ret_clusters[p]  # id, not a clusters
                logger.debug('%s -data_dict j', j)
                y_node = data_dict[j]
                del data_dict[j]
                _y_is_cluster = self.ret_clusters[j]
                y_is_cluster = Group.will_cluster(y_size, _y_is_cluster, &y_node, v, x_size, x_is_cluster, &x_node)
                self.ret_clusters[j] = y_is_cluster

            x_is_cluster = Group.will_cluster(x_size, x_is_cluster, &x_node, v, y_size, _y_is_cluster, &y_node)
            self.ret_clusters[i] = x_is_cluster

            self.ret_clusters[p] = Group.aggregate(x_size, x_is_cluster, &x_node, v, y_size, y_is_cluster, &y_node)
            assert (self.ret_clusters[p] > 0)
            data_dict[p] = x_node
            logger.debug('%s +data_dict p', p)

        del outlier
        return self.ret_clusters, self.ret_sizes, None

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
        cdef np.intp_t i, p, pp, label, offset, cluster_size

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
                if self.ret_clusters[pp] < 0 and cluster_size >= limitL and pp not in exclude:
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

        logger = logging.getLogger(__package__)

        if size_range is None:
            limitL, limitH = 0, 0
        else:
            limitL, limitH = size_range[0], size_range[1]
        if limitL < 0 or limitL > self._U.p_size:
            logger.info('label: size_range[0] is ignored. Cannot use '+str(limitL))
            limitL = 0
        if limitL < 1:
            limitL = int(limitL * self._U.p_size)

        if limitH <= 0 or limitH > self._U.p_size + 1:
            logger.info('label: size_range[1] is ignored. Cannot use ' + str(limitH))
            limitH = self._U.p_size
        if limitH <= 1:
            limitH = int(limitH * self._U.p_size + 1)

        if not exclude:
            exclude = []

        if ret_labels is not None and len(ret_labels) < self._U.p_size:
            logger.info('ERROR: labels buffer is too small '+str(len(ret_labels))+' '+str(self._U.p_size))
            return

        ret_labels = self._mark_labels(ret_labels,
                                 exclude, limitL, limitH)

        if fix_outliers == 1 and len(np.unique(ret_labels)) > 1:
            if edgepairs_arr is not None:
                self._fixem(edgepairs_arr, self._U.p_size - 1, ret_labels)
            else:
                logger.warning('To fix_outliers pass edgepairs '+str(edgepairs_arr))

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
