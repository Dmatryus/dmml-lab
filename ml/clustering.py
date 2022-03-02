import copy
from abc import ABC, abstractmethod
from timeit import default_timer
import sklearn.cluster as skc

import numpy as np

from math.geometry import PointSet, Rangemeter


class Clustering(ABC):
    def __init__(self, analysis_mode=False):
        self.analysis_mode = analysis_mode
        self.result = None

    @abstractmethod
    def clustering(self, vectorized_data: np.array) -> np.array:
        pass


class SimplexBlackHole(Clustering):
    def update_range_function(self):
        if self.relative:
            self.range_function = lambda x, y: ((y - x) / x)
        else:
            self.range_function = lambda x, y: y - x

    def __init__(self, epsilon: float = 1, relative=False, analysis_mode=False):
        super().__init__(analysis_mode)
        self.distances = None
        self.epsilon = epsilon
        self.relative = relative
        self.range_function = None
        self.ranges = []
        self.epochs = []
        self.cluster_ranges = []

    def clustering(self, vectorized_data: np.array):
        def clusters_to_result():
            for i, c in enumerate(clusters):
                for key in c:
                    self.result[key] = i

        def _save_epoch():
            if self.analysis_mode:
                clusters_to_result()
                self.epochs.append(self.result.copy())

        def black_hole_filter(black_hole_vector):
            new_clusters = []
            if self.analysis_mode:
                range_map = {-1: 0, -2: 0}
                temp_distances = {k: black_hole_vector[k] for k in range(1, len(black_hole_vector))}
                sorted_points = sorted(temp_distances.items(), key=lambda kv: kv[1])
                prev_point = sorted_points[0][1]
                for key, point in sorted_points[1:]:
                    range_map[key + 2] = self.range_function(prev_point, point)
                    prev_point = point
                self.ranges.append(range_map)
            for c in clusters:
                temp_distances = {k: black_hole_vector[k] for k in c}
                sorted_points = sorted(temp_distances.items(), key=lambda kv: kv[1])
                new_cluster = [sorted_points[0][0]]
                prev_point = sorted_points[0][1]
                for key, point in sorted_points[1:]:
                    ef = self.range_function(prev_point, point)
                    if self.analysis_mode:
                        self.cluster_ranges.append(ef)
                    if ef < self.epsilon:
                        new_cluster.append(key)
                    else:
                        new_clusters.append(new_cluster)
                        new_cluster = [key]
                    prev_point = point
                new_clusters.append(new_cluster)

            return new_clusters

        self.update_range_function()
        self.cluster_ranges = []
        points = PointSet.init_from_array(vectorized_data)
        simplex = points.circumscribe_simplex()
        rangemeter = Rangemeter()
        self.distances = rangemeter.range_matrix(simplex, points)
        self.result = np.zeros(len(vectorized_data))
        clusters = [list(range(len(points)))]
        self.epochs = []
        for d in self.distances:
            clusters = black_hole_filter(d)
            print('clusters', len(clusters))
            _save_epoch()
        if self.analysis_mode:
            return self.epochs[-1]
        else:
            clusters_to_result()
            return self.result


class DBSCAN(Clustering):

    def init_meta(self, **kwargs):
        self._meta = skc.DBSCAN(eps=self.epsilon, min_samples=self.min_points, **kwargs)

    def __init__(self, epsilon=0.5, min_points=5, analysis_mode=False, **kwargs):
        super().__init__(analysis_mode)
        self.epsilon = epsilon
        self.min_points = min_points
        self._meta = None
        self.init_meta(**kwargs)

        self.outers_indexes = None

    def _prepare_result(self, ft):
        self.result = ft.labels_
        self.outers_indexes = [i for i, x in enumerate(self.result) if x == -1]

    def clustering(self, vectorized_data: np.array) -> np.array:
        ft = self._meta.fit(vectorized_data)
        self._prepare_result(ft)
        return self.result


class Hierarchical(Clustering):
    def init_meta(self, **kwargs):
        self._meta = skc.AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.affinity,
                                                 linkage=self.linkage, distance_threshold=self.distance_threshold,
                                                 **kwargs)

    def __init__(self, n_clusters=None, affinity="euclidean", linkage='ward', distance_threshold=None,
                 analysis_mode=False, **kwargs):
        super().__init__(analysis_mode)
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.init_meta(**kwargs)

    def clustering(self, vectorized_data: np.array) -> np.array:
        ft = self._meta.fit(vectorized_data)
        self.result = ft.labels_
        return self.result
