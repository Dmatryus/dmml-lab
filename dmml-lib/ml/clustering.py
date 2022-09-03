import copy
from abc import ABC, abstractmethod
from timeit import default_timer
import sklearn.cluster as skc

import numpy as np

from math.geometry import PointSet, Rangemeter

# TODO: Rework for better usage: calculate + analysis in one class + minimal automl
class Clustering(ABC):
    def __init__(self, analysis_mode=False):
        self.analysis_mode = analysis_mode
        self.result = None

    @abstractmethod
    def clustering(self, vectorized_data: np.array) -> np.array:
        pass


class SimplexBlackHoleClustering(Clustering):
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
                temp_distances = {
                    k: black_hole_vector[k] for k in range(1, len(black_hole_vector))
                }
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
            print("clusters", len(clusters))
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


class HierarchicalClustering(Clustering):
    def init_meta(self, **kwargs):
        self._meta = skc.AgglomerativeClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
            **kwargs
        )

    def __init__(
        self,
        n_clusters=None,
        affinity="euclidean",
        linkage="ward",
        distance_threshold=None,
        analysis_mode=False,
        **kwargs
    ):
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

class ClusteringAnalysis:
    def __init__(
        self,
        object_: Clustering,
        data=None,
        data_2d=None,
        plots_path=None,
        plot_mark=None,
        **kwargs,
    ):
        self.object_ = object_
        self.data = data
        self.data_2d = data_2d
        self.plots_path = plots_path
        self.plot_mark = plot_mark
        self.metrics = {}

    def data_to_2d(self, **kwargs):
        mode = kwargs.get("mode_data_to_2d") or "umap"
        if mode == "umap":
            self.data_2d = umap.UMAP().fit_transform(self.data)

    def result_map(self, **kwargs):
        if self.data_2d is None:
            self.data_to_2d(**kwargs)

        scatter_map(
            self.data_2d,
            self.object_.result,
            self.plots_path,
            "Result 2d map",
            self.plot_mark,
        )

    @staticmethod
    def scatter_map(data_2d, values, plots_path=None, title=None, mark=None, **kwargs):
        plt.figure(figsize=kwargs.get("figsize", (12, 10)))
        plt.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c=values,
            edgecolor="none",
            alpha=kwargs.get("alpha", 0.4),
            s=40,
            cmap=plt.cm.get_cmap("nipy_spectral", len(np.unique(values))),
        )

        title = title + f" ({mark})" if mark else title

        plt.title(title)
        if plots_path and title:
            plt.savefig(plots_path / f"{title}.jpg")
        else:
            plt.show()

    def calculate_metric(self):
        label_counts = sorted(
            np.array(np.unique(self.object_.result, return_counts=True)).T,
            key=lambda kv: kv[1],
            reverse=True,
        )
        self.metrics["labels_count"] = len(np.unique(self.object_.result))
        self.metrics["top_10_clusters"] = {
            label: count for label, count in label_counts[:10]
        }
        X = self.data if self.data is not None else self.data_2d
        if X is not None:
            self.metrics["silhouette_score"] = metrics.silhouette_score(
                X, self.object_.result
            )
            self.metrics["calinski_harabasz_score"] = metrics.calinski_harabasz_score(
                X, self.object_.result
            )
            self.metrics["davies_bouldin_score"] = metrics.davies_bouldin_score(
                X, self.object_.result
            )

    def analyze(self):
        self.result_map()
        self.calculate_metric()


class DBSCANAnalysis(ClusteringAnalysis):
    def outers_map(self):
        if len(self.object_.outers_indexes) > 0:
            outers_points = np.array(
                [self.data_2d[i] for i in self.object_.outers_indexes]
            )
            scatter_map(
                outers_points,
                [-1 for i in outers_points],
                self.plots_path,
                "Outers Map",
                self.plot_mark,
            )

    def clear_map(self):
        if len(self.object_.outers_indexes) > 0:
            points = np.array(
                [
                    self.data_2d[i]
                    for i in range(len(self.data_2d))
                    if i not in self.object_.outers_indexes
                ]
            )
            scatter_map(
                points,
                [v for v in self.object_.result if v >= 0],
                self.plots_path,
                "Clear Map",
                self.plot_mark,
            )

    def calculate_metric(self):
        super().calculate_metric()
        self.metrics["outrrs_count"] = len(self.object_.outers_indexes)
        self.metrics["clean_clusters"] = 1 - len(self.object_.outers_indexes) / len(
            self.object_.result
        )

    def analyze(self):
        self.result_map()
        self.outers_map()
        self.clear_map()

        self.calculate_metric()


class SimplexBlackHoleAnalysis(ClusteringAnalysis):
    def __init__(
        self,
        object_: Clustering,
        data=None,
        data_2d=None,
        plots_path=None,
        plot_mark=None,
        **kwargs,
    ):
        super().__init__(object_, data, data_2d, plots_path, plot_mark, **kwargs)

    def distances_plot(self, **kwargs):
        plt.figure(figsize=kwargs.get("figsize", (12, 10)))
        for d in self.object_.distances:
            plt.plot(sorted(d))
        plt.legend(range(1, len(self.object_.distances) + 1))
        title = "Black Hole Distance"
        title = title + f" ({self.plot_mark})" if self.plot_mark else title
        plt.grid()
        plt.title(title)
        if self.plots_path and title:
            plt.savefig(self.plots_path / f"{title}.jpg")
        else:
            plt.show()

    def black_hole_ranges_plot(self, mode="full", **kwargs):
        plt.figure(figsize=kwargs.get("figsize", (12, 10)))
        if mode == "up":
            x = []
            y = []
            for range_ in self.object_.ranges:
                for i, r in enumerate(range_.values()):
                    if r > self.object_.epsilon:
                        x.append(r)
                        y.append(i)
            ranges = [
                [x for x in r.values() if x > self.object_.epsilon]
                for r in self.object_.ranges
            ]
            title = "Black Hole Ranges Over Epsilon"
        elif mode == "down":
            ranges = [
                [x for x in r.values() if x < self.object_.epsilon]
                for r in self.object_.ranges
            ]
            title = "Black Hole Ranges Under Epsilon"
        else:
            ranges = [r.values() for r in self.object_.ranges]
            title = "Black Hole Ranges"
        for r in ranges:
            plt.plot(r)
        plt.plot([self.object_.epsilon for i in range(len(ranges[0]))])
        plt.legend(range(1, len(self.object_.ranges) + 1))
        title = title + f" ({self.plot_mark})" if self.plot_mark else title
        plt.title(title)
        plt.grid()
        if self.plots_path and title:
            plt.savefig(self.plots_path / f"{title}.jpg")
        else:
            plt.show()

    def cluster_range_plot(self, mode="full", **kwargs):
        plt.figure(figsize=kwargs.get("figsize", (12, 10)))
        if mode == "up":
            ranges = [
                x for x in self.object_.cluster_ranges if x > self.object_.epsilon
            ]
            title = "Cluster Ranges Over Epsilon"
        elif mode == "down":
            ranges = [
                x for x in self.object_.cluster_ranges if x < self.object_.epsilon
            ]
            title = "Cluster Ranges Under Epsilon"
        else:
            ranges = self.object_.cluster_ranges
            title = "Cluster Ranges"
        plt.plot(ranges)
        plt.plot([self.object_.epsilon for i in range(len(ranges))])
        title = title + f" ({self.plot_mark})" if self.plot_mark else title
        plt.title(title)
        plt.grid()
        if self.plots_path and title:
            plt.savefig(self.plots_path / f"{title}.jpg")
        else:
            plt.show()

    def black_hole_distance_maps(self):
        for i, d in enumerate(self.object_.distances):
            scatter_map(
                self.data_2d,
                d,
                self.plots_path,
                f"Black hole distances {i}",
                self.plot_mark,
            )
        scatter_map(
            self.data_2d,
            [np.mean(point) for point in np.array(self.object_.distances).T],
            self.plots_path,
            f"Black hole distances mean",
            self.plot_mark,
        )

    def clatreization_epoch_maps(self):
        for i, e in enumerate(self.object_.epochs):
            scatter_map(
                self.data_2d,
                e,
                self.plots_path,
                f"Clatreization epoch {i}",
                self.plot_mark,
            )

    def range_maps(self):
        for i, r in enumerate(self.object_.ranges):
            scatter_map(
                self.data_2d,
                list(r.values()),
                self.plots_path,
                f"Ranges {i}",
                self.plot_mark,
            )
        scatter_map(
            self.data_2d,
            [
                np.sum(point)
                for point in np.array([list(r.values()) for r in self.object_.ranges]).T
            ],
            self.plots_path,
            f"Sum ranges",
            self.plot_mark,
        )

    def analyze(self):
        self.distances_plot()
        self.black_hole_ranges_plot()
        self.black_hole_ranges_plot("up")
        self.black_hole_ranges_plot("down")

        self.cluster_range_plot()
        self.cluster_range_plot("up")
        self.cluster_range_plot("down")

        self.black_hole_distance_maps()

        self.range_maps()

        self.clatreization_epoch_maps()
        super().analyze()


class ClusteringAnalyser:
    def __init__(self, report_path: str, plot_mark: str = None):
        self.report_path = Path(report_path)
        self._plots_path = self.report_path / "plots"
        if not self._plots_path.is_dir():
            self._plots_path.mkdir()

        self.plot_mark = plot_mark

    _analysis_mapping = {
        DBSCAN: DBSCANAnalysis,
        SimplexBlackHole: SimplexBlackHoleAnalysis,
    }

    def analyze(self, object_: Clustering, data=None, data_2d=None, **kwargs):
        class_ = object_.__class__
        analysis_class = (
            self._analysis_mapping[class_]
            if class_ in self._analysis_mapping
            else ClusteringAnalysis
        )
        analysis = analysis_class(
            object_, data, data_2d, self._plots_path, self.plot_mark, **kwargs
        )
        analysis.analyze()
        return analysis