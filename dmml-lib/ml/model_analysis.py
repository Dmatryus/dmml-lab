import umap
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from lib.ml.clustering import Clustering, DBSCAN, SimplexBlackHole, Hierarchical
from pathlib import Path


def plot(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        plt.grid()
        plt.show()

    return wrapper


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


if __name__ == "__main__":
    import pandas as pd

    def srg_research():
        moscow_vectors = pd.read_csv(
            "/home/dmatryus/Projects/SRG/adjustments/moscow_vectors.csv"
        )
        input_vectors = moscow_vectors.values
        input_vectors = input_vectors
        embedding = umap.UMAP().fit_transform(input_vectors)
        pd.DataFrame(embedding).to_csv(
            "/home/dmatryus/Projects/SRG/adjustments/umap.csv", index=False
        )
        embedding = pd.read_csv(
            "/home/dmatryus/Projects/SRG/adjustments/umap.csv"
        ).values

        # sbh_mark = 0.0001
        # sbh = SimplexBlackHole(sbh_mark, False, True)
        # v = sbh.clustering(input_vectors)
        # ac = ClusteringAnalyser(r'/home/dmatryus/Projects/SRG/adjustments/price clusterization/sbh', str(sbh_mark)).analyze(
        #     sbh,
        #     data_2d=input_vectors
        # )
        # print(ac.metrics)

        # dbscan_mark = 0.6
        # dbscan = DBSCAN(min_points=20, epsilon=dbscan_mark)
        # v = dbscan.clustering(input_vectors)
        # # sbhc_analysis(embedding, sbh)
        #
        # ac = ClusteringAnalyser(r'/home/dmatryus/Projects/SRG/adjustments/price clusterization/dbscan',
        #                         str(dbscan_mark)).analyze(
        #     dbscan,
        #     data_2d=embedding
        # )
        # print(ac.metrics)

        # outers_points = np.array([input_vectors[i] for i in dbscan.outers_indexes])
        # outers_points_2d = np.array([embedding[i] for i in dbscan.outers_indexes])

        outers_points = input_vectors
        outers_points_2d = embedding
        h_mark = 25
        h = Hierarchical(distance_threshold=h_mark)
        h.clustering(outers_points)
        ac = ClusteringAnalyser(
            r"/home/dmatryus/Projects/SRG/adjustments/price clusterization/hierarchical",
            str(h_mark),
        ).analyze(h, data_2d=outers_points_2d)
        print(ac.metrics)
        # ocs = h.result + dbscan.result.max()
        # oc = 0
        # clusters = dbscan.result
        # for i, c in enumerate(dbscan.result):
        #     if c == -1:
        #         clusters[i] = ocs[oc]
        #         oc += 1
        # print(len(input_vectors), len(clusters))
        moscow_vectors["cluster"] = h.result
        print(len(moscow_vectors["cluster"].unique()))
        moscow_vectors.to_csv(
            "/home/dmatryus/Projects/SRG/adjustments/moscow_clusters.csv", index=False
        )

    def simplex_black_hole_research(test_name, epsilon=0.1, n_samples=15000):
        from sklearn import datasets

        def generate_ansio(n_samples):
            random_state = 170
            X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
            transformation = [[0.6, -0.6], [-0.4, 0.8]]
            X_aniso = np.dot(X, transformation)
            return (X_aniso, y)

        def test_data(data, epsilon, name):
            sbh = SimplexBlackHole(epsilon, analysis_mode=True)
            sbh.clustering(data)
            analysis = ClusteringAnalyser(
                f"/home/dmatryus/Projects/self/ul_lib/experiments/scikitlearn_data_experiment/sbh/{name}",
                str(epsilon),
            ).analyze(sbh, data_2d=data)
            print(analysis.metrics)

        test_datasets = {
            "noisy_circles": datasets.make_circles(
                n_samples=n_samples, factor=0.5, noise=0.05
            ),
            "noisy_moons": datasets.make_moons(n_samples=n_samples, noise=0.05),
            "blobs": datasets.make_blobs(n_samples=n_samples, random_state=8),
            "ansio": generate_ansio(n_samples),
            "no_structure": (np.random.rand(n_samples, 2), None),
        }

        test_data(test_datasets[test_name][0], epsilon, test_name)

    # simplex_black_hole_research('noisy_circles', 0.0025) #bad
    simplex_black_hole_research("noisy_moons", 0.0025)  # bad
