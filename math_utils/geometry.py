import copy
from typing import Collection, List
from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    def __init__(self):
        self._tensor = None
        self.dimension = 0

    @abstractmethod
    def _set_tensor(self, meta_tensor):
        pass

    def __str__(self):
        return str(self._tensor)

    @abstractmethod
    def _init_from_tensor(self, tensor: np.array):
        pass

    def __lt__(self, other):
        return self._tensor < other._tensor

    def __le__(self, other):
        return self._tensor <= other._tensor

    def __eq__(self, other):
        return self._tensor == other._tensor

    def __ne__(self, other):
        return self._tensor != other._tensor

    def __gt__(self, other):
        return self._tensor > other._tensor

    def __ge__(self, other):
        return self._tensor >= other._tensor

    def __hash__(self):
        return self._tensor.hash()

    def __len__(self):
        return len(self._tensor)

    def __getitem__(self, key):
        return self._tensor[key]

    def __copy__(self):
        return self._init_from_tensor(self._tensor)

    def __neg__(self):
        return self._init_from_tensor(-self._tensor)

    def __add__(self, other):
        if isinstance(other, Geometry):
            return self._init_from_tensor(self._tensor + other._tensor)
        return self._init_from_tensor(self._tensor + other)

    def __sub__(self, other):
        if isinstance(other, Geometry):
            return self._init_from_tensor(self._tensor - other._tensor)
        return self._init_from_tensor(self._tensor - other)

    def __mul__(self, other):
        if isinstance(other, Geometry):
            return self._init_from_tensor(self._tensor * other._tensor)
        return self._init_from_tensor(self._tensor * other)

    def __truediv__(self, other):
        if isinstance(other, Geometry):
            return self._init_from_tensor(self._tensor / other._tensor)
        return self._init_from_tensor(self._tensor / other)


class Rangemeter:
    @staticmethod
    def norm(x: Geometry, y: Geometry):
        t_tensor = x - y
        return np.linalg.norm(t_tensor)

    @property
    def func(self):
        return self.__func

    @func.setter
    def func(self, func):
        self._function = self._get_function(func)
        self.__func = func

    def _get_function(self, func: str):
        return self.norm

    def __init__(self, func: str = 'squared'):
        self.__func = func
        self._function = self.norm

    def range(self, x: Geometry, y: Geometry):
        return self._function(x, y)

    def range_matrix(self, x: Geometry, y: Geometry):
        # May be optimization
        distances = []
        for i in x:
            distances.append([])
            for j in y:
                distances[-1].append(self.range(i, j))
        return np.array(distances)


class Point(Geometry):
    def _set_tensor(self, meta_tensor: Collection[float]):
        self._tensor = np.array(meta_tensor)
        self.dimension = len(meta_tensor)

    @property
    def tensor(self):
        return self._tensor

    @property
    def coordinates(self):
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, cs):
        self._set_tensor(cs)
        self.__coordinates = self._tensor

    def __init__(self, coordinates: Collection[float]):
        super().__init__()
        self.__coordinates = None
        self.coordinates = coordinates

    def _init_from_tensor(self, tensor: np.array):
        return self.__class__(tensor)


class PointSet(Geometry):
    def _check_dimensions(self):
        dimension = None
        for p in self.points:
            if dimension is None:
                dimension = p.dimension
            elif dimension != p.dimension:
                raise ValueError("Not all points have the same dimension.")
        return dimension

    @property
    def tensor(self):
        return self._tensor

    def _set_tensor(self, meta_tensor: List[Point]):
        self._tensor = np.asarray([p.tensor for p in meta_tensor])
        self.dimension = self._check_dimensions()

    @property
    def points(self) -> List[Point]:
        return self.__points

    @points.setter
    def points(self, point_collection: Collection[Point]):
        self.__points = list(point_collection)
        self._set_tensor(self.__points)

    def __init__(self, points: Collection[Point]):
        super().__init__()
        self.__points = None
        self.points = points

    def _init_from_tensor(self, tensor: np.array):
        return self.__class__([Point(v) for v in tensor])

    @staticmethod
    def init_from_array(array: np.array):
        return PointSet([Point(v) for v in array])

    def append(self, other):
        if isinstance(other, Point):
            ps = copy.copy(self.points)
            ps.append(other)
            return self.__class__(ps)
        elif isinstance(other, self.__class__):
            return self.__class__(self.points + other.points)
        else:
            raise TypeError(f"Unsupported type for append {other.__class__.__name__}")

    def max_point(self):
        return Point([v.max() for v in self._tensor.T])

    def min_point(self) -> Point:
        return Point([v.min() for v in self._tensor.T])

    def normalization(self):
        shift = self.min_point()
        shifted = self - shift
        scale = shifted.max_point()
        result_points = shifted / scale
        return result_points, shift, scale

    def denormalization(self, zero_shift, normalization_scale):
        return (self * normalization_scale) + zero_shift

    def circumscribe_simplex(self):
        simplex = self._init_from_tensor(
            np.vstack((np.zeros(self.dimension), np.eye(self.dimension) * 2))
        )
        norm, shift, scale = self.normalization()
        return simplex.denormalization(shift, scale)


if __name__ == "__main__":
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()
    ax = Axes3D(fig)

    p = Point([1, -2, 3])
    ps = PointSet([p, p + 1, p * 2, p * 3, p * -3, p - 7])
    simplex = ps.circumscribe_simplex()

    print(simplex)

    ax.scatter(simplex.tensor.T[0], simplex.tensor.T[1], simplex.tensor.T[2])
    ax.scatter(ps.tensor.T[0], ps.tensor.T[1], ps.tensor.T[2], alpha=0.5)

    pyplot.show()
