from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class Metric(ABC):
    def __init__(self, evaluated_field=None, compared_field=None):
        self.evaluated_field = evaluated_field
        self.compared_field = compared_field

    def check(self, keys: Iterable) -> bool:
        return self.evaluated_field in keys and self.compared_field in keys

    @abstractmethod
    def calculate(self, row: pd.Series):
        return


class NormalizedMetric(Metric):
    @abstractmethod
    def calculate(self, row: pd.Series):
        pass

    @abstractmethod
    def min_max(self) -> Tuple[float, float]:
        return -1, -1

    def normalize(self, result):
        try:
            mm = self.min_max()
            return (result - mm[0]) / mm[1]
        except:
            return


class NumCategoryMetric(NormalizedMetric):
    def __init__(self, evaluated_field=None, compared_field=None, categories: Tuple = None):
        super().__init__(evaluated_field, compared_field)
        self.categories = categories

    def get_category(self, value):
        if value < self.categories[0][0]:
            return 0

        return next((i + 1 for i, c in enumerate(self.categories) if c[0] <= value < c[1]), len(self.categories) + 1)

    def calculate(self, row: pd.Series):
        try:
            return np.abs(
                self.get_category(row[self.evaluated_field])
                - self.get_category(row[self.compared_field])
            )
        except:
            return

    def min_max(self) -> Tuple[float, float]:
        return 0, len(self.categories) + 1


class MatrixMetric(Metric):
    @staticmethod
    def get_matrix_by_values(values=None, names: Tuple = None):
        return pd.DataFrame(data=values, columns=names, index=names)

    def __init__(self, evaluated_field=None, compared_field=None, matrix: pd.DataFrame = None):
        super().__init__(evaluated_field, compared_field)
        self.matrix = matrix

    def calculate(self, row: pd.Series):
        try:
            return self.matrix.loc[
                row[self.evaluated_field], row[self.compared_field]
            ]
        except:
            return

    def min_max(self) -> Tuple[float, float]:
        return self.matrix.min().min(), self.matrix.max().max()


class NumDifferentMetric(NumCategoryMetric):
    def __init__(self, evaluated_field=None, compared_field=None, categories: Tuple = None, percent: bool = True,
                 absolute=True):
        super().__init__(evaluated_field, compared_field, categories)
        self.percent = percent
        self.absolute = absolute

    def calc_dif(self, row: pd.Series):
        dif = row[self.evaluated_field] - row[self.compared_field]
        if self.percent:
            dif = dif / row[self.evaluated_field] * 100
        if self.absolute:
            dif = abs(dif)
        return dif

    def calculate(self, row: pd.Series):
        try:
            return self.get_category(self.calc_dif(row))
        except:
            return
