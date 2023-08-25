from typing import List, Dict, Callable

import pandas as pd
import scipy.stats as ss

try:
    from .pipeline import Pipeline, Executor
except:
    from pipeline import Pipeline, Executor


class NAAnalysis(Executor):
    def __init__(
        self, name: str = "NAAnalysis", previous_executors: List[Executor] = None
    ):
        super().__init__(name, previous_executors)

    def execute(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return pd.DataFrame(
            [data.isna().sum(), data.isna().sum() / len(data) * 100],
            index=["absolute", "percent"],
        ).T


class FeatureTypeAnalysis(Executor):
    def __init__(
        self,
        name: str = "FeatureTypeAnalysis",
        previous_executors: List[Executor] = None,
        dropna: bool = True,
        analyzing_columns=None,
    ):
        super().__init__(name, previous_executors)
        self.dropna = dropna
        self.analyzing_columns = set(analyzing_columns) if analyzing_columns else set()

    def type_by_unique(self, nunique: pd.Series, value: int, type_name: str) -> Dict:
        result = nunique[nunique == value].index
        if len(result) == 0:
            return {}
        self.analyzing_columns = self.analyzing_columns - set(result)
        return {c: type_name for c in result}

    def execute(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        self.analyzing_columns = (
            set(data.columns)
            if len(self.analyzing_columns) == 0
            else self.analyzing_columns
        )
        data = data[list(self.analyzing_columns)]
        nunique = data.nunique(dropna=self.dropna)
        dtypes = data.dtypes

        result = self.type_by_unique(nunique, 1, "const")
        result.update(self.type_by_unique(nunique, 2, "bool"))
        result.update(
            {
                c: "bool"
                for c in set(dtypes[dtypes == bool].index) & self.analyzing_columns
            }
        )
        self.analyzing_columns -= set(result.keys())

        category_columns = set(dtypes[dtypes == object].index) & self.analyzing_columns
        result.update({c: "categorical" for c in category_columns})
        self.analyzing_columns -= set(result.keys())

        result.update({c: "numerical" for c in self.analyzing_columns})
        self.analyzing_columns = set()
        return pd.DataFrame(
            [result, dtypes.to_dict()], index=["feature_type", "dtype"]
        ).T


class CorrelationAnalysis(Executor):
    def __init__(
        self,
        name: str = "CorrelationAnalysis",
        previous_executors: List[Executor] = None,
        method="pearson",
    ):
        super().__init__(name, previous_executors)
        self.method = method

    def calc_correlation(
        self, data: pd.DataFrame, correlation_function: Callable, without_category: bool
    ) -> pd.DataFrame:
        if without_category:
            result = pd.DataFrame(columns=data.columns, index=data.columns)
            for x in data.columns:
                if data[x].dtype != object:
                    for y in data.columns:
                        if data[y].dtype != object:
                            result.loc[x, y] = correlation_function(data[x], data[y])[0]
            return result

        return pd.DataFrame(
            [
                [correlation_function(data[x], data[y])[0] for x in data.columns]
                for y in data.columns
            ],
            columns=data.columns,
            index=data.columns,
        )

    def execute(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.method == "pearson":
            return self.calc_correlation(data, ss.pearsonr, without_category=True)
        if self.method == "kendall":
            return self.calc_correlation(data, ss.kendalltau, without_category=False)
        if self.method == "spearman":
            return self.calc_correlation(data, ss.spearmanr, without_category=False)

