from typing import List, Dict

import pandas as pd

try:
    from .pipeline import Pipeline, Executor
except:
    from pipeline import Pipeline, Executor


class NAAnalysis(Executor):
    def __init__(self, parent_pipeline: Pipeline = None):
        super().__init__("NAAnalysis", parent_pipeline)

    def execute(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return pd.DataFrame(
            [data.isna().sum(), data.isna().sum() / len(data) * 100],
            index=["absolute", "percent"],
        ).T


class FeatureTypeAnalysis(Executor):
    def __init__(self, parent_pipeline: Pipeline = None, dropna: bool = True, analyzing_columns=None):
        super().__init__("TypeAnalysis", parent_pipeline)
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
    def __init__(self, parent_pipeline: Pipeline = None, method="pearson"):
        super().__init__("CorrelationAnalysis", parent_pipeline)
        self.method = method

    def execute(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return pd.DataFrame(data.corr(self.method), index=data.columns, columns=data.columns)