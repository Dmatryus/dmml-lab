from typing import List, Dict

import pandas as pd

try:
    from .pipeline import Pipeline, Pipe
except:
    from pipeline import Pipeline, Pipe


class NAAnalysis(Pipe):
    def __init__(self):
        super().__init__("NAAnalysis")

    def fill(self, input_stream: Dict) -> Dict[str, pd.DataFrame]:
        data = input_stream["data"]
        return {
            self.name: pd.DataFrame(
                [data.isna().sum(), data.isna().sum()/len(data)*100], index=["Absolute", "Percent"]
            ).T
        }


class TypeAnalysis(Pipe):
    def __init__(self, dropna: bool = True, analyzing_columns=None):
        super().__init__("TypeAnalysis")
        self.dropna = dropna
        self.analyzing_columns = set(analyzing_columns) if analyzing_columns else set()

    def type_by_unique(self, nunique: pd.Series, value: int, type_name: str) -> Dict:
        result = nunique[nunique == 1].index
        self.analyzing_columns = self.analyzing_columns - set(result)
        return {c: type_name for c in result}

    def fill(self, input_stream: Dict) -> Dict[str, pd.DataFrame]:
        data = input_stream["data"]
        self.analyzing_columns = data.columns if len(self.analyzing_columns) == 0 else self.analyzing_columns
        nunique = data.nunique(dropna=self.dropna)
        dtypes = data.dtypes

        result = self.type_by_unique(nunique, 1, "const")
        result.update(self.type_by_unique(nunique, 2, "bool")).update(
            {
                c: "bool"
                for c in set(dtypes[dtypes == bool].index) - self.analyzing_columns
            }
        )

        # TODO: add other types
        return result

