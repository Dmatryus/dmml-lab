from abc import ABC, abstractmethod
from typing import Set, Iterable, Dict, Union, List, Callable

import pandas as pd
from tqdm.auto import tqdm


class CheckMetric(ABC):
    def __init__(self, name: str = None):
        self.name = name or self.__name__

    @abstractmethod
    def check(self, values) -> bool:
        pass


class CheckDf(CheckMetric):
    def __init__(
        self,
        name: str = None,
        columns: Iterable[Union[str, List[str]]] = None,
        func: Callable = None,
    ):
        super().__init__(name)
        self.columns = columns
        self.func = func

    def check_func(self, values: pd.DataFrame, column: Union[str, List[str]]):
        return self.func is None

    def check(self, values: pd.DataFrame) -> bool:
        columns = self.columns or values.columns
        if self.func is not None:
            return all(self.func(values, c) for c in columns)
        else:
            return all(self.check_func(values, c) for c in columns)


class CheckZeros(CheckDf):
    DEFAULT_ZEROS = {"", 0, " "}

    def __init__(
        self, name: str = None, columns: Iterable[str] = None, zero_values: Set = None
    ):
        self.zero_values = zero_values or self.DEFAULT_ZEROS
        super().__init__(name, columns)

    def check_func(self, values: pd.DataFrame, column: str) -> bool:
        return all(z not in values[column] for z in self.zero_values)


class CheckNA(CheckDf):
    def __init__(self, name: str = None, columns: Iterable[str] = None):
        super().__init__(name, columns)

    def check_func(self, values: pd.DataFrame, column: str) -> bool:
        return len(values[values[column]].isna()) == 0


class CheckUniqueness(CheckDf):
    def __init__(self, name: str = None, columns: Union[str, List[str]] = None):
        super().__init__(name, columns)

    def check_func(self, values: pd.DataFrame, column: Union[str, List[str]]) -> bool:
        return len(values) == len(values.drop_duplicates(subset=column))


class CheckRegexp(CheckDf):
    def __init__(
        self, name: str = None, columns: Iterable[str] = None, regexp: str = ""
    ):
        self.regexp = regexp
        super().__init__(name, columns)

    def check_func(self, values: pd.DataFrame, column: str) -> bool:
        return len(values) == len(values[values[column].str.contains(self.regexp)])


class FixMethod(ABC):
    @abstractmethod
    def fix(self, values):
        pass


class FixDf(FixMethod):
    def __init__(
        self, columns: Iterable[Union[str, List[str]]] = None, func: Callable = None
    ):
        self.columns = columns
        self.func = func

    def fix_func(
        self, values: pd.DataFrame, columns: Union[str, List[str]]
    ) -> pd.DataFrame:
        return self.func(values, columns) if self.func is not None else values

    def fix(self, values: pd.DataFrame) -> pd.DataFrame:
        columns = self.columns or values.columns
        return self.func(values, columns)


class FixByDropNA(FixDf):
    def __init__(self, columns: Iterable[Union[str, List[str]]] = None):
        super().__init__(columns)

    def fix_func(
        self, values: pd.DataFrame, columns: Union[str, List[str]]
    ) -> pd.DataFrame:
        return values.dropna(subset=columns)


class FixByReplace(FixDf):
    def __init__(
        self, replace_map: Dict, columns: Iterable[Union[str, List[str]]] = None
    ):
        self.replace_map = replace_map
        super().__init__(columns)

    def fix_func(
        self, values: pd.DataFrame, columns: Union[str, List[str]]
    ) -> pd.DataFrame:
        for c in columns:
            for k, v in self.replace_map.items():
                values[c] = values[c].replace(k, v)
        return values


class FixByDeduplicate(FixDf):
    def __init__(self, columns: Iterable[Union[str, List[str]]] = None):
        super().__init__(columns)

    def fix_func(
        self, values: pd.DataFrame, columns: Union[str, List[str]]
    ) -> pd.DataFrame:
        return values.drop_duplicates(subset=columns)


class DataQA:
    check_fix_map: Dict[CheckMetric, FixMethod]

    def __init__(self, check_set: Dict[CheckMetric, FixMethod], pbar: bool = True):
        self.check_fix_map = check_set
        self.report = {}
        self.pbar = pbar

    def process(self, data):
        progress = tqdm(self.check_fix_map.items(), disable=not self.pbar)
        for check, fix in progress:
            progress.set_description(check.name)
            self.report[check.name] = check.check(data)
            if self.report[check.name] and fix is not None:
                data = fix.fix(data)
        return data
