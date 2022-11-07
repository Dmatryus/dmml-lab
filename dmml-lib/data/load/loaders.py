from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import pandas as pd

from ..data_types import TableData


class LoadFormats(Enum):
    PANDAS = "pandas"


class AbstractLoader(ABC):
    def __init__(self, data_source, load_format: LoadFormats = None):
        self.data_source = data_source
        self.load_format = load_format or LoadFormats.PANDAS

    @abstractmethod
    def load_data(self, **kwargs):
        pass


class FileLoader(AbstractLoader):
    data_path: Path

    def load_data(self, **kwargs):
        self.data_path = Path(self.data_source)
        if not self.data_path.exists():
            raise FileExistsError(f"Файл {self.data_path} не найден.")

        extension = self.data_path.suffix
        if self.load_format == LoadFormats.PANDAS:
            if extension == ".csv":
                return TableData(pd.read_csv(self.data_path, **kwargs))


print(Path(__file__).suffix)
