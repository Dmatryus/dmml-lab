from abc import abstractmethod
from typing import List, Dict

import pandas as pd


class Executor:
    def __init__(self, name, previous_executors: List = None):
        self.name = name
        self.previous_executors = previous_executors or []

    @abstractmethod
    def execute(self, input_stream: Dict) -> Dict:
        pass


class Pipeline:
    def __init__(self, pipes: List[Executor]):
        self.pipes = pipes
        for pipe in pipes:
            pipe.parent_pipeline = self

    def execute(self, data: pd.DataFrame) -> Dict:
        outputs = {}
        for executor in self.pipes:
            outputs.update(executor.execute(data))
        return outputs
