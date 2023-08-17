from abc import abstractmethod
from typing import List, Dict


class Executor:
    def __init__(self, name, parent_pipeline: Pipeline = None):
        self.name = name
        self.parent_pipeline = parent_pipeline

    @abstractmethod
    def execute(self, input_stream: Dict) -> Dict:
        pass


class Pipeline:
    def __init__(self, pipes: List[Executor]):
        self.pipes = pipes
        for pipe in pipes:
            pipe.parent_pipeline = self

    def execute(self, data):
        outputs = {}
        for executor in self.pipes:
            outputs.update(executor.execute(data))
        return outputs
