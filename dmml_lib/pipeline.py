from abc import abstractmethod
from typing import List, Dict

class Pipe:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def execute(self, input_stream: Dict) -> Dict:
        pass


class Pipeline:
    def __init__(self, pipes: List[Pipe]):
        self.pipes = pipes

    def execute(self, data):
        outputs = {}
        for pipe in self.pipes:
            outputs.update(pipe.execute(data))
        return outputs
