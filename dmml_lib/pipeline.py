from abc import abstractmethod
from typing import List, Dict

class Pipe:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fill(self, input_stream: Dict) -> Dict:
        pass


class Pipeline:
    def __init__(self, pipes: List[Pipe]):
        self.pipes = pipes

    def fill(self, data):
        outputs = {}
        for pipe in self.pipes:
            outputs.update(pipe.fill(data))
        return outputs
