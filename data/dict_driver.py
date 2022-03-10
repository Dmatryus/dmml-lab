from functools import reduce
from typing import Dict, Tuple, Iterable


class DeepPath:
    path: Tuple

    def __init__(self, path: Tuple = None):
        self.path = path

    def init_from_str(self, str_: str, sep: str = "."):
        self.path = tuple(str_.split(sep))

    def get_from_dict(self, dict_: Dict, safe: bool = False, default=None):
        if not self.path:
            if safe:
                return default
            else:
                raise ValueError(f"Path is not valid {self.path}")

        if not safe:
            return reduce(lambda x, y: x[y], self.path, dict_)
        try:
            return reduce(lambda x, y: x.get(y, default), self.path, dict_)
        except Exception as e:
            # Может возникнуть, если промежуточное звено не является dict
            return default


def value_key_convert(
    dict_: Dict, value_path: DeepPath = None, flatten: bool = False, **kwargs
):
    result = {}
    for k, v in dict_.items():
        value = value_path.get_from_dict(v, **kwargs) if value_path else v
        if value in result:
            result[value].append(k)
        else:
            result[value] = [k]
    if flatten:
        result = {k: v[0] for k, v in result.items()}
    return result
