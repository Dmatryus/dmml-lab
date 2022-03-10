import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from data.dict_driver import value_key_convert, DeepPath


def test_value_key_convert():
    with open(
        Path(__file__).parent / "test_data" / "test_sql_config.json", mode="r"
    ) as f:
        dict_ = json.load(f)
    print(
        value_key_convert(
            DeepPath(("query_config", "tables")).get_from_dict(dict_),
            DeepPath(("short",)),
            flatten=True,
        )
    )


if __name__ == "__main__":
    test_value_key_convert()
