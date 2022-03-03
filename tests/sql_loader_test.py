import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from data.db_load import SqlLoader


def test_init_by_path():
    loader = SqlLoader(path_to_config=Path(__file__).parent / 'test_data' / 'test_sql_config.json')
    print(loader)


if __name__ == '__main__':
    test_init_by_path()
