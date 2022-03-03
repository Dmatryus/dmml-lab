from pathlib import Path

from data.db_load import SqlLoader


def test_init_by_path():
    loader = SqlLoader(path_to_config=Path(__file__).parent / 'test_data' / 'test_sql_config.json')
    print(loader)

