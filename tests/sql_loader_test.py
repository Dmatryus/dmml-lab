import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from data.db_load import SqlLoader


def init_by_path():
    loader = SqlLoader(
        path_to_config=Path(__file__).parent / "test_data" / "test_sql_config.json"
    )
    print(loader)
    return loader


def query_build(loader: SqlLoader):
    print(loader.build_query(limit=10))
    print(loader.build_query(['city_title'], conditions=['city_id < 100']))
    print(
        loader.build_query(
            [
                "id",
                "decision",
                "total_space",
                "report_price",
                "price",
                "expert_price",
                "deviation",
                "city_id",
                "created_on",
            ],
            conditions=['created_on > "2022-02-25"', "city_id = 1"],
        )
    )


if __name__ == "__main__":
    loader = init_by_path()
    query_build(loader)
