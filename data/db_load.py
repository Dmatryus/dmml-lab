# Copyright © 2022 Dmity Bulychev. Contacts: dmatryus.sqrt49@yandex.ru
# License: http://opensource.org/licenses/MIT

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Tuple, List
import pandas as pd

import pymysql

from data.dict_driver import value_key_convert, DeepPath


class JsonStoredConfig(ABC):
    @staticmethod
    @abstractmethod
    def init_from_dict(dict_config: Dict):
        pass

    def init_from_file(self, path_to_config: Union[Path, str]):
        if isinstance(path_to_config, str):
            path_to_config = Path(path_to_config)
        with open(path_to_config, mode="r") as f:
            dict_config = json.load(f)

        return self.__class__.init_from_dict(dict_config)


class DriverConfig(JsonStoredConfig):
    @staticmethod
    def init_from_dict(dict_config: Dict):
        return DriverConfig(**dict_config)

    SUPPORTED_DRIVERS_LIST = ["mysql"]

    def __init__(self, driver: str, host: str, user: str, password: str):
        self.driver = driver.lower()
        self.host = host
        self.user = user
        self.password = password


class QueryConfig(JsonStoredConfig):
    @staticmethod
    def init_from_dict(dict_config: Dict):
        return QueryConfig(**dict_config)

    def __init__(self, fields_mapping: Dict, tables: Dict, select_db: str = None):
        self.fields_mapping = fields_mapping
        self.tables = tables
        self.select_db = select_db


class SqlConfig(JsonStoredConfig):
    driver_config: DriverConfig
    query_config: QueryConfig

    @staticmethod
    def init_from_dict(dict_config: Dict):
        return SqlConfig(
            DriverConfig.init_from_dict(dict_config["driver_config"]),
            QueryConfig.init_from_dict(dict_config["query_config"])
            if "query_config" in dict_config
            else None,
        )

    def __init__(
            self, driver_config: DriverConfig = None, query_config: QueryConfig = None
    ):
        self.driver_config = driver_config
        self.query_config = query_config


class SqlLoader:
    config: SqlConfig

    def __init__(
            self, config: SqlConfig = None, path_to_config: Union[Path, str] = None
    ):
        self.config = config
        if not self.config and path_to_config:
            self.config = SqlConfig().init_from_file(path_to_config)

    def load_by_query(self, query: str, db=None, learn=False):
        data = None
        db = db or self.config.query_config.select_db
        if self.config.driver_config.driver == "mysql":
            connect = pymysql.connect(
                host=self.config.driver_config.host,
                user=self.config.driver_config.user,
                passwd=self.config.driver_config.password,
                use_unicode=True,
                charset="utf8",
            )
            if db:
                connect.select_db(db)
            data = pd.read_sql(query, connect)
            connect.close()
        return data

    def build_query(self, fields: List[str] = None, conditions: List[str] = None):
        def build_fields():
            if not fields:
                return "*"
            else:
                return "\n\t".join(
                    [
                        f"{self.config.query_config.fields_mapping[f]} {f}"
                        for f in fields
                    ]
                )

        def build_tables():
            table_shorts = value_key_convert(
                self.config.query_config.tables, DeepPath(("short",)), flatten=True
            )
            tables_in_query = {table_shorts[f[: f.find(".")]] for f in fields}

            result = "from "
            for i, table in enumerate(tables_in_query):
                short_path = DeepPath((table, "short"))
                if i == 0:
                    result += f"{table} {short_path.get_from_dict(self.config.query_config.tables)}"
                else:
                    for t in tables_in_query[:i]:
                        if table in t['joines']:
                            result += t['joines'][table]
            return result

        query = "select " + build_fields() + '\n' + build_tables()
