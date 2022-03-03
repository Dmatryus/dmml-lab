# Copyright © 2022 Dmity Bulychev. Contacts: dmatryus.sqrt49@yandex.ru
# License: http://opensource.org/licenses/MIT
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, Dict, Tuple


class JsonStoredConfig(ABC):
    @staticmethod
    @abstractmethod
    def init_from_dict(dict_config: Dict):
        pass

    @staticmethod
    def init_from_file(path_to_config: Union[Path, str]):
        if isinstance(path_to_config, str):
            path_to_config = Path(path_to_config)
        with open(path_to_config, mode='r') as f:
            dict_config = json.load(f)

        return JsonStoredConfig.init_from_dict(dict_config)


class DriverConfig(JsonStoredConfig):
    @staticmethod
    def init_from_dict(dict_config: Dict):
        return DriverConfig(**dict_config)

    SUPPORTED_DRIVERS_LIST = ['MySQL']

    def __init__(self, driver: str, host: str, user: str, password: str):
        self.driver = driver
        self.host = host
        self.user = user
        self.password = password


class QueryConfig(JsonStoredConfig):
    @staticmethod
    def init_from_dict(dict_config: Dict):
        return QueryConfig(**dict_config)

    def __init__(self, fields_mapping: Dict, tables: Tuple[Dict], select_db: str = None):
        self.fields_mapping = fields_mapping
        self.tables = tables
        self.select_db = select_db


class SqlConfig(JsonStoredConfig):
    @staticmethod
    def init_from_dict(dict_config: Dict):
        return SqlConfig(DriverConfig.init_from_dict(dict_config['driver_config']), QueryConfig.init_from_dict(
            dict_config['query_config']) if 'query_config' in dict_config else None)

    def __init__(self, driver_config: DriverConfig, query_config: QueryConfig = None):
        self.driver_config = driver_config
        self.query_config = query_config


class SqlLoader:
    def init_by_path(self, path_to_config: Union[Path, str]):
        self.config = SqlConfig.init_from_file(path_to_config)

    def __init__(self, config: SqlConfig = None, path_to_config: Union[Path, str] = None):
        self.config = config
        if path_to_config:
            self.init_by_path(path_to_config)
