# Copyright © 2022 Dmity Bulychev. Contacts: dmatryus.sqrt49@yandex.ru
# License: http://opensource.org/licenses/MIT
import json
from enum import Enum
from pathlib import Path
from typing import Union


class Driver(Enum):
    MySQL = 'MySQL'


class SqlConfig:
    @staticmethod
    def init_from_file(path_to_config: Union[Path, str]):
        if isinstance(path_to_config, str):
            path_to_config = Path(path_to_config)
        with open(path_to_config, mode='r') as f:
            dict_config = json.load(f)

        return SqlConfig(**dict_config)

    def __init__(self, driver: Union[Driver, str], host: str, user: str, password: str):
        self.driver = driver if isinstance(driver, str) else driver.value
        self.host = host
        self.user = user
        self.password = password


class SqlLoader:
    def init_by_path(self, path_to_config: Union[Path, str]):
        self.config = SqlConfig.init_from_file(path_to_config)

    def __init__(self, config: SqlConfig = None, path_to_config: Union[Path, str] = None):
        self.config = config
        if path_to_config:
            self.init_by_path(path_to_config)
