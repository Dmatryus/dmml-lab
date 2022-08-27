import pathlib
from setuptools import setup, find_packages
from os.path import join, dirname

try:
    long_description = pathlib.Path(join(dirname(__file__), "../readme.md")).read_text()

except:
    long_description = "A library for creating machine learning pipelines."

setup(
    name="dmml-lab",
    packages=find_packages(),
    version="0.0a0",
    license="MIT",
    description="",
    author="dmatryus",
    author_email="dmatryus.sqrt49@yandex.ru",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/dmatryus.sqrt49/dmml-lab",
    keywords=["STATICS", "TIME_SERIES", "MACHINE_LEARNING"],
    install_requires=["scipy", "pandas", "matplotlib", "pymysql", "tqdm"],
)
