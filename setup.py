from setuptools import setup, find_packages
from os.path import join, dirname

try:
    with open(join(dirname(__file__), "readme.md")) as fh:
        long_description = fh.read()
except:
    long_description = (
        "Convinient statistical description of dataframes and time series."
    )

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
    keywords=["STATICS", "TIME_SERIES"],
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "pymysql"],
)
