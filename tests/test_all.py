import sys
from pathlib import Path
import random

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from dmml_lib.data_analyze import NAAnalysis


def generate_data(rows=100, columns=10, na_values=10, random_state=None):
    # Create a DataFrame with random values
    np.random.seed(random_state)
    data = pd.DataFrame(np.random.rand(rows, columns), columns=range(columns))

    # Set some values to NaN

    for _ in range(na_values):
        data.loc[random.choice(data.index), random.choice(data.columns)] = None

    return data


def testNAAnalysis():
    n_na = 20
    data = generate_data(150, 4, n_na, 7)
    result = NAAnalysis().fill({"data": data})

    print(result)

    # Check that the NA values are found
    assert (
        result["NAAnalysis"]["Absolute"].iloc[0] == data.iloc[:, 0].isna().sum()
    ), f"Expected {data.iloc[:, 0].isna().sum()}, got {result['NAAnalysis']['Absolute'].iloc[0]}"
    
    # Check that the percentage of NA values is correct
    assert (
        result["NAAnalysis"]["Percent"].iloc[0]
        == data.iloc[:, 0].isna().sum() / len(data) * 100
    ), f"Expected {data.iloc[:, 0].isna().sum() / len(data) * 100}%, got {result['NAAnalysis']['Percent'].iloc[0]}"


testNAAnalysis()
