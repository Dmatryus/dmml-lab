import sys
from pathlib import Path
import random

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from dmml_lib.data_analyze import NAAnalysis, FeatureTypeAnalysis


def generate_data(rows=100, columns=10, random_state=None):
    # Create a DataFrame with random values
    np.random.seed(random_state)
    data = pd.DataFrame(np.random.rand(rows, columns), columns=range(columns))

    return data


def testNAAnalysis():
    data = generate_data(150, 4, 7)
    data[0] = None
    data.iloc[-7:, 1] = None
    data.iloc[7, 2] = None

    result = NAAnalysis().execute({"data": data})

    # Check full na column
    assert result["absolute"].iloc[0] == len(data)
    assert result["percent"].iloc[0] == 100

    # Check partial na column
    assert result["absolute"].iloc[1] == 7

    # Check one na row
    assert result["absolute"].iloc[2] == 1

    # Check row without na
    assert result["absolute"].iloc[3] == 0
    assert result["percent"].iloc[3] == 0

    # Check that the NA values are found
    assert (
        result["absolute"].iloc[0] == data.iloc[:, 0].isna().sum()
    ), f"Expected {data.iloc[:, 0].isna().sum()}, got {result['absolute'].iloc[0]}"
    assert (
        result["percent"].iloc[0] == data.iloc[:, 0].isna().sum() / len(data) * 100
    ), f"Expected {data.iloc[:, 0].isna().sum() / len(data) * 100}%, got {result['percent'].iloc[0]}"


def testFeatureTypeAnalysis():
    data = generate_data(150, 6, 7)
    data[0] = 7
    data[1] = 7
    data.iloc[3, 1] = None
    data[2] = data[2] > 0.5
    data[3] = data[2]
    data.iloc[3, 3] = None
    data[4] = data[4].astype(str)

    dropna_fta = FeatureTypeAnalysis()
    dropna_result = dropna_fta.execute({"data": data})

    print(dropna_result)

    no_dropna_fta = FeatureTypeAnalysis(dropna=False)
    no_dropna_result = no_dropna_fta.execute({"data": data})
    print(no_dropna_result)

    # Check const
    assert (
        dropna_result["feature_type"].iloc[0] == "const"
    ), f"Expected 'const', got {dropna_result['feature_type'].iloc[0]}"
    assert (
        dropna_result["feature_type"].iloc[1] == "const"
    ), f"Expected 'const', got {dropna_result['feature_type'].iloc[1]}"
    assert (
        no_dropna_result["feature_type"].iloc[0] == "const"
    ), f"Expected 'const', got {no_dropna_result['feature_type'].iloc[0]}"
    assert (
        no_dropna_result["feature_type"].iloc[1] == "bool"
    ), f"Expected 'bool', got {no_dropna_result['feature_type'].iloc[1]}"

    # Check bool
    assert (
        dropna_result["feature_type"].iloc[2] == "bool",
        f"Expected 'bool', got {dropna_result['feature_type'].iloc[2]}",
    )
    assert (
        dropna_result["feature_type"].iloc[3] == "bool",
        f"Expected 'bool', got {dropna_result['feature_type'].iloc[3]}",
    )
    assert (
        no_dropna_result["feature_type"].iloc[2] == "bool",
        f"Expected 'bool', got {no_dropna_result['feature_type'].iloc[2]}",
    )
    assert (
        no_dropna_result["feature_type"].iloc[3] != "bool",
        f"Expected not 'bool', got {no_dropna_result['feature_type'].iloc[3]}",
    )

    # Check categorical
    assert (
        dropna_result["feature_type"].iloc[3] != "categorical",
        f"Expected not 'categorical', got {dropna_result['feature_type'].iloc[3]}",
    )
    assert (
        dropna_result["feature_type"].iloc[4] == "categorical",
        f"Expected 'categorical', got {dropna_result['feature_type'].iloc[4]}",
    )
    assert (
        no_dropna_result["feature_type"].iloc[3] == "categorical",
        f"Expected 'categorical', got {no_dropna_result['feature_type'].iloc[3]}",
    )
    assert (
        no_dropna_result["feature_type"].iloc[4] == "categorical",
        f"Expected 'categorical', got {no_dropna_result['feature_type'].iloc[4]}",
    )

    # Check numerical
    assert (
        dropna_result["feature_type"].iloc[5] == "numerical"
    )
    assert (
        no_dropna_result["feature_type"].iloc[5] == "numerical"
    )

    # result = FeatureTypeAnalysis().execute({"data": data})

    # print(result)

    # # Check that the NA values are found
    # assert (
    #     result["TypeAnalysis"]["absolute"].iloc[0] == data.iloc[:, 0].isna().sum()
    # ), f"Expected {data.iloc[:, 0].isna().sum()}, got {result['TypeAnalysis']['absolute'].iloc[0]}"

    # # Check that the percentage of NA values is correct
    # assert (
    #     result["TypeAnalysis"]["percent"].iloc[0]
    #     == data.iloc[:, 0].isna().sum() / len(data) * 100
    # ), f"Expected {data.iloc[:, 0].isna().sum() / len(data) * 100}%, got {result['TypeAnalysis']['percent'].iloc[0]}"


testFeatureTypeAnalysis()
