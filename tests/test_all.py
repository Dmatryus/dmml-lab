import sys

from pathlib import Path

import random


import pandas as pd

import numpy as np


sys.path.append(str(Path(__file__).parent.parent))


import dmml_lib.data_analyze as da


def generate_data(rows=100, columns=10, random_state=None):
    # Create a DataFrame with random values

    np.random.seed(random_state)

    return pd.DataFrame(np.random.rand(rows, columns), columns=range(columns))


def test_na_analysis():
    # Generate data

    data = generate_data(150, 4, 7)

    # Set some values to None

    data[0] = None

    data.iloc[-7:, 1] = None

    data.iloc[7, 2] = None

    # Execute NA analysis

    result = da.NAAnalysis().execute(data)

    # Check full NA column

    assert result["absolute"].iloc[0] == len(data)

    assert result["percent"].iloc[0] == 100

    # Check partial NA column

    assert result["absolute"].iloc[1] == 7

    # Check one NA row

    assert result["absolute"].iloc[2] == 1

    # Check row without NA

    assert result["absolute"].iloc[3] == 0

    assert result["percent"].iloc[3] == 0

    # Check that the NA values are found

    expected_absolute = data.iloc[:, 0].isna().sum()

    expected_percent = expected_absolute / len(data) * 100

    assert (
        result["absolute"].iloc[0] == expected_absolute
    ), f"Expected {expected_absolute}, got {result['absolute'].iloc[0]}"

    assert (
        result["percent"].iloc[0] == expected_percent
    ), f"Expected {expected_percent}%, got {result['percent'].iloc[0]}"

    # Generate data

    data = generate_data(150, 6, 7)

    # Modify data

    data[0] = 7

    data[1] = 7

    data.iloc[3, 1] = None

    data[2] = data[2] > 0.5

    data[3] = data[2]

    data.iloc[3, 3] = None

    data[4] = data[4].astype(str)

    # Perform feature type analysis with dropna=True
    dropna_result = da.FeatureTypeAnalysis().execute(data)

    # Perform feature type analysis with dropna=False
    no_dropna_result = da.FeatureTypeAnalysis(dropna=False).execute(data)

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
    assert dropna_result["feature_type"].iloc[5] == "numerical"
    assert no_dropna_result["feature_type"].iloc[5] == "numerical"


def test_correlation_analysis():
    def corr_test(corr, category):
        assert round(corr.loc[0, 0], 2) == 1, f"Expected 1, got {corr.loc[0,0]}"
        assert (
            round(corr.loc[0, "anti"], 2) == -1
        ), f"Expected -1, got {corr.loc[0, 'anti']}"
        assert (
            abs(corr.loc[0, 1]) < 0.3
        ), f"Correlation between random numbers is not close to 0"
        assert (
            round(corr.loc[0, "multi"], 2) == 1
        ), f"Expected 1, got {corr.loc[0, 'multi']}"

        if category:
            assert (
                round(corr.loc[0, "same str"], 2) == 1
            ), f"Expected 1, got {corr.loc[0,0]}"
            assert abs(corr.loc[0, 1]) < 0.3, f"Correlation is not close to 0"

    data = generate_data(1000, 2, 7)

    data["anti"] = data.iloc[:, 0] * -1
    data["multi"] = data.iloc[:, 0] * 7
    data["same str"] = data.iloc[:, 0].astype(str) + "!"
    data["not same str"] = data.iloc[:, 1].astype(str) + "!"

    pearson = da.CorrelationAnalysis().execute(data)
    corr_test(pearson, False)
    assert pd.isna(
        (pearson.loc["same str", "same str"])
    ), f"Pearson correlation on categorical columns is not NaN"

    kendall = da.CorrelationAnalysis(method="kendall").execute(data)
    corr_test(kendall, True)

    spearman = da.CorrelationAnalysis(method="spearman").execute(data)
    print(spearman)
    corr_test(spearman, True)


def test_statbox_analysis():
    data = generate_data(100, 4, 7)
    default_stats = da.StatBoxAnalysis().execute(data)
    assert len(data.columns) == 4
