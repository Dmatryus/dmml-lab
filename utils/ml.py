import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance

from .dataset import ModelData


def etrp_importance(data: ModelData) -> pd.DataFrame:
    """
    Calculate permutation feature importance using Extra Trees Regressor.

    Parameters:
    data (ModelData): An object containing training and test datasets.

    Returns:
    pd.DataFrame: A DataFrame with feature importances, including mean and standard deviation.
    """
    # Initialize and train the model
    model = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1)
    model.fit(data.get_data("train", "features"), data.get_data("train", "target"))
    result = permutation_importance(
        model,
        data.get_data("test", "features"),
        data.get_data("test", "target"),
        n_repeats=30,
    )
    return pd.DataFrame(
        data={
            "importances mean": result.importances_mean,
            "importances std": result.importances_std,
        },
        index=data.features,
    ).sort_values(by="importances mean", ascending=False)


def feature_importance_plot(importances: pd.DataFrame):
    """
    Plot permutation feature importances.

    Parameters:
    importances (pd.DataFrame): A DataFrame containing feature importances with mean and standard deviation.
    """
    importances = importances.sort_values(by="importances mean", ascending=False)
    diff = importances["importances mean"].sort_values(ascending=True).diff()
    plt.bar(
        importances.index,
        importances["importances mean"],
        yerr=importances["importances std"],
        alpha=0.7,
    )
    plt.plot(diff.index, diff, color="black", alpha=0.7)
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Permutation Importances of Features")
    plt.show()
