from typing import List, Optional, Literal

import pandas as pd
from sklearn.model_selection import train_test_split


class ModelData:
    """Class for managing dataset splits and feature/target extraction.

    Args:
        data (pd.DataFrame): The dataset.
        features (List[str]): List of feature column names.
        target (Optional[str]): Name of the target column.

    Attributes:
        data (pd.DataFrame): The dataset.
        target (Optional[str]): Name of the target column.
        features (List[str]): List of feature column names.
        index_sets (Dict[str, List]): Dictionary storing train, test, and validation indices.
    """

    def __init__(
        self, data: pd.DataFrame, features: List[str], target: Optional[str] = None
    ):
        self.data = data
        self.features = features
        self.target = target
        self.index_sets = {"train": list(data.index), "test": [], "valid": []}

    def split_data(self, test_size: float, valid_size: float = 0):
        """Split the dataset into train, test, and optionally validation sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            valid_size (float): Proportion of the dataset to include in the validation split.
        """
        self.index_sets["train"], self.index_sets["test"] = train_test_split(
            list(self.data.index), test_size=test_size
        )
        if valid_size > 0:
            self.index_sets["train"], self.index_sets["valid"] = train_test_split(
                self.index_sets["train"], test_size=valid_size
            )

    def get_data(
        self,
        indexes_set: Literal["train", "test", "valid", "all"] = "all",
        fields_set: Literal["all", "features", "target"] = "all",
    ):
        """Retrieve data based on specified indexes and fields.

        Args:
            indexes_set (Literal["train", "test", "valid", "all"]): The set of indexes to retrieve.
            fields_set (Literal["all", "features", "target"]): The set of fields to retrieve.

        Returns:
            pd.DataFrame or pd.Series: The requested data subset.

        Raises:
            ValueError: If an invalid fields_set value is provided or if target is None.
        """
        if indexes_set == "all":
            indexes = list(self.data.index)
        else:
            indexes = self.index_sets[indexes_set]

        if fields_set == "all":
            return self.data.loc[indexes, :]
        elif fields_set == "features":
            return self.data.loc[indexes, self.features]
        elif fields_set == "target":
            return None if self.target is None else self.data.loc[indexes, self.target]
        else:
            raise ValueError("Invalid fields_set value or target is None.")
