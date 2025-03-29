from pathlib import Path
from typing import List, Optional, Literal, Union, Dict, Any, get_args

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from pandas._typing import InterpolateOptions


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

    @staticmethod
    def default_index_set(data: pd.DataFrame) -> dict:
        """Generate default train, test, and validation indices.

        Args:
            data (pd.DataFrame): The input data for which to generate indices.

        Returns:
            dict: A dictionary containing three keys: 'train', 'test', and 'valid', each associated with a list of indices.
        """
        return {"train": list(data.index), "test": [], "valid": []}

    def __init__(
        self,
        data: Union[pd.DataFrame, Path, str],
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        read_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Dataset class.

        Parameters:
        data (Union[pd.DataFrame, Path, str]): The data source, which can be a pandas DataFrame,
            a file path (str or Path) to a CSV or Parquet file, or a string representing the file path.
        features (List[str]): A list of column names representing the features in the dataset.
        target (Optional[str]): The name of the target column in the dataset. Defaults to None if
            the dataset is used for unsupervised learning or feature engineering.
        read_kwargs (Dict[str, Any]): Additional keyword arguments to be passed to the file reading
            functions (pd.read_csv or pd.read_parquet). Defaults to an empty dictionary.

        Raises:
        ValueError: If the file format is not supported (only CSV and Parquet are supported).
        """
        data = Path(data) if isinstance(data, str) else data
        if isinstance(data, Path):
            if read_kwargs is None:
                read_kwargs = {}
            if data.suffix == ".csv":
                data = pd.read_csv(data, **read_kwargs)
            elif data.suffix == ".parquet":
                data = pd.read_parquet(data, **read_kwargs)
            else:
                raise ValueError(
                    "Unsupported file format. Only CSV and Parquet are supported."
                )
        self.data = data
        self.features = features if features else self.data.columns.tolist()
        self.target: Optional[str] = target
        self.features.remove(self.target) if self.target in self.features else None
        self.index_sets = self.default_index_set(self.data)

    @classmethod
    def load_california_housing_data(cls):
        """Load California housing data and initialize dataset attributes.

        This method fetches the California housing dataset, concatenates the features and target into a single DataFrame,
        sets the features and target attributes, and generates default train, test, and validation indices.

        Returns:
            Dataset: The initialized dataset object with the loaded data.
        """
        california_housing = fetch_california_housing(as_frame=True)
        data = pd.concat([california_housing.data, california_housing.target], axis=1)
        features = california_housing.feature_names
        target = california_housing.target.name
        return cls(data, features, target)

    def split_data(self, test_size: float = 0, valid_size: float = 0):
        """Split the dataset into train, test, and optionally validation sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            valid_size (float): Proportion of the dataset to include in the validation split.
        """
        if test_size == 0 and valid_size == 0:
            self.index_sets = self.default_index_set(self.data)
        self.index_sets["train"], self.index_sets["test"] = train_test_split(
            list(self.data.index), test_size=test_size
        )
        if valid_size > 0:
            self.index_sets["train"], self.index_sets["valid"] = train_test_split(
                self.index_sets["train"], test_size=valid_size
            )

    def get_index_set(
        self, index_set: Literal["train", "test", "valid", "all"] = "all"
    ) -> List:
        if index_set == "all":
            return list(self.data.index)
        else:
            return self.index_sets[index_set]

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
        indexes = self.get_index_set(indexes_set)
        if fields_set == "all":
            return self.data.loc[indexes, :]
        elif fields_set == "features":
            return self.data.loc[indexes, self.features]
        elif fields_set == "target":
            return None if self.target is None else self.data.loc[indexes, self.target]
        else:
            raise ValueError("Invalid fields_set value or target is None.")

    def set_data(
        self,
        new_data: pd.DataFrame,
        indexes_set: Literal["train", "test", "valid", "all"] = "all",
        fields_set: Literal["all", "features", "target"] = "all",
    ):
        indexes = self.get_index_set(indexes_set)
        if fields_set == "all":
            self.data.loc[indexes, :] = new_data
        if fields_set == "features":
            self.data.loc[indexes, self.features] = new_data
        if fields_set == "target":
            self.data.loc[indexes, self.target] = new_data.values

    def encode_categoricals(
        self,
        encoder,
        indexes_set: Literal["train", "test", "valid", "all"] = "all",
        reverse=False,
        **kwargs
    ):
        """Encode categorical features using the provided encoder."""
        if not reverse:
            categorical_features = self.get_data(
                indexes_set=indexes_set, fields_set="features"
            ).select_dtypes(include="object")
            encoded_features = encoder.fit_transform(categorical_features, **kwargs)
            self.data = self.data.drop(columns=categorical_features.columns)
            self.data = self.data.join(encoded_features)
        else:
            reversed_data = self.data.copy()
            encoder_value_mapping = {
                v["col"]: v["mapping"] for v in encoder.ordinal_encoder.mapping
            }

            for col in encoder.cols:
                mapping = encoder.mapping[col].to_dict()
                value_mapping = encoder_value_mapping[col]

                reverse_mapping = {v: k for k, v in mapping.items()}
                reverse_value_mapping = {v: k for k, v in value_mapping.items()}

                mapped_col = self.data[col].map(reverse_mapping)
                mapped_col = mapped_col.map(reverse_value_mapping)
                reversed_data[col] = mapped_col

            self.data = reversed_data
        return encoder

    def scale_features(
        self,
        scaler,
        indexes_set: Literal["train", "test", "valid", "all"] = "all",
        reverse=False,
        **kwargs
    ):
        if not reverse:
            scaled_data = scaler.fit_transform(
                self.get_data(indexes_set=indexes_set), **kwargs
            )
        else:
            scaled_data = scaler.inverse_transform(
                self.get_data(indexes_set=indexes_set), **kwargs
            )
        scaled_data = pd.DataFrame(
            scaled_data,
            columns=self.get_data(indexes_set=indexes_set).columns,
            index=(
                self.index_sets[indexes_set]
                if indexes_set != "all"
                else self.data.index
            ),
        )
        self.data = self.data.drop(index=scaled_data.index)
        self.data = pd.concat([self.data, scaled_data], axis=0).sort_index()
        return scaler

    def fillna(
        self,
        indexes_set: Literal["train", "test", "valid", "all"] = "all",
        fields_set: Literal["all", "features", "target"] = "all",
        method: Literal[
            "mean",
            "median",
            "mode",
            "linear",
            "time",
            "index",
            "values",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "barycentric",
            "polynomial",
            "krogh",
            "piecewise_polynomial",
            "spline",
            "pchip",
            "akima",
            "cubicspline",
            "from_derivatives",
            "bfill",
            "ffill",
        ] = "mean",
        value=None,
    ):
        if value is not None:
            self.set_data(
                self.get_data(indexes_set, fields_set).fillna(value),
                indexes_set,
                fields_set,
            )
        if method == "bfill":
            self.set_data(
                self.get_data(indexes_set, fields_set).bfill(), indexes_set, fields_set
            )
        elif method == "ffill":
            self.set_data(
                self.get_data(indexes_set, fields_set).ffill(), indexes_set, fields_set
            )
        elif method == "mean":
            self.set_data(
                self.get_data(indexes_set, fields_set).apply(
                    lambda x: x.fillna(x.mean())
                ),
                indexes_set,
                fields_set,
            )
        elif method == "median":
            self.set_data(
                self.get_data(indexes_set, fields_set).apply(
                    lambda x: x.fillna(x.median())
                ),
                indexes_set,
                fields_set,
            )
        if method == "mode":
            self.set_data(
                self.get_data(indexes_set, fields_set).apply(
                    lambda x: x.fillna(x.mode()[0])
                ),
                indexes_set,
                fields_set,
            )
        elif method in get_args(InterpolateOptions):
            self.set_data(
                self.get_data(indexes_set, fields_set).interpolate(method=method),
                indexes_set,
                fields_set,
            )

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def _repr_html_(self):
        return self.data._repr_html_()

    def __len__(self):
        return len(self.data)
