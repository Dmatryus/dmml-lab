from typing import Optional, Dict, Any, Literal
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import norm, stats


class Metric(ABC):
    def __init__(
        self,
        target_field: str,
        mark_field: Optional[str] = None,
        mark_type: Literal[None, "filter", "strat"] = None,
    ):
        self.target_field = target_field
        self.mark_field = mark_field
        self.mark_type = mark_type

    @abstractmethod
    def calc_function(
        self, data: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        raise NotImplementedError

    def strat_calc_function(
        self, data: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            dict(list(data.groupby(self.mark_field).apply(self.calc_function)))
        )

    def calc(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.mark_type == "filter" and self.mark_field is not None:
            result = self.calc_function(data)
            result["filtered"] = self.calc_function(data[data[self.mark_field] == 1])[
                "metric"
            ]
            result["diff"] = result["metric"] - result["filtered"]
            result["relative_diff"] = result["diff"] / result["metric"]
            return pd.DataFrame([result], index=[self.__class__.__name__])
        if self.mark_type == "strat" and self.mark_field is not None:
            return self.strat_calc_function(data)
        return pd.DataFrame(self.calc_function(data))


class Std(Metric):
    def calc_function(
        self, data: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        return {"metric": data[self.target_field].std()}

    def strat_calc_function(
        self, data: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        # Calculate total mean
        total_mean = data[self.target_field].mean()

        # Group by strata and calculate means and variances within each stratum
        grouped = data.groupby(self.mark_field)
        dict_result = {
            "mean": grouped[self.target_field].mean(),
            "var": grouped[self.target_field].var(ddof=1),
            "size": grouped.size(),
        }

        result = pd.DataFrame(dict_result)

        result["metric"] = sum(
            (
                dict_result["var"][stratum]
                + (dict_result["mean"][stratum] - total_mean) ** 2
            )
            * (dict_result["size"][stratum] / len(data))
            for stratum in dict_result["mean"].index
        )

        return pd.DataFrame(dict_result)


class DataLoss(Metric):
    def calc_function(
        self, df: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        """
        Calculate the data loss metric.
        """
        return {"metric": len(df)}


class Mean(Metric):
    def calc_function(
        self, df: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        """
        Calculate the mean metric.
        """
        return {"metric": df[self.target_field].mean()}


class TradeoffMetric(Metric, ABC):
    _support_metric_classes: list[type]

    def __init__(
        self,
        target_field: str,
        mark_field: Optional[str] = None,
        mark_type: Literal[None, "filter", "strat"] = None,
        tradeoff_factor: float = 1,
    ):
        super().__init__(target_field, mark_field, mark_type)
        if mark_type == "strat" or mark_type is None:
            raise ValueError("TradeoffMetric only supports mark_type='filter'")
        self.tradeoff_factor = tradeoff_factor
        self._support_metrics = [
            c(target_field, mark_field, "filter") for c in self._support_metric_classes
        ]

    @abstractmethod
    def calc_function(
        self, df: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        raise NotImplementedError

    def calc(self, data: pd.DataFrame) -> pd.DataFrame:
        additional_metrics = pd.concat([c.calc(data) for c in self._support_metrics])
        additional_metrics.loc[self.__class__.__name__, :] = [
            self.calc_function(data, additional_metrics)["metric"],
            None,
            None,
            None,
        ]

        return additional_metrics


class StdDataLossTradeoff(TradeoffMetric):
    _support_metric_classes = [Std, DataLoss]

    def calc_function(
        self, df: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        return {
            "metric": additional_metrics.loc["Std", "relative_diff"]
            * np.power(
                1 - additional_metrics.loc["DataLoss", "relative_diff"],
                self.tradeoff_factor,
            )
        }


class StdMeanTradeoff(TradeoffMetric):
    _support_metric_classes = [Std, Mean]

    def calc_function(
        self, df: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        return {
            "metric": additional_metrics.loc["Std", "relative_diff"]
            * np.power(
                1 - additional_metrics.loc["Mean", "relative_diff"],
                self.tradeoff_factor,
            )
        }


def group_std(
    data: pd.DataFrame,
    target: str,
    filter_field: Optional[str],
    strat_field: Optional[str],
) -> pd.DataFrame:
    """
    Calculate the standard deviation for each group in the data.

    Parameters:
    data (pd.DataFrame): The input data containing the target and group columns.
    target (str): The column name of the target variable.
    group (str): The column name of the group variable.

    Returns:
    pd.DataFrame: A DataFrame with each group's size and variance, along with the total size and variance.
    """
    groups_var = data.groupby(group).agg(
        group_size=(target, "count"), var=(target, "var")
    )
    groups_var.loc["total"] = [len(data), data[target].var()]
    return groups_var


def strat_std(
    data: pd.DataFrame, target: str, group: str, strat: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate the stratified standard deviation for each group in the data.

    Parameters:
    data (pd.DataFrame): The input data containing the target, group, and stratification columns.
    target (str): The column name of the target variable.
    group (str): The column name of the group variable.
    strat (Optional[str], optional): The column name of the stratification variable. Defaults to None.

    Returns:
    pd.DataFrame: A DataFrame with each group's size and stratified variance, along with the total size and variance.
    """
    if not strat:
        return group_std(data, target, group)

    total_var = 0

    groups_var = data.groupby([group, strat]).agg(group_size=(target, "count"))
    groups_var[f"{target}_strat_var"] = 0

    for strat, strat_data in data.groupby(strat):
        total_strat_val = strat_data[target].var() * len(strat_data) / len(data)
        total_var += total_strat_val
        for group, group_strated_data in strat_data.groupby(group):
            group_strat_var = (
                group_strated_data[target].var()
                * len(group_strated_data)
                / len(groups_var.loc[group, "group_size"])
            )
            groups_var.loc[group, f"{target}_strat_var"] += (
                groups_var.loc[group, f"{target}_strat_var"] + group_strat_var
            )

    groups_var.loc["total"] = [len(data), total_var]
    return groups_var


def mde(
    data: pd.DataFrame,
    target: str,
    group: str,
    significance: float = 0.05,
    power: float = 0.8,
) -> float:
    """
    Calculate the Minimum Detectable Effect (MDE) for a given group in the data.

    Parameters:
    data (pd.DataFrame): The input data containing the target and group columns.
    target (str): The column name of the target variable.
    group (str): The column name of the group variable.
    significance (float, optional): The significance level for the MDE calculation. Defaults to 0.05.
    power (float, optional): The power for the MDE calculation. Defaults to 0.8.

    Returns:
    float: The calculated MDE value.
    """
    grouped_data = sorted(list(data.groupby(group)), key=lambda x: x[0])
    control_data = grouped_data[0][1][target]
    test_data = grouped_data[1][1][target]

    m = norm.ppf(1 - significance / 2) + norm.ppf(power)

    n_control, n_test = len(control_data), len(test_data)

    var_control, var_test = control_data.var(ddof=1), test_data.var(ddof=1)
    return m * np.sqrt(var_test / n_test + var_control / n_control)


def strat_mde(
    data: pd.DataFrame,
    target: str,
    group: str,
    strat: Optional[str] = None,
    significance: float = 0.05,
    power: float = 0.8,
) -> float:
    """
    Calculate the stratified Minimum Detectable Effect (MDE) for a given group in the data.

    Parameters:
    data (pd.DataFrame): The input data containing the target, group, and stratification columns.
    target (str): The column name of the target variable.
    group (str): The column name of the group variable.
    strat (Optional[str], optional): The column name of the stratification variable. Defaults to None.
    significance (float, optional): The significance level for the MDE calculation. Defaults to 0.05.
    power (float, optional): The power for the MDE calculation. Defaults to 0.8.

    Returns:
    float: The calculated stratified MDE value.
    """
    if not strat:
        return mde(data, target, group, significance, power)

    m = norm.ppf(1 - significance / 2) + norm.ppf(power)

    var_test, var_control = 0, 0

    for s, strat_data in data.groupby(strat):
        grouped_data = sorted(list(strat_data.groupby(group)), key=lambda x: x[0])
        control_data = grouped_data[0][1][target]
        test_data = grouped_data[1][1][target]

        n_control, n_test = len(control_data), len(test_data)
        strat_control_var = control_data.var(ddof=1) * len(control_data) / n_control
        strat_test_var = test_data.var(ddof=1) * len(test_data) / n_test

        var_control += strat_control_var
        var_test += strat_test_var

    return m * np.sqrt(var_test / n_test + var_control / n_control)


def data_loss(data: pd.DataFrame, group: str, filter_field: str) -> pd.DataFrame:
    """
    Calculate the data loss percentage after applying a filter.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    group (str): The column name to group the data by.
    filter_field (str): The column name to apply the filter on.

    Returns:
    pd.DataFrame: A DataFrame containing the original count, filtered count, and loss percentage for each group.
    """
    orig_data_count = data.groupby(group).agg(orig_count=(group, "count"))
    filtered_data_count = (
        data[data[filter_field]].groupby(group).agg(filtered_count=(group, "count"))
    )
    data_count = orig_data_count.join(filtered_data_count)
    data_count.loc["total", "orig_count"] = len(data)
    data_count.loc["total", "filtered_count"] = len(data[data[filter_field]])
    data_count["loss"] = 1 - data_count["filtered_count"] / data_count["orig_count"]
    return data_count


def mean_diff(data: pd.DataFrame, group: str, filter_field: str) -> pd.DataFrame:
    orig_data_count = data.groupby(group).agg(orig_count=(group, "mean"))
    filtered_data_count = (
        data[data[filter_field]].groupby(group).agg(filtered_count=(group, "mean"))
    )
    data_count = orig_data_count.join(filtered_data_count)
    data_count.loc["total", "orig_count"] = len(data)
    data_count.loc["total", "filtered_count"] = len(data[data[filter_field]])
    data_count["loss"] = 1 - data_count["filtered_count"] / data_count["orig_count"]
    return data_count


def ate(data: pd.DataFrame, target: str, group: str) -> float:
    """
    Calculate the Average Treatment Effect (ATE) for the given data.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    target (str): The column name of the target variable.
    group (str): The column name to group the data by.

    Returns:
    float: The calculated ATE value.
    """
    result = data.groupby(group).agg(ate=(target, "mean")).sort_index()
    return result.iloc[1, 0] - result.iloc[0, 0]


def ttest(
    data: pd.DataFrame, target: str, group: str, significance: float = 0.05
) -> Dict[str, Any]:
    """
    Perform a t-test between two groups in the data.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    target (str): The column name of the target variable.
    group (str): The column name to group the data by.
    significance (float): The significance level for the t-test.

    Returns:
    Dict[str, Any]: A dictionary containing the t-statistic, p-value, pass/fail status, mean difference, and relative mean difference.
    """
    grouped_data = sorted(list(data.groupby(group)), key=lambda x: x[0])
    control_data = grouped_data[0][1][target]
    test_data = grouped_data[1][1][target]

    stat, p_value = stats.ttest_ind(control_data, test_data)
    result = {
        "stat": stat,
        "p_value": p_value,
        "pass": p_value < significance,
        "mean_diff": test_data.mean() - control_data.mean(),
    }
    result["relative_mean_diff"] = result["mean_diff"] / control_data.mean()
    return result


def std_data_loss_tradeoff(
    data: pd.DataFrame, target: str, group: str, filter_field: str
) -> Dict[str, Any]:
    """
    Calculate the tradeoff between standard deviation and data loss.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    target (str): The column name of the target variable.
    group (str): The column name to group the data by.
    filter_field (str): The column name to apply the filter on.

    Returns:
    Dict[str, Any]: A dictionary containing the original standard deviation, filtered standard deviation, data loss, standard deviation ratio, and tradeoff value.
    """
    result = {
        "std_original": group_std(data, target, group),
        "std_filtered": group_std(data[data[filter_field]], target, group),
        "data_loss": data_loss(data, group, filter_field)["loss"],
    }
    result["std_ratio"] = result["std_filtered"] / result["std_original"]
    result["tradeoff"] = result["std_ratio"] * np.power(result["data_loss"] + 100, 3)
    return result


def std_mean_dif_tradeoff(
    data: pd.DataFrame, target: str, group: str, filter_field: str
) -> Dict[str, Any]:
    """
    Calculate the tradeoff between standard deviation and data loss.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    target (str): The column name of the target variable.
    group (str): The column name to group the data by.
    filter_field (str): The column name to apply the filter on.

    Returns:
    Dict[str, Any]: A dictionary containing the original standard deviation, filtered standard deviation, data loss, standard deviation ratio, and tradeoff value.
    """
    result = {
        "std_original": group_std(data, target, group),
        "std_filtered": group_std(data[data[filter_field]], target, group),
        "data_loss": data_loss(data, group, filter_field)["loss"],
    }
    result["std_ratio"] = result["std_filtered"] / result["std_original"]
    result["tradeoff"] = result["std_ratio"] * np.power(result["data_loss"] + 100, 3)
    return result
