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


class OnGroupMetric(Metric, ABC):
    def __init__(
        self,
        target_field: str,
        mark_field: Optional[str] = None,
        mark_type: Literal[None, "filter", "strat"] = None,
        group_field: Optional[str] = None,
    ):
        super().__init__(target_field, mark_field, mark_type)
        self.group_field = group_field

    def split_data(self, data: pd.DataFrame):
        return sorted(list(data.groupby(self.group_field)), key=lambda x: x[0])


class StatisticGroupMetric(OnGroupMetric, ABC):
    def __init__(
        self,
        target_field: str,
        mark_field: Optional[str] = None,
        mark_type: Literal[None, "filter", "strat"] = None,
        group_field: Optional[str] = None,
        significance: float = 0.05,
        power: float = 0.8,
    ):
        super().__init__(target_field, mark_field, mark_type, group_field)
        self.significance = significance
        self.power = power


class MDE(StatisticGroupMetric):
    def calc_function(
        self, data: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        grouped_data = self.split_data(data)
        control_data = grouped_data[0][1][self.target_field]
        test_data = grouped_data[1][1][self.target_field]

        m = norm.ppf(1 - self.significance / 2) + norm.ppf(self.power)

        n_control, n_test = len(control_data), len(test_data)

        var_control, var_test = control_data.var(ddof=1), test_data.var(ddof=1)
        result = m * np.sqrt(var_test / n_test + var_control / n_control)
        return {
            "metric": result,
            "control_size": n_control,
            "test_size": n_test,
            "var_control": var_control,
            "var_test": var_test,
        }

    def strat_calc_function(
        self, data: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        m = norm.ppf(1 - self.significance / 2) + norm.ppf(self.power)

        var_test, var_control = 0, 0
        g_n_test, g_n_control = 0, 0

        result = []

        for s, strat_data in data.groupby(self.mark_field):
            grouped_data = self.split_data(strat_data)
            control_data = grouped_data[0][1][self.target_field]
            test_data = grouped_data[1][1][self.target_field]

            n_control, n_test = len(control_data), len(test_data)
            strat_control_var = control_data.var(ddof=1) * len(control_data) / n_control
            strat_test_var = test_data.var(ddof=1) * len(test_data) / n_test

            var_control += strat_control_var
            var_test += strat_test_var
            g_n_control += n_control
            g_n_test += n_test

            result.append(
                {
                    "strata": s,
                    "metric": None,
                    "control_size": n_control,
                    "test_size": n_test,
                    "var_control": strat_control_var,
                    "var_test": strat_test_var,
                }
            )

        result.append(
            {
                "strata": "total",
                "metric": m * np.sqrt(var_test / g_n_test + var_control / g_n_control),
                "control_size": g_n_control,
                "test_size": g_n_test,
                "var_control": var_control,
                "var_test": var_test,
            }
        )
        return pd.DataFrame(result)


class TTest(StatisticGroupMetric):
    def calc_function(
        self, data: pd.DataFrame, additional_metrics: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        grouped_data = self.split_data(data)
        control_data = grouped_data[0][1][self.target_field]
        test_data = grouped_data[1][1][self.target_field]

        stat, p_value = stats.ttest_ind(control_data, test_data)
        result = {
            "stat": stat,
            "p_value": p_value,
            "pass": p_value < self.significance,
            "mean_diff": test_data.mean() - control_data.mean(),
        }
        result["relative_mean_diff"] = result["mean_diff"] / control_data.mean()
        return result
