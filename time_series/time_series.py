from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from pandas import DatetimeIndex
from ..statistic.stat_box import WeightedMean, Std

from fast_enum import FastEnum

warn = True


class TimeSeriesTypes(FastEnum):
    """
    Types of tame series
    """
    INDEX: 'TimeSeriesTypes' = 'index'
    GROUP: 'TimeSeriesTypes' = 'group'
    ROLLING: 'TimeSeriesTypes' = 'rolling'
    DIFF: 'TimeSeriesTypes' = 'diff'
    LINEAR_TREND: 'TimeSeriesTypes' = 'linear_trend'
    EXP_1: 'TimeSeriesTypes' = 'exp1'
    EXP_2: 'TimeSeriesTypes' = 'exp2'


class TimeSeries:
    """
    Base class of time series
    """

    def __init__(self, data: pd.DataFrame,
                 type_: TimeSeriesTypes = None, type_params=None):
        """
        :param data: DataFrame with data for constructing time series
        :param type_: type of time series
        :param type_params: parameters of time series
        """
        self.data = data
        self.type_ = type_
        self.type_params = type_params if type_ is not None else {}

    def set_index(self, order_column: str):
        """
        Construct index by order column
        :param order_column: name of order column
        :return: None
        """
        self.data = self.data.set_index(order_column).sort_index()
        self.type_ = TimeSeriesTypes.INDEX
        return self

    def update_params(self, addition_params: dict) -> dict:
        """
        Update type parameters
        :param addition_params: addition parameters
        :return: updated parameters
        """
        self.type_params.update(addition_params)
        return self.type_params

    def get_statistics(self) -> dict:
        wm = WeightedMean(pd.Series(self.data.index))
        if len(self.data.columns) == 1:
            data = self.data[self.data.columns[0]]
            return {
                'mean': wm.calculate(data),
                'std': Std().calculate(data),

            }



def group(ts: TimeSeries, group_window, group_fields=None, agg="mean") -> TimeSeries:
    """
    Group time series by group_window
    :param ts: changed TimeSeries
    :param group_window: the window that the grouping is based on
    :param group_fields:the field by which the grouping takes place
    :param agg: aggregate function
    :return:
    """
    if ts.type_ is None:
        raise ValueError("Need indexed TimeSeries")
    order_column = ts.data.index.name
    data = ts.data.copy().reset_index()
    grouper = pd.Grouper(key=order_column, freq=group_window)
    if group_fields is not None:
        grouped_data = data.groupby(grouper)[group_fields]
    else:
        grouped_data = data.groupby(grouper)
    if agg is not None:
        grouped_data = grouped_data.agg(agg)
    result = TimeSeries(grouped_data, TimeSeriesTypes.GROUP, ts.type_params)
    result.update_params({'group_window': group_window})
    return result


def rolling_trend(ts: TimeSeries, rolling_window, agg='mean') -> TimeSeries:
    """
    The trend obtained by rolling smoothing
    :param ts: changed TimeSeries
    :param rolling_window: rolling window
    :param agg: aggregate function
    :return: TimeSeries with rolling trend
    """
    return TimeSeries(
        ts.data.rolling(rolling_window).agg(agg),
        type_=TimeSeriesTypes.ROLLING,
        type_params=ts.update_params({'rolling_window': rolling_window})
    )


def exp1(ts: TimeSeries, alpha: float) -> TimeSeries:
    """
    Exponential smoothing
    :param ts: changed TimeSeries
    :param alpha: alpha parameter
    :return: TimeSeries with exponential smoothing
    """
    tts = ts.data.copy()
    result = {
        tts.index.name: tts.index
    }
    tts = tts.to_frame(tts) if isinstance(tts, pd.Series) else tts
    for c in tts.columns:
        series = tts[c]
        t_result = [series.iloc[0]]
        for n in range(1, len(series)):
            t_result.append(alpha * series.iloc[n] + (1 - alpha) * t_result[n - 1])
        result[c] = t_result

    result = pd.DataFrame(result).set_index(tts.index.name).sort_index()
    result = TimeSeries(result, TimeSeriesTypes.EXP_1, ts.type_params)
    result.update_params({'exp1_alpha': alpha})
    return result


def exp2(ts: TimeSeries, alpha: float, beta: float, step=None):
    """
    Double exponential smoothing
    :param ts: changed TimeSeries
    :param alpha: alpha parameter
    :param beta: beta parameter
    :param step: forecasting interval
    :return: TimeSeries with double exponential smoothing
    """
    tts = ts.data.copy()
    level = 0
    trend = 0
    result = {
        tts.index.name: list(tts.index)
    }
    if step:
        result[tts.index.name].append(tts.index[-1] + step)
    tts = tts.to_frame(tts) if isinstance(tts, pd.Series) else tts
    for c in tts.columns:
        series = tts[c]
        t_result = [series.iloc[0]]
        for n in range(1, len(series)):
            if n == 1:
                level, trend = series.iloc[0], series.iloc[1] - series.iloc[0]
            value = result[-1] if n >= len(result[tts.index.name]) else series[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            t_result.append(level + trend)
        result[c] = t_result

    result = pd.DataFrame(result).set_index(tts.index.name).sort_index()
    result = TimeSeries(result, TimeSeriesTypes.EXP_2, ts.type_params)
    result.update_params({'exp2_alpha': alpha, 'exp2_beta': alpha})
    return result


def diff(ts: TimeSeries, method='sequential', percent: bool = False):
    """
    Time series of differences
    :param ts: changed TimeSeries
    :param method: sequential - sequential difference of the next step relative to the previous one,
    end - difference relative to the last element
    :param percent: difference in percent
    :return: TimeSeries of differences
    """
    tts = ts.data.copy().dropna()
    result = {
        tts.index.name: tts.index
    }
    tts = tts.to_frame(tts) if isinstance(tts, pd.Series) else tts

    if method == "sequential":
        for c in tts.columns:
            tts['next'] = np.array(list(tts[c].iloc[1:].values) + [0])
            result[c] = tts['next'] - tts[c]
            if percent:
                result[c] = result[c] / tts[c] * 100

    elif method == "end":
        for c in tts.columns:
            result[c] = tts[c].iloc[-1] - tts[c]
            if percent:
                result[c] = result[c] / tts[c].iloc[-1] * 100
    else:
        raise ValueError(f"Not valid method ({method})")
    result = pd.DataFrame(result).set_index(tts.index.name).sort_index()
    result = TimeSeries(result, TimeSeriesTypes.DIFF, ts.type_params)
    result.update_params({'diff_method': method, 'diff_percent': percent})
    return result


def linear_trend(ts: TimeSeries):
    """
    Linear trend
    :param ts: changed TimeSeries
    :return: linear trend
    """
    tts = ts.data.copy()
    result = {
        tts.index.name: tts.index
    }
    tts = tts.to_frame(tts) if isinstance(tts, pd.Series) else tts

    lr_function = None
    for c in tts.columns:
        y = tts[c].values.reshape(1, -1)[0]
        x_is_time = isinstance(tts.index, DatetimeIndex)
        try:
            x = (
                [t.timestamp() for t in tts.index]
                if x_is_time
                else tts.index
            )
            lr_function = np.poly1d(np.polyfit(x, y, 1))
        except:
            x = range(len(tts))
            lr_function = np.poly1d(np.polyfit(x, y, 1))

        result[c] = lr_function(x)

    result = pd.DataFrame(result).set_index(tts.index.name).sort_index()
    result = TimeSeries(result, TimeSeriesTypes.DIFF, ts.type_params)
    result.update_params({'lr_function': lr_function})
    return result


def plot(
        ts: Union[TimeSeries, List[TimeSeries]],
        figsize: Tuple[int, int] = (15, 5),
        title: str = None,
        legend=None,
        xticks=None,
        yticks=None,
        xlabel=None,
        ylabel=None,
        save_directory=None,
        show=True
):
    plt.figure(figsize=figsize)
    ts = [ts] if isinstance(ts, TimeSeries) else ts
    for t in ts:
        tts = t.data.copy()
        plt.plot(tts)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if title is not None:
        plt.title(title)
        if save_directory:
            plt.savefig(Path(save_directory) / f"{title}.jpg")
    if show:
        plt.show()


if __name__ == "__main__":
    # Generate data
    data = pd.DataFrame(
        {
            "time": [datetime.now() + relativedelta(days=i) for i in range(365)],
            "value": [
                np.random.randint(-30, 30) + np.random.randint(-i / 7, i / 3 + 1)
                for i in range(365)
            ],
        }
    )
    # Indexed data
    ts = TimeSeries(data)
    ts.set_index('time')
    plot(ts, title="Indexed data")
    # Grouped data
    gts = group(ts, "30d")
    plot(gts, title="Grouped data")
    # Rolling trend
    rts = rolling_trend(ts, "30d")
    plot(rts, title="Rolling trend")
    # EXP_1
    alpha = 0.02
    e1ts = exp1(ts, alpha)
    plot(e1ts, title=f"Exp_1 a trend (alpha = {alpha})")
    # EXP_2
    alpha = 0.6
    beta = 0.9
    e2ts = exp2(ts, alpha, beta)
    plot(e2ts, title=f"Exp_2 a trend (alpha = {alpha} beta={beta})")
    # Linear trend
    lts = linear_trend(rts)
    plot(lts, title="Linear trend")
    # Diff
    sdts = diff(rts, "sequential", True)
    plot(sdts, title="Sequential diff of rolling data")
    edts = diff(rts, "end", True)
    plot(edts, title="End diff of rolling data")
    ledrs = linear_trend(edts)
    plot([edts, ledrs], legend=['edts', 'ledrs'], title="Linear trend of end dif of rolling data")
