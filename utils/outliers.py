import numpy as np
import pandas as pd


class QuantileTailClipper:
    def __init__(
        self,
        contamination=None,
        learning_rate=0.001,
        max_iter=None,
        min_diff=None,
        algorithm="auto",
    ):
        self.contamination = contamination
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_diff = min_diff
        self.lower_quantile = 0
        self.upper_quantile = 1
        self.algorithm = algorithm

    @staticmethod
    def _calculate_diff(x: pd.Series, lower_quantile: float, upper_quantile: float):
        lv, uv = None, None
        if lower_quantile > 0:
            lv = x.quantile(lower_quantile)
        if upper_quantile < 1:
            uv = x.quantile(upper_quantile)
        cx = x.copy()
        if lv is not None:
            cx = cx[cx >= lv]
        if uv is not None:
            cx = cx[cx <= uv]
        return 1 - cx.std() / x.std()

    def fit(self, X: np.array, y=None):
        best_diff = 0
        best_i = 0
        iteration = 0

        x = pd.Series(X.flatten())
        if self.contamination:
            self.upper_quantile = 1 - self.contamination

            for i in np.arange(
                0, self.contamination + self.learning_rate, self.learning_rate
            ):
                i_lower_quantile = self.lower_quantile + i
                i_upper_quantile = self.upper_quantile + i
                diff = self._calculate_diff(x, i_lower_quantile, i_upper_quantile)
                if diff > best_diff:
                    best_diff = diff
                    best_i = i

                if self.max_iter:
                    iteration += 1
                    if iteration >= self.max_iter:
                        break

                if self.min_diff:
                    if best_diff < self.min_diff:
                        break

            self.lower_quantile += best_i
            self.upper_quantile += best_i

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = pd.Series(X.flatten())
        r = (
            (
                (x >= x.quantile(self.lower_quantile))
                & (x <= x.quantile(self.upper_quantile))
            )
            * 1
        ).replace({0: -1})
        return r.values

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X)
        return self.predict(X)
