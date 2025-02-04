from dataclasses import dataclass
from typing import Literal, List, Dict, Callable, Iterable, Optional

import optuna
import pandas as pd
from sklearn.model_selection import train_test_split


class SuggestParameter:
    """Base class for defining a parameter to be suggested by Optuna."""

    name: str


class SuggestNumParameter(SuggestParameter):
    """Class for defining a numerical parameter to be suggested by Optuna.

    Attributes:
        type_ (Literal["int", "float"]): The type of the numerical parameter.
        low (float): The lower bound of the parameter range.
        high (float): The upper bound of the parameter range.
    """

    type_: Literal["int", "float"]
    low: float
    high: float


class SuggestCategoricalParameter(SuggestParameter):
    """Class for defining a categorical parameter to be suggested by Optuna.

    Attributes:
        choices (List): A list of possible choices for the categorical parameter.
    """

    choices: List


def default_model_creation(model_class: type, **model_parameters):
    """Default function for creating a model instance.

    Args:
        model_class (type): The class of the model to be instantiated.
        **model_parameters: Additional parameters to pass to the model constructor.

    Returns:
        An instance of the specified model class.
    """
    return model_class(**model_parameters)


@dataclass
class OptunaProfile:
    """Dataclass for defining an Optuna optimization profile.

    Attributes:
        create_model_function (Callable): Function to create a model instance.
        parameters (List[SuggestParameter]): List of parameters to optimize.
        metric_function (Callable): Function to compute the optimization metric.
        direction (Literal["maximize", "minimize"]): Direction of optimization.
        n_trials (int): Number of trials for optimization.
        timeout (Optional[int]): Timeout for optimization in seconds.
        n_jobs (int): Number of parallel jobs for optimization.
    """

    parameters: List[SuggestParameter]
    metric_function: Callable
    direction: Literal["maximize", "minimize"]
    n_trials: int
    create_model_function: Callable = default_model_creation
    timeout: Optional[int] = None
    n_jobs: int = -1

class OptunaProfiler:
    """Class for performing hyperparameter optimization using Optuna.

    Args:
        data (ModelData): The dataset to use for optimization.
        profile (OptunaProfile): The optimization profile defining the model and parameters.

    Attributes:
        data (ModelData): The dataset to use for optimization.
        profile (OptunaProfile): The optimization profile defining the model and parameters.
        study (optuna.study.Study): The Optuna study object for optimization.
    """

    def __init__(self, data: ModelData, profile: OptunaProfile):
        self.data = data
        self.profile = profile
        self.study = optuna.create_study(direction=self.profile.direction)

    def objective(self, trial: optuna.trial.Trial):
        """Objective function for Optuna optimization.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object.

        Returns:
            float: The computed metric value for the trial.
        """
        trial_params = {}
        for parameter in self.profile.parameters:
            if isinstance(parameter, SuggestCategoricalParameter):
                trial_params[parameter.name] = trial.suggest_categorical(
                    parameter.name, parameter.choices
                )
            else:
                suggest_type = (
                    trial.suggest_int
                    if parameter.type_ == "int"
                    else trial.suggest_float
                )
                trial_params[parameter.name] = suggest_type(
                    parameter.name,
                    parameter.low,
                    parameter.high,
                )

        model = self.profile.create_model_function(**trial_params)
        model.fit(
            self.data.get_data("train", "features"),
            self.data.get_data("train", "target"),
        )

        predictions = model.predict(self.data.get_data("test", "features"))
        return self.profile.metric_function(
            predictions, self.data.get_data("test", "target")
        )

    def optimize(self):
        """Run the optimization process using Optuna."""
        self.study.optimize(
            self.objective,
            n_trials=self.profile.n_trials,
            n_jobs=self.profile.n_jobs,
            timeout=self.profile.timeout,
            show_progress_bar=True,
        )
