from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from core.landmark import Landmark


@dataclass
class TimeStepRewardDataSimplified:
    simplification: float | None
    time: float
    root_q_function: list[np.ndarray[float, float] | float]
    nodes_counter: NodeCounter

    def __str__(self) -> str:
        root_value = (
            f"({max(x[0] for x in self.root_q_function)}, {max(x[1] for x in self.root_q_function)})"
            if isinstance(self.root_q_function[0], np.ndarray)
            else str(max(self.root_q_function))
        )
        return f"\n    Time: {self.time}, Root Value Function Bounds: {root_value}"


class TimeStepRewardData(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: float | None, value: TimeStepRewardDataSimplified):
        if not isinstance(key, (float, type(None))):
            raise TypeError("Key must be a float or None")
        if not isinstance(value, TimeStepRewardDataSimplified):
            raise TypeError("Value must be a TimeStepRewardDataSimplified instance")
        super().__setitem__(key, value)

    def __getitem__(self, key: float | None) -> TimeStepRewardDataSimplified:
        if not isinstance(key, (float, type(None))):
            raise TypeError("Key must be a float or None")
        return super().__getitem__(key)

    def __str__(self) -> str:
        return "".join(f"Simplification {k}: {v}\n" for k, v in self.items())


@dataclass
class ResultsData:
    simplification: list[float | None]
    time: tuple[np.ndarray[float], np.ndarray[float]]
    speed_up: tuple[np.ndarray[float], np.ndarray[float]]
    factors: np.ndarray[int]
    n_da_nodes: np.ndarray[int]
    factors_eliminated: tuple[np.ndarray[int], np.ndarray[int]]
    nodes_eliminated: tuple[np.ndarray[int], np.ndarray[int]]
    node_elimination_rate: tuple[np.ndarray[float], np.ndarray[float]]
    factor_elimination_rate: tuple[np.ndarray[float], np.ndarray[float]]


@dataclass
class LandmarkBunch:
    landmarks: list[Landmark]
    detection_range: float
    center: np.ndarray


@dataclass
class NodeCounter:
    da_nodes: int = 0
    factors: int = 0
    nodes_removed: int = 0
    factors_removed: int = 0

    def __add__(self, other: NodeCounter) -> NodeCounter:
        return NodeCounter(
            da_nodes=self.da_nodes + other.da_nodes,
            factors=self.factors + other.factors,
            nodes_removed=self.nodes_removed + other.nodes_removed,
            factors_removed=self.factors_removed + other.factors_removed,
        )

    @property
    def node_elimination_rate(self) -> float:
        if self.da_nodes == 0:
            return 0.0
        return self.nodes_removed / self.da_nodes

    @property
    def factor_elimination_rate(self) -> float:
        if self.factors == 0:
            return 0.0
        return self.factors_removed / self.factors
