from __future__ import annotations

from abc import ABC, abstractmethod


class Reward(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def belief_reward(self, belief) -> float:
        pass

    @abstractmethod
    def state_reward(self, state) -> float:
        pass

    @abstractmethod
    def total_reward(self, belief, state) -> float:
        pass

    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        pass
