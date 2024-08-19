from __future__ import annotations

from typing import Self
from abc import ABC, abstractmethod


class Belief(ABC):
    def __init__(self, belief_object) -> None:
        """
        initialize a belief with implementation of choice
        Parameters
        ----------
        belief_object : _type_
            the belief representation: particles, pdf, slices, factor graph, etc.
        """
        self.belief = belief_object

    @abstractmethod
    def prediction_step(self, action) -> None:
        """propogate belief from b_k -> b_{k+1}-

        Parameters
        ----------
        action : _type_
            sampled action, transistion factor, etc.
        """
        pass

    @abstractmethod
    def update_step(self) -> None:
        """perform inference to bring belief from b_{k+1}- -> b_{k+1}

        Parameters
        ----------
        observation : _type_
            sampled observation, observation factor, etc.
        """
        pass

    @abstractmethod
    def inference(self) -> None:
        """perform full inference to bring belief from b_{k} -> b_{k+1}

        Parameters
        ----------
        observation : _type_
            sampled observation, observation factor, etc.
        """
        pass

    @abstractmethod
    def copy(self) -> Self:
        pass
