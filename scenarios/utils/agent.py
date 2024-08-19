from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from slicesInference.distributions import PDF

from .beliefs import Belief
from .environment import Environment


class Agent(ABC):
    """An agent that traverses the environment
    The agent can take actions and observations, and maintains a belief
    """

    def __init__(
        self,
        action_noise: PDF,
        observation_noise: PDF,
        belief: Belief,
        start_location: np.ndarray,
    ) -> None:
        self._belief = belief
        self.action_noise = action_noise
        self.observation_noise = observation_noise
        self.path = [start_location]

    @abstractmethod
    def move_and_update_agent_belief(self, action, environment: Environment) -> None:
        self.belief.propogate(action, self.action_noise)
        self.move_agent(action)
        observations = self.get_observations(environment)
        self.belief.inference(observations, self.observation_noise)

    @abstractmethod
    def move_agent(self, action):
        pass

    @abstractmethod
    def get_observations(self, environment: Environment):
        pass

    @abstractmethod
    def copy(self) -> Agent:
        pass

    @property
    def belief(self) -> Belief:
        return self._belief
