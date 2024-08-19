from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Coords(ABC):
    def __init__(self, coords: np.ndarray) -> None:
        self.coords = coords
        self.x = coords[0]
        self.y = coords[1]


class Action(ABC):
    @abstractmethod
    def __init__(self, transition: np.ndarray) -> None:
        self.transition = transition
        """
        Parameters
        ----------
        action : np.ndarray
            array that describes action, to be impemented
        """


class ActionSpace(ABC):
    def __init__(self, actions: list) -> None:
        self.actions = actions
        self._index = 0
        """
        Parameters
        ----------
        action : np.ndarray
            array that describes action, to be impemented
        """

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.actions):
            action = self[self._index]
            self._index += 1
            return action
        else:
            self._index = 0
            raise StopIteration

    def __getitem__(self, index: int):
        return self.actions[index]

    def copy(self) -> ActionSpace:
        return ActionSpace(self.actions)


class Observation(ABC):
    @abstractmethod
    def __init__(self, measurement: np.ndarray) -> None:
        """
        Parameters
        ----------
        observation : np.ndarray
            array that describes relative observation, to be implemented
        """
        self.measurement = measurement


class Pose(ABC):
    def __init__(self, location: np.ndarray, uuid: int) -> None:
        """
        Parameters
        ----------
        location : np.ndarray
            agent coordinates
        uuid : int
            pose unique id
        """
        self.gt = location
        self.uuid = uuid


class Landmark(ABC):
    def __init__(self, location: np.ndarray, landmark_number: int) -> None:
        """
        Parameters
        ----------
        location : np.ndarray
            landmark coordinates
        symbol : str
            landmark identifier
        """
        self.gt = location
        self.landmark_number = landmark_number
        self.uuid = None

    @property
    def symbol(self) -> str:
        return "l_" + "{" + f"{self.landmark_number}" + "}"


# Not used


class Obstacle(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def intersect(self, point_1: np.ndarray, point_2: np.ndarray) -> bool:
        # Check if path intersects obstacle
        pass

    @abstractmethod
    def collision(self, point: np.ndarray) -> bool:
        # Check if point is inside obstacle
        pass

    @abstractmethod
    def plot_obstacle(self, figure) -> None:
        pass
