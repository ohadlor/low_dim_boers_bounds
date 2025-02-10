from __future__ import annotations
from abc import ABC

import numpy as np


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


class CircleActions(ActionSpace):
    def __init__(self, partitions: int, radius: float, center: bool = False) -> None:
        """Create circular action space with given partitions and radius
        Parameters
        ----------
        partitions : int
            Number of partitions (actions) in the circle
        radius : float
            Radius of the circle
        center : bool, optional
            If True, include the center (0, 0) as an action, by default False
        """
        if center:
            self.actions = np.empty((partitions + 1, 2), dtype=np.float16)
            self.actions[-1] = np.array([0, 0])
        else:
            self.actions = np.empty((partitions, 2), dtype=np.float16)
        for i in range(partitions):
            self.actions[i] = radius * np.array(
                [np.cos(2 * np.pi * i / partitions), np.sin(2 * np.pi * i / partitions)], dtype=np.float16
            )
        super().__init__(self.actions)


class UnitCircleActions(CircleActions):
    def __init__(self, partitions: int, center: bool = False) -> None:
        super().__init__(partitions=partitions, radius=1.0, center=center)
