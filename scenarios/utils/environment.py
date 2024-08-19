from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Optional

import numpy as np

from .basic_classes import Landmark, Obstacle


class Environment(ABC):
    """Environment for agent to traverse,
    contains landmarks for loop closer observations,
    can contain obstacles if needed
    """

    def __init__(
        self,
        rng: np.random.Generator,
        size: np.ndarray,
        n_obstacles: Optional[int] = None,
        n_landmarks: Optional[int] = None,
        landmarks: Optional[Sequence[Landmark]] = None,
        obstacles: Optional[Sequence[Obstacle]] = None,
    ) -> None:
        self.DIM = 2
        self.rng = rng
        self.n_obstacles = n_obstacles
        self.n_landmarks = n_landmarks
        self.STATE_SPACE = size

        if landmarks:
            self.landmarks = landmarks
        elif n_landmarks:
            self._set_landmarks()

        if obstacles:
            self.obstacles = obstacles
        elif n_obstacles:
            self._set_obstacles()
        else:
            self.obstacles = []

    def _set_obstacles(self):
        self.obstacles: Sequence[Obstacle] = []
        for _ in range(self.n_obstacles):
            self.obstacles.append(self._generate_obstacle())

    @abstractmethod
    def _set_landmarks(self, landmark_class: type[Landmark]):
        self.landmarks = {}
        for index in range(self.n_landmarks):
            self.landmarks[f"l_{index}"] = self._generate_landmark(landmark_class)

    @abstractmethod
    def _generate_obstacle(self, obstacle_class: type[Obstacle]):
        pass

    @abstractmethod
    def _generate_landmark(self, landmark_class: type[Landmark]):
        pass
