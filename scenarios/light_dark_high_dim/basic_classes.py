from __future__ import annotations
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scenarios.utils import Obstacle, Landmark, distance, ActionSpace


class UnitCircleActions(ActionSpace):
    def __init__(self, partitions: int, center: bool = False) -> None:
        if center:
            self.actions = [np.array([0, 0])]
        else:
            self.actions = []
        for i in range(partitions):
            action = np.array([np.cos(2 * np.pi * i / partitions), np.sin(2 * np.pi * i / partitions)], dtype = np.float16)
            self.actions.append(action)
        super().__init__(self.actions)


class CricleActions(UnitCircleActions):
    def __init__(self, partitions: int, radius: float) -> None:
        super().__init__(partitions)
        self.actions = [action * radius for action in self.actions]


class Beacon(Landmark):
    def __init__(
        self,
        location: np.ndarray,
        beacon_num: int,
        success_prob: float,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(location, beacon_num)
        self.success_prob = success_prob
        self.rng = rng

    def success(self, n_samples: int = 1) -> np.ndarray[bool]:
        """Check if success occurs.

        Returns
        -------
        bool
            True if success occurs, False otherwise.
        """
        return self.rng.binomial(1, self.success_prob, n_samples).astype(bool)

    def plot(self, axes: plt.Axes, radius: float) -> None:
        beacon_drawing = patches.Circle(self.gt, radius, alpha=self.success_prob * 0.7, color="blue")
        axes.plot(self.gt[0], self.gt[1], "o", color="black")
        axes.add_artist(beacon_drawing)


class Circle(Obstacle):
    def __init__(self, center: np.ndarray, radius: float) -> None:
        self.center = center
        self.radius = radius

    def intersect(self, point_1: np.ndarray, point_2: np.ndarray) -> bool:
        # Check if path intersects circle
        det_argument = np.concatenate((point_1 - self.center, point_2 - self.center))
        det = np.linalg.det(det_argument)
        dist = det / distance(point_1, point_2)
        return dist <= self.radius

    def collision(self, point: np.ndarray) -> bool:
        # Check if point is in circle
        dist = np.linalg.norm(point - self.center)
        return dist <= self.radius

    def plot(self, axes: plt.Axes) -> None:
        obstacle_drawing = patches.Circle(self.center, self.radius, set_facecolor="red")
        axes.add_artist(obstacle_drawing)
