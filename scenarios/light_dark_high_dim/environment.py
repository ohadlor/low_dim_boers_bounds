from typing import Optional, Sequence

import numpy as np

from scenarios.utils import Environment, Landmark, Obstacle

from .basic_classes import Beacon, Circle


class LightDark2D(Environment):
    """
    A 2D environment for the LightDark scenario.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator.
    size : np.ndarray, optional
        The size of the environment, by default np.array([10, 10]).
    n_beacons : int, optional
        The number of beacons in the environment, by default 0.
    n_obstacles : int, optional
        The number of obstacles in the environment, by default 0.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        size: np.ndarray = np.array([10, 10]),
        n_beacons: int = 0,
        n_obstacles: int = 0,
        landmarks: Optional[Sequence[Landmark]] = None,
        obstacles: Optional[Sequence[Obstacle]] = None,
    ) -> None:
        super().__init__(
            rng=rng,
            size=size,
            n_obstacles=n_obstacles,
            n_landmarks=n_beacons,
            landmarks=landmarks,
            obstacles=obstacles,
        )

    def _set_landmarks(self, landmark_class: type[Landmark] = Beacon):
        """
        Set the landmarks in the environment.

        Parameters
        ----------
        landmark_class : type[Landmark]
            The class of the landmark.
        """
        self.landmarks = {}
        for index in range(self.n_landmarks):
            location = self.rng.random(size=self.DIM) * self.STATE_SPACE
            failure_prob = self.rng.random(size=1)
            landmark = self._generate_landmark(index, location, failure_prob, landmark_class=landmark_class)
            self.landmarks[landmark.symbol] = landmark

    def _generate_landmark(
        self, index: int, location: np.ndarray, failure_prob: float, landmark_class: type[Landmark]
    ) -> Landmark:
        """
        Generate a landmark.

        Parameters
        ----------
        index : int
            The index of the landmark.
        location : np.ndarray
            The location of the landmark.
        failure_prob : float
            The failure probability of the landmark.
        landmark_class : type[Landmark]
            The class of the landmark.

        Returns
        -------
        Landmark
            The generated landmark.
        """
        landmark = landmark_class(
            location=location,
            symbol="l_" + "{" + f"{index}" + "}",
            failure_prob=failure_prob,
            rng=self.rng,
        )
        return landmark

    def _generate_obstacle(self, obstacle_class=Circle) -> Obstacle:
        """
        Generate an obstacle.

        Parameters
        ----------
        obstacle_class : Circle, optional
            The class of the obstacle, by default Circle.

        Returns
        -------
        Obstacle
            The generated obstacle.
        """
        loc = self.rng.random(size=self.DIM) * self.STATE_SPACE
        radius = self.rng.random(size=1) * self.STATE_SPACE
        obstacle = obstacle_class(center=loc, radius=radius)
        return obstacle
