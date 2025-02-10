from __future__ import annotations
from typing import Optional, Sequence, TYPE_CHECKING
from itertools import compress

import numpy as np

from .utilities import distance

if TYPE_CHECKING:
    from .beliefs import ParticleBelief
    from .landmark import Landmark
    from .pdf import Truncated2DGaussianPDF, PDF


class Agent:
    """
    Agent class for the LightDark scenario.

    Parameters
    ----------
    action_noise : PDF
        The probability density function representing the action noise.
    observation_noise : PDF
        The probability density function representing the observation noise.
    belief : ParticleBelief
        The belief object representing the agent's belief state, in the form of a factor graph.
    start_location : np.ndarray
        The starting location of the agent.
    agent_observation_range : float
        The range within which the agent can observe landmarks.
    """

    def __init__(
        self,
        action_noise: PDF,
        observation_noise: Truncated2DGaussianPDF,
        belief: ParticleBelief,
        start_location: np.ndarray,
        agent_observation_range: float,
        rng: Optional[np.random.Generator | int] = None,
    ) -> None:

        self._belief = belief
        self.action_noise = action_noise
        self.observation_noise = observation_noise
        self.path = [start_location]
        self.observation_range = agent_observation_range
        self.obs_dim = 2

        self.rng = np.random.default_rng(rng)

    def move_and_update_agent_belief(
        self,
        action: np.ndarray,
        landmarks: Sequence[Landmark],
    ) -> None:
        """
        Move the agent according to the given action and update the agent's belief.

        Parameters
        ----------
        action : np.ndarray
            The action to be taken by the agent.
        environment : LightDark2D
            The environment in which the agent is moving, with observable landmarks
        Returns
        -------
        None
        """
        observation_noise = self.observation_noise

        self.move_agent(action, stocastic=False)
        self.belief.prediction_step(action, self.action_noise)
        landmarks, observations = self.get_observations(landmarks, observation_noise)
        self.belief.update_step(landmarks, observations, observation_noise)

    def get_observations(
        self, landmarks: Sequence[Landmark], observation_noise: Truncated2DGaussianPDF
    ) -> tuple[list[Landmark], np.ndarray]:
        """Get set of observations from landmarks in range

        Parameters
        ----------
        landmarks : Sequence[Landmark]
            A sequence of Landmark objects.
        observation_noise : Truncated2DGaussianPDF
            The observation noise model.

        Returns
        -------
        tuple[list[Landmark], np.ndarray]
            A tuple containing a list of landmarks in range and their corresponding observations.
        """
        pose = self.path[-1]
        # Find landmarks in range
        landmarks = self.get_landmarks_in_range(landmarks, pose)
        # Remove landmarks that have failed
        landmarks = compress(landmarks, [landmark.success() for landmark in landmarks])

        observations = []

        for landmark in landmarks:
            observation = self._get_observation(landmark.loc, pose, observation_noise)
            observations.append(observation)

        return landmarks, observations

    def _get_observation(
        self, landmark: np.ndarray, pose: np.ndarray, observation_noise: Truncated2DGaussianPDF
    ) -> np.ndarray:
        """
        Sample observation based on the landmark and pose.
        Model given as : observation = pose - landmark + noise

        Parameters
        ----------
        landmark : np.ndarray
            The coordinates of the landmark.
        pose : np.ndarray
            The current pose of the agent.
        observation_noise : Truncated2DGaussianPDF
            The observation noise model.

        Returns
        -------
        np.ndarray
            The calculated observation.

        """
        rel_pos = pose - landmark
        noise = observation_noise.sample()
        return rel_pos + noise

    def get_landmarks_in_range(self, landmarks: Sequence[Landmark], pose: np.ndarray) -> list[Landmark]:
        """
        Get the landmarks within the observation range of the agent.

        Parameters
        ----------
        landmarks : list[Landmark]
            A list of available Landmark objects in the environment.
        pose : np.ndarray
            An array representing the current pose of the agent.

        Returns
        -------
        list[Landmark]
            A list of Landmark objects that are within the observation range of the agent.
        """
        landmarks_in_range = []
        for landmark in landmarks:
            dist = distance(landmark.loc, pose, axis=1)
            if dist <= self.observation_range:
                landmarks_in_range.append(landmark)
        return landmarks_in_range

    def move_agent(self, action: np.ndarray, stocastic: bool = True) -> None:
        """Move the agent to a new sampled location and append the location to the path.

        Parameters
        ----------
        action : np.ndarray
            The comanded action.
        """
        location = action + self.path[-1]
        if stocastic:
            noise = self.action_noise.sample()
            location += noise
        self.path.append(location)

    def copy(self) -> Agent:
        return Agent(
            self.action_noise,
            self.observation_noise,
            self.belief.copy(),
            self.path[-1],
            self.observation_range,
            self.rng,
        )

    @property
    def belief(self) -> ParticleBelief:
        return self._belief

    @belief.setter
    def belief(self, new_belief: ParticleBelief) -> None:
        self._belief = new_belief
