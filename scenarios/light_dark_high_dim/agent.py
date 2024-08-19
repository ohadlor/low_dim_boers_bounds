from __future__ import annotations
from typing import Sequence
from itertools import compress

import numpy as np

from slicesInference.distributions import PDF
from slicesInference.functions import LinearFunction

from scenarios.utils import Agent, distance

from .beliefs import SlicesBelief
from .basic_classes import Beacon
from .reward import LightDarkReward


class LightDarkAgent(Agent):
    """
    Agent class for the LightDark scenario.

    Parameters
    ----------
    action_noise : PDF
        The probability density function representing the action noise.
    observation_noise : PDF
        The probability density function representing the observation noise.
    belief : SlicesBelief
        The belief object representing the agent's belief state, in the form of a factor graph.
    start_location : np.ndarray
        The starting location of the agent.
    agent_observation_range : float
        The range within which the agent can observe landmarks.
    """

    def __init__(
        self,
        action_noise: PDF,
        observation_noise: PDF,
        belief: SlicesBelief,
        start_location: np.ndarray,
        agent_observation_range: float,
    ) -> None:

        self._belief = belief
        self.action_noise = action_noise
        self.observation_noise = observation_noise
        self.path = [start_location]
        self.observation_range = agent_observation_range
        self.obs_dim = self.get_observation(np.array([[0, 0]]), np.array([[0, 0]])).size

        self.cum_reward = 0

    def move_and_update_agent_belief(
        self,
        actions: list[np.ndarray],
        landmarks: Sequence[Beacon],
        reward: LightDarkReward,
    ) -> None:
        """
        Move the agent according to the given action and update the agent's belief.

        Parameters
        ----------
        action : np.ndarray
            The action to be taken by the agent.
        environment : LightDark2D
            The environment in which the agent is moving, with observable landmarks
        n_samples : int
            The number of samples to use for belief inference.

        Returns
        -------
        None
        """
        for action in actions:
            self.move_agent(action)
            self.belief.prediction_step(LinearFunction(np.eye(2), action), self.action_noise)
            landmarks, observations = self.get_observations(landmarks)
            self.belief.update_step(landmarks, observations, self.observation_noise)
            self.belief.inference()
            self.cum_reward += self.calc_reward(reward)
        # code for marginalizing out past poses to create new initial belief on current pose + landmarks
        # WIP
        if False:
            self.belief = self.belief.get_prior_belief()
            self.belief.inference()

    def get_observations(self, landmarks: Sequence[Beacon]) -> dict[int, np.ndarray]:
        """Get set of observations from landmarks in range

        Parameters
        ----------
        environment : LightDark2D
            environment object

        Returns
        -------
        tuple[list[str],list[np.ndarray]]
            list of sampled observations with corresponding landmark symbols
        """
        pose = self.path[-1]
        # Find landmarks in range
        landmarks = self.get_landmarks_in_range(landmarks, pose)
        # Remove landmarks that have failed
        landmarks = compress(landmarks, [landmark.success() for landmark in landmarks])

        observations = []

        for landmark in landmarks:
            observations.append(self.get_observation(landmark.gt, pose))

        return landmarks, observations

    def get_observations_from_belief(
        self,
        joint_samples: dict[int, np.ndarray],
        landmarks: list[Beacon],
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Get observations from the belief particles.
        For planning.

        Parameters
        ----------
        pose_particle : np.ndarray
            The pose particle representing the agent's pose.
        landmarks_particle : np.ndarray
            The landmarks particle representing the agent's belief about the landmarks.
        n_samples : int
            The number of samples to generate for each observation.

        Returns
        -------
        np.ndarray
            A [l joint_samples X m landmarks X n observation dim] numpy array representing the observations.
        """
        # if no observed landmark, return no observations
        if not landmarks:
            return np.array([])

        current_pose = self.belief.current_pose
        observations = np.zeros(
            (
                len(list(joint_samples.values())[0]),
                len(landmarks),
                self.obs_dim,
                n_samples,
            )
        )
        for index, landmark in enumerate(landmarks):
            observations[:, index] = self.get_observation(
                joint_samples[landmark.uuid], joint_samples[current_pose], n_samples
            )

        return observations

    def get_observation(self, landmark: np.ndarray, pose: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Sample observation based on the landmark and pose.
        Model given as : observation = landmark - pose + noise

        Parameters
        ----------
        landmark : np.ndarray
            The coordinates of the landmark.
        pose : np.ndarray
            The current pose of the agent.
        n_samples : int, optional
            The number of samples to generate (default is 1).

        Returns
        -------
        np.ndarray
            The calculated observation.

        """
        rel_dist = landmark - pose
        noise = self.observation_noise.sample(len(pose) * n_samples).reshape((*pose.shape, n_samples))
        return np.repeat(np.expand_dims(rel_dist, len(rel_dist.shape)), n_samples, axis=2) + noise

    def get_betas(self, joint_samples: dict[int, np.ndarray], landmarks: Sequence[Beacon]) -> np.ndarray:
        """
        Get the data association of the landmarks being observed.
        For planning.

        Parameters
        ----------
        environment : LightDark2D
            The environment object containing the landmarks.

        Returns
        -------
        np.ndarray
            The data association of the landmarks.
        """
        n_samples = len(list(joint_samples.values())[0])
        betas = np.zeros((n_samples, len(landmarks)))
        current_pose = self.belief.current_pose

        for index, landmark in enumerate(landmarks):
            # Check if landmark fails to create an observation
            # Helps in promoting different data associations
            success = landmark.success(n_samples)
            betas[:, index] = success & np.array(
                np.linalg.norm(joint_samples[current_pose] - joint_samples[landmark.uuid], axis=1)
                <= self.observation_range
            )
        return betas

    def get_landmarks_in_range(self, landmarks: Sequence[Beacon], pose: np.ndarray) -> list[Beacon]:
        """
        Get the landmarks within the observation range of the agent.

        Parameters
        ----------
        landmarks : list[Beacon]
            A list of available Beacon objects in the environment.
        pose : np.ndarray
            An array representing the current pose of the agent.

        Returns
        -------
        list[Beacon]
            A list of Beacon objects that are within the observation range of the agent.
        """
        landmarks_in_range = []
        for landmark in landmarks:
            dist = distance(landmark.gt, pose)
            if dist <= self.observation_range:
                landmarks_in_range.append(landmark)
        return landmarks_in_range

    def move_agent(self, action: np.ndarray) -> None:
        """Move the agent to a new sampled location and append the location to the path.

        Parameters
        ----------
        action : np.ndarray
            The comanded action.
        """
        location = self.path[-1]
        noise = self.action_noise.sample(1)
        new_location = location + action + noise
        self.path.append(new_location)

    def calc_reward(self, reward: LightDarkReward) -> float:
        """Calculate the reward for the agent's current location.

        Parameters
        ----------
        reward : LightDarkReward
            The reward object to calculate the reward.
        """
        joint = self.belief.joint_samples(reward.N_S)
        current_pose = self.belief.current_pose
        marginal = self.belief.marginals([current_pose])
        marginal_samples = marginal[current_pose].sample(reward.N_S)
        return reward.total_reward(joint, marginal_samples)

    def copy(self) -> LightDarkAgent:
        return LightDarkAgent(
            self.action_noise,
            self.observation_noise,
            self.belief.copy(),
            self.path[-1],
            self.observation_range,
        )

    @property
    def belief(self) -> SlicesBelief:
        return self._belief

    @belief.setter
    def belief(self, new_belief: SlicesBelief) -> None:
        self._belief = new_belief
