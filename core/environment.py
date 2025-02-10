from typing import Optional, Sequence
import itertools

import numpy as np

from .utilities import distance
from .landmark import Landmark
from .pdf import Truncated2DGaussianPDF


class Environment:
    def __init__(
        self,
        landmarks: Sequence[Landmark],
        observation_noise: type[Truncated2DGaussianPDF],
        detection_range: float,
        rng: Optional[np.random.Generator | int] = None,
    ) -> None:
        self.rng = np.random.default_rng(rng)

        self.landmarks = landmarks
        for i, landmark in enumerate(self.landmarks):
            landmark.id = i
        self.observation_noise = observation_noise
        self.detection_range = detection_range
        self.obs_dim = 2
        """
        Initialize the environment with landmarks, observation noise, and detection range.

        Parameters
        ----------
        landmarks : Sequence[Landmark]
            The landmarks in the environment.
        observation_noise : type[Truncated2DGaussianPDF]
            The noise model for observations.
        detection_range : float
            The range within which landmarks can be detected.
        rng : Optional[np.random.Generator | int], optional
            The random number generator or seed for reproducibility. Defaults to None.
        """

    def get_betas_in_range(self, pose: np.ndarray, obs_range: float) -> np.ndarray:
        """
        Get da_space given a pose and observation range.

        Parameters
        ----------
        pose : np.ndarray
            The agent's pose.
        obs_range : float
            The observation range of the agent.
        Returns
        -------
        Sequence[Landmark]
            The landmarks in range.
        """
        betas = np.empty((len(pose), len(self.landmarks)), dtype=int)
        for i, landmark in enumerate(self.landmarks):
            betas[:, i] = (distance(pose, landmark.loc, axis=1) <= obs_range).astype(int)
        return betas

    def full_da_observations(
        self, state: np.ndarray, samples_per_state: int = 1
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Get the full data association of the landmarks being observed and the full observations
        For evaluation.

        Parameters
        ----------

        Returns
        -------
        tuple[list[np.ndarray], np.ndarray]
            The full observations and the full data association.
        """
        state = state.repeat(samples_per_state, axis=0)
        betas = self.get_betas_in_range(state, self.detection_range)
        observations = []

        for beta, pose in zip(betas, state):
            landmarks = self.beta_to_landmarks(beta)
            observation = np.empty([len(landmarks), self.obs_dim])
            for index, landmark in enumerate(landmarks):
                observation[index] = self.get_observation(pose.reshape((1, -1)), landmark.loc)
            observations.append(observation)
        return observations, betas

    def get_observation(self, state: np.ndarray, landmark: np.ndarray, samples_per_state: int = 1) -> np.ndarray:
        """
        Get the observation of the landmark being observed.

        Parameters
        ----------
        state : np.ndarray
            The agent's state.
        landmark : np.ndarray
            The landmark's state.

        Returns
        -------
        np.ndarray
            The observation of the landmark.
        """
        n = state.shape[0] * samples_per_state
        noise = self.observation_noise.sample(n)
        return self.relative_position(landmark, state.repeat(samples_per_state, axis=0)) + noise

    def beta_to_landmarks(self, beta: np.ndarray) -> list[Landmark]:
        """
        Convert a beta vector to a list of landmarks.

        Parameters
        ----------
        beta : np.ndarray
            A binary array indicating which landmarks are observed.

        Returns
        -------
        list[Landmark]
            A list of landmarks corresponding to the beta vector.
        """
        landmarks = list(itertools.compress(self.landmarks, beta.astype(bool)))
        return landmarks

    def landmarks_to_beta(self, landmarks: Sequence[Landmark]) -> np.ndarray:
        beta = np.zeros(len(self.landmarks))
        for landmark in landmarks:
            beta[landmark.id] = 1
        return beta

    def get_sub_da_and_weights(self, da_space: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Get the sub data association and weights.

        Parameters
        ----------
        beta : np.ndarray
            The data association.

        Returns
        -------
        tuple[list[np.ndarray], np.ndarray]
            The sub data association and weights.
        """

        # non zero indicies that can be toggled
        indicies = np.where(da_space)[0]
        n = len(indicies)
        n_betas = 2**n
        betas = np.empty((n_betas, len(da_space)), dtype=int)
        weights = np.empty(n_betas, dtype=float)
        for i, truncated_beta in enumerate(itertools.product([0, 1], repeat=n)):
            beta = np.zeros(len(da_space), dtype=int)
            beta[indicies] = truncated_beta
            observed_landmarks = self.beta_to_landmarks(beta)
            beta_diff = da_space - beta
            unobserved_landmarks = self.beta_to_landmarks(beta_diff)
            weight = np.ones(1, dtype=float)
            for landmark in observed_landmarks:
                weight *= landmark.success_prob
            for landmark in unobserved_landmarks:
                weight *= 1 - landmark.success_prob
            betas[i] = beta
            weights[i] = weight
        zero_weight_indicies = np.where(weights == 0)[0]
        betas = np.delete(betas, zero_weight_indicies, axis=0)
        weights = np.delete(weights, zero_weight_indicies)
        return betas, weights

    def in_da_space(self, pose: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Check if the given pose is within the detection range of all landmarks derived from beta.
        Parameters
        ----------
        pose : np.ndarray
            The pose to check, typically a 1D array representing the position.
        beta : np.ndarray
            The parameters used to derive the landmarks.
        Returns
        -------
        np.ndarray
            A boolean array indicating whether the pose is within the detection range of all landmarks.
        """

        landmarks = self.beta_to_landmarks(beta)
        in_range = np.empty((len(landmarks), len(pose)), dtype=bool)
        for i, landmark in enumerate(landmarks):
            in_range[i] = distance(pose, landmark.loc, axis=1) <= self.detection_range
        return in_range.all(axis=0)

    @staticmethod
    def relative_position(landmark: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Get the relative position of the landmark to the agent.

        Parameters
        ----------
        landmark : np.ndarray
            The landmark's state.
        pose : np.ndarray
            The agent's state.

        Returns
        -------
        np.ndarray
            The relative position of the landmark.
        """
        return pose - landmark

    def __str__(self) -> str:
        landmark_locations = [str(landmark.loc) for landmark in self.landmarks]
        return f"Landmark locations: {landmark_locations}"
