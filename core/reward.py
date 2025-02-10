from __future__ import annotations
from typing import Sequence, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from .utilities import distance

if TYPE_CHECKING:
    from .tree_node import DANode
    from .pdf import Truncated2DGaussianPDF, PDF
    from .beliefs import ParticleBelief


@dataclass
class EntropyParameters:
    normalizer: float
    norm_entropy: float
    pos_entropy: float
    neg_entropy: float


class Reward:
    def __init__(
        self,
        goal_location: np.ndarray,
        transition_noise: PDF,
        obs_noise: Truncated2DGaussianPDF,
        discount_factor: float = 0.95,
        information_weight: float = 0.5,
    ) -> None:
        self.discount_factor = discount_factor
        self.goal_location = goal_location
        self.information_weight = information_weight
        self.transition_noise = transition_noise
        self.obs_noise = obs_noise

        self.obs_noise_params()

    def belief_reward(
        self,
        propogated_belief: ParticleBelief,
        prior_belief: ParticleBelief,
        observation: np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[float, EntropyParameters]:
        entropy, entropy_data = self.entropy_estimator(propogated_belief, prior_belief, observation, landmarks)
        return -entropy, entropy_data

    def entropy_estimator(
        self,
        belief: ParticleBelief,
        prior_belief: ParticleBelief,
        observation: np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[float, EntropyParameters]:
        """An entropy estimator as formulated in the paper.

        Parameters
        ----------
        belief : ParticleBelief
            The current belief represented by particles.
        prior_belief : ParticleBelief
            The prior belief represented by particles.
        observation : np.ndarray
            The observation vector.
        landmarks : np.ndarray
            The landmarks positions.

        Returns
        -------
        tuple[float, EntropyParameters]
            A tuple containing the entropy value and entropy parameters.
        """

        prior_weights = prior_belief.weights
        prior_belief_samples = prior_belief.particles
        belief_samples = belief.particles
        weights = belief.weights

        if landmarks.shape[0] > 0:
            noise = observation - (belief_samples[:, np.newaxis] - landmarks[np.newaxis])
            obs_likelihood = self.obs_noise.likelihood(noise).prod(axis=-1)

            # ! For debugging purposes
            atol = np.maximum(self.obs_noise.Min ** landmarks.shape[0], 1e-12)
            zero_likelihood = np.isclose(obs_likelihood, 0, atol=atol)
            if zero_likelihood.any():
                print(f"Warning: {zero_likelihood.sum()} observation likelihoods are zero.")
        else:
            obs_likelihood = 1

        transition_likelihood = self.transition_noise.likelihood(
            belief_samples[:, np.newaxis] - prior_belief_samples[np.newaxis]
        )

        normalizer = (weights * obs_likelihood).sum(dtype=float)
        norm_entropy = -normalizer * np.log(normalizer)
        entropy_vector = (
            -weights * obs_likelihood * np.log(obs_likelihood * (prior_weights * transition_likelihood).sum(axis=-1))
        )
        mask = entropy_vector > 0
        pos_entropy = entropy_vector[mask].sum(dtype=float)
        neg_entropy = entropy_vector[~mask].sum(dtype=float)

        entropy_data = EntropyParameters(normalizer, norm_entropy, pos_entropy, neg_entropy)

        entropy = pos_entropy + neg_entropy - norm_entropy
        # self.check_validity(entropy)
        return entropy, entropy_data

    def state_reward(self, belief: ParticleBelief) -> float:
        if self.goal_location is None:
            return 0
        reward = -np.dot(belief.weights, distance(belief.particles, self.goal_location, axis=1))
        return reward

    def reward_bounds(self, beta: np.ndarray, ref_da_node: DANode | None) -> tuple[np.ndarray[float, float], int]:
        """Calculate the lower and upper bounds of the expected reward given beta.
        The bounds are formualted in the paper.

        Parameters
        ----------
        beta : np.ndarray
            The reward da being bounded
        ref_da_node : TNode
            The reference DA node and beta used for the bounds

        Returns
        -------
        tuple[np.ndarray, int]
            The bounds and the number of factors eliminated.
            Note, bounds are reversed, as the reward is negative entropy
        """
        entropy_params = ref_da_node.entropy_params
        normalizer = entropy_params.normalizer
        norm_entropy = entropy_params.norm_entropy
        pos_entropy = entropy_params.pos_entropy
        neg_entropy = entropy_params.neg_entropy

        beta_diff = self.beta_diff(beta, ref_da_node.beta)
        n_diff = beta_diff.sum()

        obs_min = self.obs_min if self.obs_min > 1 else self.obs_max
        obs_max = self.obs_max if self.M_log > 0 else self.obs_min
        obs_min_norm = self.obs_min if normalizer > 1 else self.obs_max

        lower_bound = (
            normalizer * n_diff * (self.log_m * obs_min**n_diff - self.M_log * obs_max ** (n_diff - 1))
            - obs_min_norm**n_diff * norm_entropy
            + self.obs_min**n_diff * pos_entropy
            + self.obs_max**n_diff * neg_entropy
        )

        obs_max = self.obs_max if self.obs_max > 1 else self.obs_min
        obs_min = self.obs_min if self.m_log > 0 else self.obs_max
        obs_max_norm = self.obs_max if normalizer > 1 else self.obs_min

        upper_bound = (
            normalizer * n_diff * (self.log_M * obs_max**n_diff - self.m_log * obs_min ** (n_diff - 1))
            - obs_max_norm**n_diff * norm_entropy
            + self.obs_max**n_diff * pos_entropy
            + self.obs_min**n_diff * neg_entropy
        )

        reward_lower_bound = -upper_bound
        reward_upper_bound = -lower_bound

        return np.array([reward_lower_bound, reward_upper_bound], dtype=float), n_diff

    # ! For debugging purposes
    def troubleshoot_reward_bounds(
        self,
        beta: np.ndarray,
        ref_da_node: DANode,
        belief: ParticleBelief,
        prior_belief: ParticleBelief,
        observation: np.ndarray,
        landmarks: np.ndarray,
    ) -> None:
        prior_weights = prior_belief.weights
        prior_belief_samples = prior_belief.particles
        belief_samples = belief.particles
        weights = belief.weights

        ref_beta = ref_da_node.beta
        ref_landmarks = np.asarray([landmark.loc for landmark in ref_da_node.landmarks])
        ref_observation = ref_da_node.observations

        beta_diff = self.beta_diff(beta, ref_beta)

        if landmarks.shape[0] > 0:
            noise = observation - (belief_samples[:, np.newaxis] - landmarks[np.newaxis])
            obs_likelihood = self.obs_noise.likelihood(noise).prod(axis=-1)
        else:
            obs_likelihood = 1

        if ref_landmarks.shape[0] > 0:
            noise = ref_observation - (belief_samples[:, np.newaxis] - ref_landmarks[np.newaxis])
            ref_obs_likelihood = self.obs_noise.likelihood(noise).prod(axis=-1)
        else:
            ref_obs_likelihood = 1

        transition_likelihood = self.transition_noise.likelihood(
            belief_samples[:, np.newaxis] - prior_belief_samples[np.newaxis]
        )

        normalizer = (weights * obs_likelihood).sum(dtype=float)
        ref_normalizer = (weights * ref_obs_likelihood).sum(dtype=float)

        # All components of the entropy
        norm_entropy = -normalizer * np.log(normalizer)
        ref_norm_entropy = -ref_normalizer * np.log(ref_normalizer)

        entropy_1 = (-weights * obs_likelihood * np.log(obs_likelihood / ref_obs_likelihood)).sum(dtype=float)

        entropy_vector = (
            -weights
            * obs_likelihood
            * np.log(ref_obs_likelihood * (prior_weights * transition_likelihood).sum(axis=-1))
        )
        entropy_2 = entropy_vector.sum(dtype=float)

        # Compare to bounds

        mask = entropy_vector > 0
        pos_entropy = entropy_vector[mask].sum(dtype=float)
        neg_entropy = entropy_vector[~mask].sum(dtype=float)

        n_diff = beta_diff.sum()

        obs_min = self.obs_min if self.obs_min > 1 else self.obs_max
        obs_max = self.obs_max if self.M_log > 0 else self.obs_min
        obs_min_norm = self.obs_min if normalizer > 1 else self.obs_max

        norm_lower = normalizer * n_diff * self.log_m * obs_min**n_diff - obs_min_norm**n_diff * ref_norm_entropy
        entropy_1_lower = -normalizer * n_diff * self.M_log * obs_max ** (n_diff - 1)
        entropy_2_lower = self.obs_min**n_diff * pos_entropy + self.obs_max**n_diff * neg_entropy

        obs_max = self.obs_max if self.obs_max > 1 else self.obs_min
        obs_min = self.obs_min if self.m_log > 0 else self.obs_max
        obs_max_norm = self.obs_max if normalizer > 1 else self.obs_min

        norm_upper = normalizer * n_diff * self.log_M * obs_max**n_diff - obs_max_norm**n_diff * ref_norm_entropy
        entropy_1_upper = -normalizer * n_diff * self.m_log * obs_min ** (n_diff - 1)
        entropy_2_upper = self.obs_max**n_diff * pos_entropy + self.obs_min**n_diff * neg_entropy

        # Perform checks
        if not norm_lower <= -norm_entropy <= norm_upper:
            print("Norm entropy not within bounds")
        if not entropy_1_lower <= entropy_1 <= entropy_1_upper:
            print("Entropy 1 not within bounds")
        if not entropy_2_lower <= entropy_2 <= entropy_2_upper:
            print("Entropy 2 not within bounds")

        entropy = -norm_entropy + entropy_1 + entropy_2
        lower_bound = norm_lower + entropy_1_lower + entropy_2_lower
        upper_bound = norm_upper + entropy_1_upper + entropy_2_upper
        return -entropy, -upper_bound, -lower_bound

    def get_reference_node(self, beta: np.ndarray, subset: Sequence[DANode] = []) -> DANode | None:
        """Get the reference DA node that is a subvector of beta

        Parameters
        ----------
        beta : np.ndarray
            The beta values to find the reference node for.
        subset : Sequence[DANode], optional
            A sequence of DANodes representing the subset to consider, by default []

        Returns
        -------
        DANode | None
            The reference DA node that contains beta as a subset, or None if not found.
        """
        # return first da node that contains beta as a subset
        for da_node in reversed(subset):
            if (beta - da_node.beta >= 0).all():
                return da_node
        return None

    def beta_diff(self, beta: np.ndarray, ref_beta: np.ndarray) -> np.ndarray:
        return np.maximum(beta - ref_beta, 0)

    def obs_noise_params(self) -> None:
        self.obs_min = self.obs_noise.Min
        self.obs_max = self.obs_noise.Max
        self.log_m = np.log(self.obs_min)
        self.log_M = np.log(self.obs_max)

        if self.obs_min <= np.exp(-1) <= self.obs_max:
            self.m_log = -np.exp(-1)
        else:
            self.m_log = np.minimum(self.obs_min * self.log_m, self.obs_max * self.log_M, dtype=float)
        self.M_log = np.maximum(self.obs_min * self.log_m, self.obs_max * self.log_M, dtype=float)

    @staticmethod
    def check_validity(value: float) -> None:
        if np.isnan(value) or np.isinf(value):
            raise ValueError("Value is invalid.")
