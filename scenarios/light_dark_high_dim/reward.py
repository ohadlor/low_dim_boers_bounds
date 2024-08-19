from __future__ import annotations
from typing import Sequence
from itertools import compress

import numpy as np

from slicesInference.factor_graph import SlicesFactorGraph
from scenarios.utils import Reward, Truncated2DGaussianPDF, TNode


class LightDarkReward(Reward):
    def __init__(
        self,
        goal_location: np.ndarray,
        observation_noise: Truncated2DGaussianPDF,
        discount_factor: float = 0.95,
        bound_simplification_factor: float = 0.5,
        information_weight: float = 0.5,
        reward_samples: int = 500,
        q: int = 2,
    ) -> None:

        self.discount_factor = discount_factor
        self.goal_location = goal_location
        self.simplification_factor = bound_simplification_factor
        self.information_weight = information_weight
        self.N_S = reward_samples
        self.obs_noise = observation_noise
        # q-norm constant
        self.q = q

        self.obs_min = self.obs_noise.Min
        self.obs_max = self.obs_noise.Max
        self.log_m = np.log(self.obs_min)
        self.log_M = np.log(self.obs_max)

        self.n_eliminated = 0

    def belief_reward(self, joint: dict[int | str, np.ndarray]) -> float:
        weights = joint["weights"]
        likelihood = joint["likelihood"]
        entropy = self.entropy(likelihood, weights)
        return -entropy

    def entropy(self, likelihood: np.ndarray, weights: np.ndarray = None) -> float:
        entropy = -np.average(np.log(likelihood), weights=weights)
        return entropy

    def state_reward(self, marginal_samples: np.ndarray) -> float:
        if self.goal_location is None:
            return 0
        reward = -np.mean(np.linalg.norm(marginal_samples - self.goal_location, axis=1))
        return reward

    def total_reward(self, joint: dict[int | str, np.ndarray], marginal_samples: np.ndarray) -> float:
        belief_dependent_reward = self.information_weight * self.belief_reward(joint)
        state_dependent_reward = (1 - self.information_weight) * self.state_reward(marginal_samples)
        return belief_dependent_reward + state_dependent_reward

    def bounds(self, comp_subset: Sequence[TNode], subset: Sequence[TNode]) -> tuple[np.ndarray[float, float], int]:
        """Calculate the bounds of the expected belief dependent reward.

        The bounds are per corrolary 7 in the paper. The subset of interest is defined by subset.
        The complementary removed betas are defined by removed_betas and their weights are defined by removed_weights.

        Parameters
        ----------
        comp_subset : Sequence[TNode]
            A sequence of numpy arrays representing the removed betas.
        subset : Sequence[TNode], optional
            An optional sequence of DANodes representing the subset to consider. Defaults to None.

        Returns
        -------
        np.ndarray[float, float]
            A numpy array representing the bounds of the expected reward.
        """
        total_bounds = np.array([0.0, 0.0])
        factors_eliminated = 0
        for da_node in comp_subset:
            beta = da_node.beta
            weight = da_node.weight
            ref_da_node = self.get_reference_node(beta, subset)
            bounds, elim = self.reward_bounds(beta, ref_da_node)
            total_bounds += weight * bounds
            factors_eliminated += elim
        return total_bounds, factors_eliminated

    def reward_bounds(self, beta: np.ndarray, ref_da_node: TNode) -> tuple[np.ndarray[float, float], int]:
        """Calculate the lower and upper bounds of the expected reward given beta.

        Parameters
        ----------
        beta : np.ndarray
            The reward da being bounded
        ref_da_node : TNode
            The reference DA node and beta used for the bounds

        Returns
        -------
        np.ndarray[float, float]
            An array containing the lower and upper bounds of the reward.
            Note, bounds are reversed, as the reward is negative entropy
        """
        beta_diff = self.beta_diff(beta, ref_da_node.beta)
        beta_prime = self.beta_prime(beta, beta_diff)

        simplified_entropy = self.simplified_entropy(ref_da_node, beta_prime)

        upper_bound = np.sum(beta_diff) * (self.log_M - self.log_m)
        lower_bound = -upper_bound - self.upper_cpm(ref_da_node, beta_diff, beta_prime)

        print(f"\nEliminated {sum(beta_diff)} factors")
        self.n_eliminated += sum(beta_diff)
        return np.array([-upper_bound, -lower_bound]) - simplified_entropy, sum(beta_diff)

    def simplified_entropy(self, ref_da_node: TNode, beta_prime: np.ndarray) -> float:

        likelihood = ref_da_node.joint_samples["likelihood"]
        simplified_entropy = 0
        # if no landmarks observed, then equation simplifies
        if np.all(ref_da_node.beta == 0):
            norm = np.linalg.norm(likelihood, ord=self.q)
            simplified_entropy += np.log(norm)
        else:
            # expectation over observations
            for child in ref_da_node.children:
                # TODO: check if this is correct
                prod = likelihood
                observations = compress(child.observations, beta_prime)
                # product of observation factors
                for observation in observations:
                    prod *= observation.likelihood(ref_da_node.joint_samples)
                removed_indicies = np.where(prod == 0)
                SlicesFactorGraph._discard_zero_likelihood_samples(ref_da_node.joint_samples, removed_indicies)
                norm = np.linalg.norm(prod, ord=self.q)
                simplified_entropy += child.weight * np.log(norm)

        likelihood = ref_da_node.joint_samples["likelihood"]
        prior_entropy = self.entropy(likelihood)

        return simplified_entropy + prior_entropy

    def upper_cpm(self, ref_da_node: TNode, beta_diff: np.ndarray, beta_prime: np.ndarray) -> float:
        q = self.q
        m = np.sum(beta_diff)
        p = q * m / (q - 1)
        cpm_diff = -m * np.log(p) / p - np.log(q) / q - m * self.log_m

        cpm_ref = 0
        likelihood = ref_da_node.joint_samples["likelihood"]
        # if no landmarks observed, then equation simplifies
        if np.all(ref_da_node.beta == 0):
            mx = np.min(likelihood)
            Mx = np.max(likelihood)
            log_mx = np.log(mx)
            log_MmMm = np.log(m * self.obs_max ** (p - 1) * self.obs_min + Mx ** (q - 1) * mx)
            cpm_ref = -log_mx + log_MmMm
        else:
            for child in ref_da_node.children:
                prod = likelihood
                observations = compress(child.observations, beta_prime)
                for observation in observations:
                    prod *= observation.likelihood(ref_da_node.joint_samples)
                mx = np.min(prod)
                Mx = np.max(prod)
                log_mx = np.log(mx)
                log_MmMm = np.log(m * self.obs_max ** (p - 1) * self.obs_min + Mx ** (q - 1) * mx)
                cpm_ref += child.weight * (-log_mx + log_MmMm)
        return cpm_diff + cpm_ref

    def get_reference_node(self, beta: np.ndarray, subset: Sequence[TNode] = []) -> TNode:
        for da_node in reversed(subset):
            beta_ref = da_node.beta
            diff = np.sum(self.beta_diff(beta, beta_ref))
            if diff:
                return da_node.copy()
        raise ValueError("No reference node found")

    def beta_diff(self, beta: np.ndarray, ref_beta: np.ndarray) -> np.ndarray:
        return np.maximum(beta - ref_beta, 0)

    def beta_prime(self, beta: np.ndarray, beta_diff: np.ndarray) -> np.ndarray:
        return beta - beta_diff

    def split_into_subsets(self, da_nodes: Sequence[TNode]) -> tuple[list[TNode], list[TNode]]:
        """Split the da_nodes into two subsets.

        Parameters
        ----------
        da_nodes : Sequence[TNode]
            A sequence of da nodes to split.

        Returns
        -------
        tuple[list[TNode], list[TNode]]
            A tuple containing the two subsets.
        """
        n = len(da_nodes)
        subset_size = round(n * self.simplification_factor)
        if np.sum(da_nodes[0].beta) == 0:
            subset = da_nodes[: max(subset_size, 1)]
            comp_set = da_nodes[max(subset_size, 1) :]
        # create a fictitious node with beta = zeros if no node with beta = zeros exists
        else:
            zero_da = [da_nodes[0].copy()]
            subset = zero_da + da_nodes[:subset_size]
            comp_set = da_nodes[subset_size:]

        return subset, comp_set
