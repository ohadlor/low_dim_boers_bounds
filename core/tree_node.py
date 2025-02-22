from __future__ import annotations
from typing import Optional, Sequence, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from .environment import Environment
    from .beliefs import ParticleBelief
    from .reward import Reward
    from .pdf import PDF
    from .landmark import Landmark


class TNode(ABC):
    def __init__(self, belief: ParticleBelief) -> None:

        self.is_root = False
        self.is_leaf = True
        self.children: list[TNode] = []
        self.parent: Optional[TNode] = None
        self.prior_belief: Optional[ParticleBelief] = None
        self.weight: float = 1

        self.belief = belief

    @abstractmethod
    def create_child_node(self) -> TNode:
        pass

    def drop_level(self) -> TNode:
        """Find a node of same type one level deeper in the tree.

        Returns
        -------
        TNode
            A node of the same type one level deeper in the tree.
            This node is a child of the current node.
        """
        deeper_node = self.children[0]
        while not isinstance(deeper_node, type(self)):
            deeper_node = deeper_node.children[0]
        return deeper_node

    def up_level(self) -> TNode:
        """Find a node of same type one level shallower in the tree.

        Returns
        -------
        TNode
            A node of the same type one level shallower in the tree.
            This node is a parent of the current node.
        """
        shallow_node = self.parent
        while not isinstance(shallow_node, type(self)):
            shallow_node = shallow_node.parent
        return shallow_node

    @property
    def n_children(self) -> int:
        return len(self.children)

    @staticmethod
    def check_validity(value):
        """Check if the value is valid (finite and not NaN).

        Parameters
        ----------
        value : float
            The value to be checked.

        Raises
        ------
        ValueError
            If the value is not finite or is NaN.
        """
        if np.isinf(value) or np.isnan(value):
            raise ValueError("Value is invalid")


class ObservationNode(TNode):
    def __init__(self, belief: ParticleBelief, da_space: np.ndarray, full_observation: np.ndarray) -> None:
        """Create an observation node. With unknown DA

        Parameters
        ----------
        belief : ParticleBelief
            The belief state of the node.
        da_space : np.ndarray
            The data association space.
        full_observation : np.ndarray
            The full observation data.

        Returns
        -------
        None
        """
        super().__init__(belief=belief)
        self.da_space = da_space
        self.full_observation = full_observation

        self.optimal_value_function = np.zeros(1, dtype=float)
        self.optimal_value_function_bounds = np.array([0.0, 0.0], dtype=float)

    def create_child_node(
        self,
        beta: np.ndarray,
        landmarks: Sequence[Landmark],
        weight: float,
        observation: np.ndarray,
        observation_noise: PDF,
    ) -> DANode:
        new_belief = self.belief.copy()
        child = DANode(new_belief, beta, landmarks, observation, weight)
        new_belief.update_step(landmarks, observation, observation_noise)
        child.prior_belief = self.prior_belief
        child.propogated_belief = self.belief
        child.parent = self
        self.children.append(child)

        return child

    def expectation(self) -> None:
        n = len(self.children)
        value_function = np.empty(n, dtype=float)
        belief_reward = np.empty(n, dtype=float)

        for index, child in enumerate(self.children):
            value_function[index] = child.optimal_value_function * child.weight
            belief_reward[index] = child.belief_reward * child.weight

        self.expected_value_function = value_function.sum(dtype=float)
        self.expected_belief_reward = belief_reward.sum(dtype=float)

    def expectation_bounds(self) -> None:
        n = len(self.children)
        value_function_bounds = np.empty((n, 2), dtype=float)
        belief_reward_bounds = np.empty((n, 2), dtype=float)

        for index, child in enumerate(self.children):
            value_function_bounds[index] = child.optimal_value_function_bounds * child.weight
            belief_reward_bounds[index] = (
                child.reward_bounds * child.weight
                if child.to_bound
                else np.repeat(child.belief_reward, 2) * child.weight
            )

        self.expected_value_function_bounds = value_function_bounds.sum(axis=0, dtype=float)
        self.expected_belief_reward_bounds = belief_reward_bounds.sum(axis=0, dtype=float)

    def split_into_subsets(self, simplification_factor: float, nodes_marked: int, total_nodes: int) -> tuple[int, int]:
        """
        Splits the children of the current node into two subsets based on the given simplification factor.
        Parameters
        ----------
        simplification_factor : float
            A factor used to define the splitting process.
        Returns
        -------
        None
        """
        full_set = self.children
        n = len(full_set)
        subset_size = int(total_nodes * simplification_factor - nodes_marked)
        subset_size = np.minimum(subset_size, self.available_to_bound)
        # first node is always the lowest da
        # da_nodes are sorted from least amount of da to most or by weight
        subset = full_set[: n - subset_size]
        comp_set = full_set[-subset_size:]

        for child in subset:
            child.to_bound = False

        for child in comp_set:
            child.to_bound = True
            child.ref_set = subset

        return subset_size, n


class ActionNode(TNode):
    def __init__(self, belief: ParticleBelief) -> None:
        """Initialize an ActionNode with the given belief.

        Parameters
        ----------
        belief : ParticleBelief
            The belief state of the node.
        """
        super().__init__(belief=belief)

    def create_child_node(
        self, da_space: np.ndarray, full_observation: np.ndarray, environment: Environment
    ) -> ObservationNode:
        """
        Create a child observation node based on the given data association space and full observation.

        Parameters
        ----------
        da_space : np.ndarray
            The data association space.
        full_observation : np.ndarray
            The full observation data.
        environment : Environment
            The environment in which the node operates.

        Returns
        -------
        ObservationNode
            The created observation node.
        """
        new_belief = self.belief.copy()
        child = ObservationNode(new_belief, da_space, full_observation)
        # remove particles outside the da space
        new_belief.filter_step(da_space, environment)
        # remove particles outside the observation space
        new_belief.filter_observation_space(
            environment.beta_to_landmarks(da_space), full_observation, environment.observation_noise
        )
        child.prior_belief = self.belief
        child.parent = self
        self.children.append(child)

        return child

    def reward_calculations(self, reward: Reward) -> None:
        self.state_reward = reward.state_reward(self.belief)
        self.check_validity(self.state_reward)

    def expectation(self) -> None:
        value_function = np.empty(len(self.children), dtype=float)
        belief_reward = np.empty(len(self.children), dtype=float)

        for index, child in enumerate(self.children):
            value_function[index] = child.optimal_value_function
            belief_reward[index] = child.expected_belief_reward

        self.expected_value_function = value_function.mean(dtype=float)
        self.expected_belief_reward = belief_reward.mean(dtype=float)

    def expectation_bounds(self) -> None:
        value_functions_bounds = np.empty((len(self.children), 2), dtype=float)
        belief_reward_bounds = np.empty((len(self.children), 2), dtype=float)

        for index, child in enumerate(self.children):
            value_functions_bounds[index] = child.optimal_value_function_bounds
            belief_reward_bounds[index] = child.expected_belief_reward_bounds

        self.expected_value_function_bounds = value_functions_bounds.mean(axis=0, dtype=float)
        self.expected_belief_reward_bounds = belief_reward_bounds.mean(axis=0, dtype=float)


class DANode(TNode):
    def __init__(
        self,
        belief: ParticleBelief,
        beta: np.ndarray,
        landmarks: Optional[Sequence[Landmark]] = None,
        observations: Optional[np.ndarray] = None,
        weight: float = 1,
    ) -> None:
        """Initialize a DANode with the given parameters.

        Parameters
        ----------
        belief : ParticleBelief
            The belief state of the node.
        beta : Optional[np.ndarray]
            The beta parameter for the node.
        landmarks : Optional[Sequence[Landmark]]
            The landmarks associated with the node.
        observations : Optional[np.ndarray]
            The observations associated with the node.
        """
        super().__init__(belief=belief)
        self.beta = beta
        self.landmarks = landmarks
        self.observations = observations
        self.weight = weight

        self.observation_noise = None
        self.propogated_belief: ParticleBelief = None
        self.to_bound: bool = False
        self.ref_set: list[DANode] = []

        self.optimal_value_function = np.zeros(1, dtype=float)
        self.optimal_value_function_bounds = np.array([0, 0], dtype=float)
        self.reward_bounds = np.array([0, 0], dtype=float)
        self.belief_reward = np.zeros(1, dtype=float)

    def create_child_node(self, action: np.ndarray, noise: PDF) -> ActionNode:
        new_belief = self.belief.copy()
        new_belief.prediction_step(action, noise)

        child = ActionNode(new_belief)
        child.prior_belief = self.belief
        child.parent = self
        self.children.append(child)

        return child

    def reward_calculations(self, reward: Reward) -> None:
        # Belief reward is given as an estimator of the entropy
        landmark_locs = np.asarray([landmark.loc for landmark in self.landmarks])
        self.belief_reward, self.entropy_params = reward.belief_reward(
            self.propogated_belief, self.prior_belief, self.observations, landmark_locs
        )

    def bound_calculations(self, reward: Reward) -> int:
        ref_da_node = reward.get_reference_node(self.beta, self.ref_set)
        self.reward_bounds, factors_eliminated = reward.reward_bounds(self.beta, ref_da_node)

        # ! For debugging purposes
        lower_bound, upper_bound = self.reward_bounds
        if not lower_bound <= self.belief_reward <= upper_bound:
            print("\nBounds are not valid:")
            ref_node = reward.get_reference_node(self.beta, self.ref_set)
            landmark_locs = np.asarray([landmark.loc for landmark in self.landmarks])

            belief_reward, lower_bound, upper_bound = reward.troubleshoot_reward_bounds(
                self.beta, ref_node, self.propogated_belief, self.prior_belief, self.observations, landmark_locs
            )
            if not np.isclose(belief_reward, self.belief_reward):
                print("Reward is not valid")
            if not np.isclose(lower_bound, self.reward_bounds[0]):
                print("Lower bound is not valid")
            if not np.isclose(upper_bound, self.reward_bounds[1]):
                print("Upper bound is not valid")

        return factors_eliminated

    def bellman_optimality(self, reward: Reward) -> None:
        if not self.children:
            return

        belief_weight = reward.information_weight
        state_weight = 1 - belief_weight
        gamma = reward.discount_factor

        self.q_function = np.empty(len(self.children), dtype=float)
        for index, child in enumerate(self.children):
            self.q_function[index] = (
                state_weight * child.state_reward
                + belief_weight * child.expected_belief_reward
                + gamma * child.expected_value_function
            )

        self.optimal_value_function = self.q_function.max()

    def bellman_optimality_bounds(self, reward: Reward) -> None:
        if not self.children:
            return
        belief_weight = reward.information_weight
        state_weight = 1 - belief_weight
        gamma = reward.discount_factor

        self.q_function_bounds = np.empty((len(self.children), 2), dtype=float)
        for index, child in enumerate(self.children):
            self.q_function_bounds[index] = np.array(
                [
                    state_weight * child.state_reward
                    + belief_weight * child.expected_belief_reward_bounds
                    + gamma * child.expected_value_function_bounds
                ]
            )

        self.optimal_value_function_bounds = self.q_function_bounds.max(axis=0)


if __name__ == "__main__":
    pass
