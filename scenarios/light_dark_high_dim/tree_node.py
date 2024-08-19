from __future__ import annotations
from itertools import compress
import time

import numpy as np

from slicesInference.distributions import PDF
from slicesInference.functions import LinearFunction

from scenarios.utils import TNode, ActionSpace

from .beliefs import SlicesBelief
from .reward import LightDarkReward


"""
self.state_reward: r_x
    calculated at action node

self.expected_belief_reward: da expectation on the obs expectation of the belief reward
self.bounds: bounds on the expectation of the belief reward
    calculated at action node

self.value_function: max on q function
self.value_function_bounds: max on q function bounds
    calculated at observation node

self.expected_value_function: da expectation on the obs expectation of the value function
self.expected_value_function_bounds: da expectation on the obs expectation of the value function bounds
    calculated at action node

self.q_function: per definition
self.q_function_bounds: per definition using expected_value_function_bounds
    calculated at observation node
"""


class ActionNodeCL(TNode):
    """
    A class representing an action node in a closed loop tree.
    Equivalent to a prior belief node

    Parameters
    ----------
    belief : SlicesBelief
        The belief associated with the action node.
    """

    def __init__(self, belief: SlicesBelief) -> None:
        super().__init__()
        self.belief = belief
        self.child_weights: dict[DANode, float] = {}

    def create_child_node(self, beta: np.ndarray, weight: float, states: np.ndarray) -> DANode:
        """
        Create a child da node based on the given data association.

        Parameters
        ----------
        beta: np.ndarray
            Boolean array indicating the data association.

        Returns
        -------
        DANode
            The created da node.
        """

        child = DANode(self.belief, beta=beta, states=states)
        child.parent = self
        child.weight = weight
        self.children.append(child)

        child.action_sequence = self.action_sequence.copy()

        return child

    def reward_calculations(self, reward: LightDarkReward, action_space: ActionSpace) -> tuple[float, float]:
        """
        Calculate the reward from with expectation over b_k+1-

        Parameters
        ----------
        reward : Reward
            The reward object used to calculate the state reward.

        Returns
        -------
        tuple[float, float]
            A tuple containing the runtime of the reward and bounds.

        """
        # moved joint sampling to child for reward calculation
        # self.joint_samples = self.belief.joint_samples(n_samples=reward.N_S)
        current_pose = self.belief.current_pose
        marginal_samples = self.joint_samples[current_pose]
        self.state_reward = reward.state_reward(marginal_samples)
        if self.state_reward == np.inf or self.state_reward == np.nan or self.state_reward == -np.inf:
            raise ValueError("State reward is infinite or nan")

        t_0 = time.perf_counter()
        self.da_expectation()
        t_1 = time.perf_counter()
        self.da_partial_expectation(reward)
        t_2 = time.perf_counter()

        reward_time = t_1 - t_0
        bounds_time = t_2 - t_1 + self.propogated_bound_time
        return reward_time, bounds_time

    def da_expectation(self) -> None:
        """
        Propagate the value function from the children to the parent node.

        Returns
        -------
        None
        """
        value_functions = [child.obs_expected_value_function * child.weight for child in self.children]
        self.expected_value_function = np.sum(value_functions)

        belief_rewards = [child.obs_expected_belief_reward * child.weight for child in self.children]
        self.expected_belief_reward = np.sum(belief_rewards)

        value_functions_bounds = [child.obs_expected_value_function_bounds * child.weight for child in self.children]
        self.expected_value_function_bounds = np.sum(value_functions_bounds, axis=0)

    def da_partial_expectation(self, reward: LightDarkReward) -> None:
        """Calculate bounds on the expected value function using the partial expectation of the value function
        for the given reward.

        This method calculates the partial expectation of the value function for the given reward.
        It randomly selects a subset of children nodes based on the simplification factor of the reward,
        and calculates the average value function of the selected subset.

        Parameters
        ----------
        reward : LightDarkReward
            The reward object used to calculate the partial expectation.

        Returns
        -------
        None
        """
        # Split the children into a subset and a complementary set
        subset, comp_subset = reward.split_into_subsets(self.children)

        # get the calculation time for the belief reward from the subset
        propogated_bound_times = [child.propogated_bound_time for child in subset]
        self.propogated_bound_time = np.sum(propogated_bound_times)

        # calculate the partial expectation of the belief reward
        if len(subset[0].children) == 0:
            partial_expected_belief_reward = 0
        elif not comp_subset:
            self.da_expectation()
            partial_expected_belief_reward = self.expected_belief_reward
        else:
            partial_belief_rewards = [child.obs_expected_belief_reward * child.weight for child in subset]
            partial_expected_belief_reward = np.sum(partial_belief_rewards)

        # calculate the bounds on the expected belief reward
        if not comp_subset:
            bounds = np.array([0.0, 0.0])
        else:
            bounds, self.factors_eliminated = reward.bounds(comp_subset, subset)
        self.expected_belief_reward_bounds = partial_expected_belief_reward + bounds


class DANode(TNode):
    """
    A class representing an action node in a tree.
    Equivalent to a prior belief node

    Parameters
    ----------
    belief : SlicesBelief
        The belief associated with the action node.
    """

    def __init__(self, belief: SlicesBelief, beta: np.ndarray, states: list[np.ndarray]) -> None:
        super().__init__()
        self.belief = belief
        self.sampled_states = states
        self.beta = beta
        self.weight = 1

    def create_child_node(self, measurements: np.ndarray | None, noise: PDF, weight: float) -> ObservationNode:
        """
        Create a child observation node based on the given measurement and noise.

        Parameters
        ----------
        measurement : np.ndarray
            The measurement used to update the belief.
        noise : PDF
            The noise distribution associated with the measurement.

        Returns
        -------
        ObservationNode
            The created observation node.
        """
        landmarks = compress(self.belief.landmarks, self.beta)
        new_belief = self.belief.copy()
        child = ObservationNode(new_belief)
        if measurements is not None:
            new_belief.update_step(landmarks, measurements, noise)
            obs_factors = [factor.factor for factor in new_belief.belief.factor_nodes.values()]
            new_belief.inference()
            for factor in obs_factors:
                landmark = factor.xj_symbol
                landmark_index = self.belief._get_var_time(self.belief._id_to_sym(landmark))
                child.observations[landmark_index] = factor

        child.parent = self
        child.beta = self.beta
        child.weight = weight
        self.children.append(child)

        child.action_sequence = self.action_sequence.copy()

        return child

    def reward_calculations(self, reward: LightDarkReward, action_space: ActionSpace) -> tuple[float, float]:
        """
        Calculate the reward for the current node.

        Parameters
        ----------
        reward : Reward
            The reward object used for calculating the reward.

        Returns
        -------
        tuple[float, float]
            A tuple containing the runtime of the bound calculations
        """
        if self.parent.joint_samples is None:
            self.joint_samples = self.belief.joint_samples(n_samples=reward.N_S)
            self.parent.joint_samples = self.joint_samples
        else:
            self.joint_samples = self.parent.joint_samples
        t_0 = time.perf_counter()
        self.observation_expectation()
        t_1 = time.perf_counter()

        bounds_time = 0
        reward_time = t_1 - t_0
        self.propogated_bound_time += reward_time

        return reward_time, bounds_time

    def observation_expectation(self) -> None:
        """
        Propagate the value function from the children to the parent node.

        Returns
        -------
        None
        """
        value_functions = [child.value_function * child.weight for child in self.children]
        self.obs_expected_value_function = np.sum(value_functions)

        belief_rewards = [child.belief_reward * child.weight for child in self.children]
        self.obs_expected_belief_reward = np.sum(belief_rewards)

        value_functions_bounds = [child.value_function_bounds * child.weight for child in self.children]
        self.obs_expected_value_function_bounds = np.sum(value_functions_bounds, axis=0)

        propogated_bound_times = [child.propogated_bound_time for child in self.children]
        self.propogated_bound_time = np.sum(propogated_bound_times)

    # Copy is used for reward when creating a new node with zero beta
    def copy(self):
        """Return a copy of the current node."""
        new_node = DANode(self.belief.copy(), self.beta, self.sampled_states.copy())
        new_node.weight = self.weight
        new_node.joint_samples = self.joint_samples.copy()
        new_node.propogated_bound_time = 0
        new_node.obs_expected_belief_reward = 0
        new_node.children = []
        return new_node


class ObservationNode(TNode):
    """A node representing an observation in a tree-based search algorithm.

    Equivalent to a belief node.

    Parameters
    ----------
    belief : SlicesBelief
        The belief state associated with the observation.
    is_root : bool, optional
        Indicates whether the node is the root of the tree. Default is False.
    """

    def __init__(self, belief: SlicesBelief, is_root: bool = False) -> None:
        super().__init__(is_root=is_root)
        self.belief = belief
        self.beta = None
        self.value_function = 0
        self.belief_reward = 0
        self.value_function_bounds = np.array([0.0, 0.0])
        self.weight = 1
        self.observations = [None] * len(belief.landmarks)

    def create_child_node(self, action: np.ndarray, noise: PDF) -> ActionNodeCL:
        """Create a child node based on an action and noise.

        This method creates a new child node by propagating the belief state
        using the given action and noise.

        Parameters
        ----------
        action : Action
            The action taken from the current observation.
        noise : PDF
            The noise associated with the action.

        Returns
        -------
        ActionNode
            The newly created child node.
        """
        new_belief = self.belief.copy()
        new_belief.prediction_step(transition=LinearFunction(np.eye(2), action), noise=noise)
        new_belief.inference()

        child = ActionNodeCL(new_belief)
        child.parent = self
        child.action_sequence = self.action_sequence.copy()
        child.action_sequence.append(action)
        self.children.append(child)

        return child

    def reward_calculations(self, reward: LightDarkReward, action_space: ActionSpace) -> tuple[float, float]:
        """
        Calculate the information theoretical reward for the current node.

        Parameters
        ----------
        reward : Reward
            The reward object used for calculating the reward.

        Returns
        -------
        tuple[float, float]
            A tuple containing the runtime of the belief dependent reward calculations
        """
        t_0 = time.perf_counter()
        self.joint_samples = self.belief.joint_samples(n_samples=reward.N_S)
        self.belief_reward = reward.belief_reward(self.joint_samples)
        if self.belief_reward == np.inf or self.belief_reward == np.nan or self.belief_reward == -np.inf:
            raise ValueError("State reward is infinite or nan")
        t_1 = time.perf_counter()
        self.calculate_q_function(action_space, reward)
        t_2 = time.perf_counter()
        self.calculate_q_function_bounds(action_space, reward)
        t_3 = time.perf_counter()

        self.propogated_bound_time = t_1 - t_0
        bounds_time = t_3 - t_2
        reward_time = t_2 - t_0
        return reward_time, bounds_time

    def calculate_q_function(self, action_space: ActionSpace, reward: LightDarkReward) -> None:
        """
        Propagate the value function from the children to the parent node.

        Returns
        -------
        None
        """
        if not self.children:
            return
        belief_weight = reward.information_weight
        state_weight = 1 - belief_weight
        gamma = reward.discount_factor

        self.q_function = [
            state_weight * child.state_reward
            + belief_weight * child.expected_belief_reward
            + gamma * child.expected_value_function
            for child in self.children
        ]
        optimal_index = np.argmax(self.q_function)
        self.optimal_action = action_space[optimal_index]
        self.value_function = self.q_function[optimal_index]

    def calculate_q_function_bounds(self, action_space: ActionSpace, reward: LightDarkReward) -> None:
        """
        Propagate the value function from the children to the parent node.

        Returns
        -------
        None
        """
        if not self.children:
            return
        belief_weight = reward.information_weight
        state_weight = 1 - belief_weight
        gamma = reward.discount_factor

        self.q_function_bounds = np.array(
            [
                state_weight * child.state_reward
                + belief_weight * child.expected_belief_reward_bounds
                + gamma * child.expected_value_function_bounds
                for child in self.children
            ]
        )
        # get maximum bounds
        optimal_index = np.argmax(self.q_function_bounds, axis=0)
        # get the optimal action from lower bounds
        self.bound_action = action_space[optimal_index[0]]
        for i, index in enumerate(optimal_index):
            self.value_function_bounds[i] = self.q_function_bounds[index, i]
