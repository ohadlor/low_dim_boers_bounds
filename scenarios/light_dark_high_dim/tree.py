from __future__ import annotations
from typing import Sequence
from operator import itemgetter

import numpy as np

from scenarios.utils import TNode, Landmark

from .basic_classes import UnitCircleActions, Beacon
from .tree_node import ObservationNode, ActionNodeCL, DANode
from .agent import LightDarkAgent
from .reward import LightDarkReward


class BeliefTree:
    def __init__(
        self,
        root_node: TNode,
        landmarks: Sequence[Beacon],
        agent: LightDarkAgent,
        action_space: UnitCircleActions,
        n_samples: int,
        depth: int,
        reward: LightDarkReward,
    ) -> None:
        """
        Initialize the BeliefTree object.

        Parameters
        ----------
        root_node : TNode
            The root node of the belief tree.
        environment : LightDark2D
            The environment in which the agent operates.
        agent : LightDarkAgent
            The agent that uses the belief tree.
        action_space : UnitCircleActions
            The action space of the agent.
        n_samples : int
            The number of state samples -> number of observation node per action node.
        depth : int
            The depth of the belief tree.

        Returns
        -------
        None
        """
        self.root_node = root_node

        self.reward = reward
        self.landmarks = landmarks
        self.agent = agent
        self.action_noise = self.agent.action_noise
        self.observation_noise = self.agent.observation_noise
        self.action_space = action_space
        self.n_samples = n_samples

        self.n_nodes = 0
        self.reward_time = 0
        self.bounds_time = 0

        self.build_tree(self.root_node, depth)

    def build_tree(
        self,
        root_node: TNode,
        depth: int,
    ) -> None:
        """
        Build the belief tree recursively.

        Parameters
        ----------
        root_node : TNode
            The current node being processed.
        depth : int
            The current depth of the tree from the bottom.

        Returns
        -------
        None
        """
        # Downward pass
        if depth != 0:
            self.agent.belief = root_node.belief.copy()

            if isinstance(root_node, ObservationNode):
                self.create_action_nodes(root_node, depth)
            elif isinstance(root_node, ActionNodeCL):
                self.create_da_nodes(root_node, depth)
            elif isinstance(root_node, DANode):
                self.create_observation_nodes(root_node, depth)
            else:
                raise ValueError("Invalid node type")

        # Upward pass
        reward_time, bounds_time = root_node.reward_calculations(self.reward, action_space=self.action_space)

        self.reward_time += reward_time
        self.bounds_time += bounds_time

    def create_action_nodes(self, observation_node: ObservationNode, depth: int) -> None:
        """
        Create action nodes for the given observation node at the specified depth.

        Parameters
        ----------
        observation_node : ObservationNode
            The observation node for which action nodes need to be created.
        depth : int
            The depth at which the action nodes are being created.

        Returns
        -------
        None
        """

        for index, action in enumerate(self.action_space):
            action_node = observation_node.create_child_node(action, self.action_noise)
            self.n_nodes += 1
            self.build_tree(action_node, depth)

    def create_da_nodes(self, action_node: ActionNodeCL, depth: int) -> None:
        """
        Create DA nodes for the given action node at the specified depth.

        Parameters
        ----------
        action_node : ActionNodeCL
            The action node for which DA nodes need to be created.
        depth : int
            The depth at which the DA nodes are being created.

        Returns
        -------
        None
        """

        # Sample betas from joint samples
        joint_samples = action_node.belief.joint_samples(self.n_samples)
        betas = self.agent.get_betas(joint_samples, self.landmarks)
        # sort betas from least landmarks to most landmarks
        sorted_indicies = np.argsort(np.sum(betas, axis=1))
        betas = betas[sorted_indicies]
        betas, indicies, counts = np.unique(betas, axis=0, return_inverse=True, return_counts=True)

        # each da node stores the state samples that have the DA
        for i, (beta, count) in enumerate(zip(betas, counts)):
            samples = {}
            for key, value in joint_samples.items():
                samples[key] = value[np.where(i == indicies)]
            da_node = action_node.create_child_node(
                beta,
                states=samples,
                weight=count / self.n_samples,
            )
            self.n_nodes += 1
            self.build_tree(da_node, depth)

    def create_observation_nodes(self, da_node: DANode, depth: int) -> None:
        """
        Create observation nodes for the given DA node at the specified depth.

        Parameters
        ----------
        da_node : DANode
            The DA node for which observation nodes need to be created.
        depth : int
            The depth at which the observation nodes are being created.

        Returns
        -------
        None
        """
        # each state in the da node generates a single observation node
        samples_per_state = 1
        observations = self.agent.get_observations_from_belief(
            joint_samples=da_node.sampled_states,
            landmarks=beta_to_landmarks(da_node.beta, self.landmarks),
            n_samples=samples_per_state,
        )

        if observations.size == 0:
            observation_node = da_node.create_child_node(None, self.observation_noise, weight=1)
            self.n_nodes += 1
            self.build_tree(observation_node, depth - 1)

        for observation in observations:
            for i in range(samples_per_state):
                observation_node = da_node.create_child_node(
                    observation[..., i], self.observation_noise, weight=1 / len(observations)
                )
                self.n_nodes += 1
                self.build_tree(observation_node, depth - 1)

    def get_node_from_level(self, level: int, node_type: str | type[TNode], base_node: TNode = None) -> TNode:
        """
        Get the node from the tree at the specified level.

        Parameters
        ----------
        level : int
            The level of the node in the tree.
        node_type : str
            The type of the node to be retrieved.

        Returns
        -------
        TNode
            The node at the specified level.
        """
        if base_node is None:
            base_node = self.root_node

        if isinstance(node_type, str):
            if node_type == "observation":
                node_type = ObservationNode
            elif node_type == "action":
                node_type = ActionNodeCL
            elif node_type == "da":
                node_type = DANode
            else:
                raise ValueError("Invalid node type")
        elif issubclass(node_type, TNode):
            pass
        else:
            raise ValueError("Invalid node type")

        if level == 0:
            while not isinstance(base_node, node_type):
                base_node = base_node.children[0]
            return base_node
        else:
            return self.get_node_from_level(level - 1, node_type, base_node.drop_level())


def beta_to_landmarks(beta: np.ndarray, landmarks: Sequence[Landmark]) -> list[Landmark]:
    da = np.nonzero(beta)[0]
    if da.size == 0:
        return []
    landmarks = itemgetter(*da)(landmarks)
    try:
        return list(landmarks)  # Try to convert the value to a list
    except TypeError:
        return [landmarks]
