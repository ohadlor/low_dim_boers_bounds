from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .tree_node import TNode, ObservationNode, ActionNode, DANode

if TYPE_CHECKING:
    from .action_space import UnitCircleActions
    from .environment import Environment
    from .pdf import PDF


class BeliefTree:
    def __init__(
        self,
        root: TNode,
        environment: Environment,
        action_space: UnitCircleActions,
        action_noise: PDF,
        depth: int,
    ) -> None:
        """
        Initialize the BeliefTree object.

        Parameters
        ----------
        root : TNode
            The root node of the belief tree.
        environment : Environment
            The environment with landmarks
        action_space : UnitCircleActions
            The action space
        depth : int
            The depth of the belief tree.

        Returns
        -------
        None
        """
        self.root = root
        self.order_da_by_weight = True

        self.environment = environment
        self.action_noise = action_noise
        self.action_space = action_space
        self.n_observation_samples = 20

        self.n_action_nodes = 0
        self.n_da_nodes = 1
        self.n_observation_nodes = 0

        self.build_tree(self.root, depth)

    def __str__(self) -> str:
        return (
            f"Tree contains {self.n_nodes} nodes, of which {self.n_action_nodes} are action nodes,"
            + f"{self.n_da_nodes} are DA nodes, and {self.n_observation_nodes} are observation nodes.\n"
            + f"On average, each observation node has {self.n_da_nodes / self.n_observation_nodes} DA nodes."
        )

    def build_tree(
        self,
        root: TNode,
        depth: int,
    ) -> None:
        """
        Build the belief tree recursively.

        Parameters
        ----------
        root : TNode
            The current node being processed.
        depth : int
            The remaining depth to be processed.

        Returns
        -------
        None
        """
        if depth != 0:
            self.belief = root.belief.copy()
            root.is_leaf = False

            if isinstance(root, ObservationNode):
                self.create_da_nodes(root, depth)
            elif isinstance(root, ActionNode):
                self.create_observation_nodes(root, depth)
            elif isinstance(root, DANode):
                self.create_action_nodes(root, depth)
            else:
                raise ValueError("Invalid node type")

    def create_action_nodes(self, node: DANode, depth: int) -> None:
        """
        Create action nodes for the given DA node at the specified depth.

        Parameters
        ----------
        node : DANode
            The DA node for which action nodes need to be created.
        depth : int
            The depth at which the action nodes are being created.

        Returns
        -------
        None
        """
        for action in self.action_space.copy():
            action_node = node.create_child_node(action, self.action_noise)
            self.n_action_nodes += 1
            self.build_tree(action_node, depth)

    def create_observation_nodes(self, node: ActionNode, depth: int) -> None:
        """
        Create observation nodes for the given action node at the specified depth.

        Parameters
        ----------
        node : ActionNode
            The action node for which observation nodes need to be created.
        depth : int
            The depth at which the DA nodes are being created.

        Returns
        -------
        None
        """
        particles = node.belief.sample(self.n_observation_samples)
        full_observations, da_spaces = self.environment.full_da_observations(
            particles,
            samples_per_state=1,
        )

        # each da node stores the state samples that have the DA
        for full_observation, da_space in zip(full_observations, da_spaces):
            child_node = node.create_child_node(da_space, full_observation, self.environment)
            self.n_observation_nodes += 1
            self.build_tree(child_node, depth)

    def create_da_nodes(self, node: ObservationNode, depth: int) -> None:
        """
        Create DA nodes for the given observation node at the specified depth.

        Parameters
        ----------
        node : ObservationNode
            The observation node for which DA nodes need to be created.
        depth : int
            The depth at which the observation nodes are being created.

        Returns
        -------
        None
        """
        da_space = node.da_space
        # betas are given sorted by least to most DA
        betas, weights = self.environment.get_sub_da_and_weights(da_space)

        # sort DA from most to least weight
        # first entries are reference das, not sorted
        if self.order_da_by_weight:
            if np.all(betas[0] == 0):
                small_betas = betas[0].reshape(1, -1)
                small_weights = np.array(weights[0]).reshape(-1)
                big_betas = betas[1:]
                big_weights = weights[1:]
            else:
                small_betas = betas[np.sum(betas, axis=1) <= 1]
                small_weights = weights[np.sum(betas, axis=1) <= 1]
                big_betas = betas[np.sum(betas, axis=1) > 1]
                big_weights = weights[np.sum(betas, axis=1) > 1]

            indicies = np.flip(np.argsort(big_weights))
            weights = np.concatenate((small_weights, big_weights[indicies]))
            betas = np.concatenate(
                (
                    small_betas,
                    big_betas[indicies],
                )
            )
            node.available_to_bound = len(big_weights)

        for beta, weight in zip(betas, weights):
            landmarks = self.environment.beta_to_landmarks(beta)

            # only keep the observations that are from the given DA
            truncated_beta = beta[da_space.astype(bool)]
            observation = np.compress(truncated_beta, node.full_observation, axis=0).reshape(
                -1, self.environment.obs_dim
            )

            child_node = node.create_child_node(
                beta, landmarks, weight, observation, self.environment.observation_noise
            )
            self.n_da_nodes += 1
            self.build_tree(child_node, depth - 1)

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
            base_node = self.root

        if isinstance(node_type, str):
            if node_type == "observation":
                node_type = ObservationNode
            elif node_type == "action":
                node_type = ActionNode
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

    @property
    def n_nodes(self) -> int:
        return self.n_action_nodes + self.n_da_nodes + self.n_observation_nodes
