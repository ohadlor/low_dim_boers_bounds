from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class TNode(ABC):
    def __init__(
        self,
        is_root: bool = False,
    ) -> None:

        self.is_root = is_root

        self.weight = 1
        self.joint_samples = None

        self.propogated_bound_time = 0

        self.action_sequence: list[np.ndarray] = []
        self.children: list[TNode] = []
        self.parent: TNode | None = None

    @abstractmethod
    def create_child_node(self) -> TNode:
        pass

    def drop_level(self) -> TNode:
        """Find a node of same type one level deeper in the tree.

        Returns
        -------
        TNode
            _description_
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
            _description_
        """
        shallow_node = self.parent
        while not isinstance(shallow_node, type(self)):
            shallow_node = shallow_node.parent
        return shallow_node

    @property
    def n_children(self) -> int:
        return len(self.children)
