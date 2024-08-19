from __future__ import annotations
import re
import itertools
from typing import Sequence

import numpy as np

from slicesInference.factor_graph import SlicesFactorGraph
from slicesInference.factor_graph.factors import (
    SymbolicR2PairWiseLinearFactor,
    PriorFactor,
    R2PairWiseLinearGaussianFactor,
)
from slicesInference.distributions import PDF
from slicesInference.functions import LinearFunction

from scenarios.utils import Belief


class SlicesBelief(Belief):
    observation_factor = SymbolicR2PairWiseLinearFactor
    transition_factor = R2PairWiseLinearGaussianFactor
    """
    Wrapper for a SlicesFactorGraph object.
    Represents a belief object as a factor graph using slices.
    Belief is assumed to be a bayes net in belief.partial_bn and new factors in belief.partial_fg by default.
    See Shienman24Arxiv for more details.

    Parameters
    ----------
    slices_factor_graph : SlicesFactorGraph
        The SlicesFactorGraph object associated with the belief.

    Attributes
    ----------
    belief : SlicesFactorGraph
        The slices factor graph object.
    symbols : dict
        A dictionary of factor graph variables.

    Methods
    -------
    __init__(slices_factor_graph)
        Initialize the SlicesBelief object.
    add_transition_factor(transition, noise)
        Add a transition factor to the graph.
    add_observation_factors(landmarks, measurements, noise)
        Add observation factors to the graph.
    inference(initials, n_samples)
        Perform inference using the belief object.
    marginals(variables)
        Access the marginal on a subset of variables.
    get_add_next_pose()
        Get the next pose in the graph.
    current_pose_samples()
        Generate samples for the current pose.
    _tuple_to_symbol(chr, index)
        Convert a tuple to a symbol.
    _get_pose_time(sym)
        Get the time of a pose.
    _symbol_to_id(chr, index)
        Convert a symbol to an ID.
    _id_to_sym(uuid)
        Convert an ID to a symbol.
    _check_and_add_variables(chr, index)
        Check and add variables to the belief.
    """

    def __init__(self, slices_factor_graph: SlicesFactorGraph):
        # belief holds the slices factor graph object.
        self.belief = slices_factor_graph
        # symbols holds dictionary of factor graph variables.
        self.symbols = self.belief.symbols

    def prediction_step(
        self,
        transition: np.ndarray | LinearFunction,
        noise: PDF,
    ) -> None:
        """
        Add transition factor to graph.

        Parameters
        ----------
        transition : np.ndarray
            received action
        noise : PDF
            noise of factor
        """
        current_pose = self.current_pose
        next_pose = self.get_add_next_pose()
        new_factor = self.transition_factor(
            current_pose,
            next_pose,
            transition,
            noise,
        )
        self.belief.add(new_factor)

    def update_step(
        self,
        landmarks: itertools.compress,
        measurements: Sequence[np.ndarray],
        noise: PDF,
    ) -> None:
        """
        Add observation factors to graph.

        Parameters
        ----------
        current_pose : str
            current pose
        landmarks : Sequence[int]
            observed landmarks
        measurements: Sequence[np.ndarray]
            received measurement
        noises : PDF
            factor noise
        """
        for landmark, measurement in zip(landmarks, measurements):
            factor = self.observation_factor(self.current_pose, landmark, measurement, noise)
            self.belief.add(factor)

    def inference(self) -> None:
        """
        Perform inference using the belief object.

        Parameters
        ----------
        n_samples : int
            The number of samples to use for inference.

        Returns
        -------
        None
        """
        self.belief.build_and_inference()

    def get_prior_belief(self) -> SlicesBelief:
        """
        Return the prior belief.

        Returns
        -------
        SlicesFactorGraph
            The prior belief.
        """
        return SlicesBelief(self.belief.remove_past_pose(self.current_pose))

    def marginals(self, variables: list[int]) -> dict[int, PriorFactor]:
        """
        Access the marginal variables.

        Parameters
        ----------
        variables : list[int]
            The list of variables to access.
        n_samples : int
            The number of samples to use for marginals.

        Returns
        -------
        dict[int, np.ndarray]
            The marginal samples of the specified variables.
        """
        marginals = {}
        for var in variables:
            marginals[var] = self.belief.variable_nodes[var].marginal

        return marginals

    def joint_samples(self, n_samples: int) -> dict[int | str, np.ndarray]:
        """
        Generate joint samples from the belief distribution.

        Parameters
        ----------
        n_samples : int
            The number of joint samples to generate.

        Returns
        -------
        dict[int | str, np.ndarray]
            A dictionary containing the joint samples. The keys represent the variable names,
            and the values are numpy arrays containing the samples.
            dict["likelihood"] : np.ndarray - The likelihood of the joint samples.
            dict["weights"] : np.ndarray - The importance weights of the joint samples.

        """
        return self.belief.importance_weighted_joint(n_samples)

    def uniform_joint_samples(self, n_samples: int) -> dict[int | str, np.ndarray]:
        """
        Generate joint samples from the belief distribution.

        Parameters
        ----------
        n_samples : int
            The number of joint samples to generate.

        Returns
        -------
        dict[int | str, np.ndarray]
            A dictionary containing the joint samples. The keys represent the variable names,
            and the values are numpy arrays containing the samples.
            dict["likelihood"] : np.ndarray - The likelihood of the joint samples.
            dict["weights"] : np.ndarray - Uniform weight after resampling.

        """
        joint = self.belief.importance_weighted_joint(n_samples)
        resample_indecies = np.random.choice(n_samples, n_samples, replace=True, p=joint["weights"])
        reweight_joint = {}
        for key, value in joint.items():
            reweight_joint[key] = value[resample_indecies]
        reweight_joint["weights"] = np.ones(n_samples) / n_samples
        return reweight_joint

    def get_add_next_pose(self) -> int:
        """
        Return the next pose in the graph.

        Returns
        -------
        int
            Pose returned as uuid
        """
        current_pose = self._id_to_sym(self.current_pose)
        next_time = self._get_var_time(current_pose) + 1
        return self._check_and_add_variables("x", next_time)

    def current_pose_samples(self, n_samples: int = 100) -> np.ndarray:
        """
        Return samples for the current pose.

        Returns
        -------
        np.ndarray
            The generated samples for the current pose.
        """
        return self.belief.variable_nodes[self.current_pose].marginal.sample(n_samples)

    def _tuple_to_symbol(self, chr: str, index: int) -> str:
        """
        Convert a tuple to a symbol.

        Parameters
        ----------
        chr : str
            The character part of the symbol.
        index : int
            The index part of the symbol.

        Returns
        -------
        str
            The converted symbol.
        """
        return f"{chr}" + "_{" + f"{index}" + "}"

    def _get_var_time(self, sym: str) -> int:
        """
        Get the time of a pose.

        Parameters
        ----------
        sym : str
            The symbol representing the pose.

        Returns
        -------
        int
            The time of the pose.
        """
        return int(re.search(r"\d+", sym).group())

    def _symbol_to_id(self, chr: str, index: int) -> int:
        """
        Convert a symbol to an ID.

        Parameters
        ----------
        chr : str
            Variable type (x or l).
        index : int
            The index part of the symbol.

        Returns
        -------
        int
            The ID corresponding to the symbol.

        Raises
        ------
        ValueError
            If the variable is not found.
        """
        label = self._tuple_to_symbol(chr, index)
        for id, sym in self.symbols.items():
            if sym == label:
                return id
        return False

    def _id_to_sym(self, uuid: int) -> str:
        """
        Convert an ID to a symbol.

        Parameters
        ----------
        uuid : int
            The ID to convert.

        Returns
        -------
        str
            The symbol corresponding to the ID.
        """
        return self.symbols[uuid]

    def _check_and_add_variables(self, chr: str, index: int) -> int:
        """
        Check and add variables to the belief.

        Parameters
        ----------
        chr : str
            The character part of the variable.
        index : int
            The index part of the variable.

        Returns
        -------
        int
            The ID of the variable.

        """
        uuid = self._symbol_to_id(chr, index)

        if not uuid:
            label = f"{chr}" + "_{" + f"{index}" + "}"
            uuid = self.belief.symbol(chr, index)
            self.symbols[uuid] = label
        return uuid

    @property
    def ordered_poses(self) -> list[int]:
        """
        The ordered poses property.

        Returns
        -------
        list[int]
            The list of ordered pose IDs.
        """
        poses = []
        times = []
        for id, sym in self.symbols.items():
            if sym[0] == "x":
                times.append(self._get_var_time(sym))
                poses.append(id)
        poses = [pose for _, pose in sorted(zip(times, poses))]
        return poses

    @ordered_poses.setter
    def ordered_poses(self, next_pose: int):
        """
        Returns the ordered poses.

        This method appends the next_pose to the list of ordered poses.

        Parameters
        ----------
        next_pose : int
            The next pose to be added to the list of ordered poses.
        """
        self.ordered_poses.append(next_pose)

    @property
    def current_pose(self) -> int:
        """
        The current pose.

        Returns
        -------
        int
            The ID of the current pose.
        """
        return self.ordered_poses[-1]

    @property
    def landmarks(self) -> list[int]:
        """
        The landmarks property.

        Returns
        -------
        list[int]
            The list of landmark IDs.
        """
        self._landmarks = []
        for id, sym in self.symbols.items():
            if sym[0] == "l":
                self._landmarks.append(id)
        return self._landmarks

    def copy(self) -> SlicesBelief:
        return SlicesBelief(self.belief.copy_with_bayes_net())
