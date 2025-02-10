from typing import Optional, Sequence
import time
import pickle
import os
from collections import defaultdict

import numpy as np

from core import (
    BeliefTree,
    ParticleBelief,
    Reward,
    ObservationNode,
    Landmark,
    ActionSpace,
    TNode,
    ActionNode,
    DANode,
    PDF,
    Environment,
)
from .data_classes import TimeStepRewardData, TimeStepRewardDataSimplified, LandmarkBunch, NodeCounter


def distance(point_1: np.ndarray, point_2: np.ndarray, axis=None) -> np.ndarray:
    """Euclidean distance between two points

    Parameters
    ----------
    point_1 : np.ndarray
        point in environment
    point_2 : np.ndarray
        point in environment

    Returns
    -------
    np.ndarray
        euclidean distance between two points
    """
    return np.linalg.norm(np.subtract(point_1, point_2).reshape((-1, 2)), axis=axis)


def set_bunches(
    landmarks_coords: Sequence[np.ndarray],
    detection_range: float,
    landmarks_per_bunch: int = 1,
    bunch_radius: float = 1,
    rng: Optional[np.random.Generator | int] = None,
) -> list[LandmarkBunch]:
    """Creates landmarks at given coordinates and adds landmarks around them.

    Parameters
    ----------
    landmarks_coords : Sequence[np.ndarray]
        Coordinates of the main landmarks.
    landmarks_per_bunch : int, optional
        Number of landmarks to create around each main landmark, including the main landmark, by default 1.
    radius : float, optional
        Radius within which to place additional landmarks around each main landmark, by default 1.
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    list[Landmark]
        List of created landmarks.
    """
    rng = np.random.default_rng(rng)

    bunches = []
    for landmark in landmarks_coords:
        landmarks = []
        landmarks.append(Landmark(landmark, 1, None))
        # Add unknown landmarks around known landmark
        for j in range(landmarks_per_bunch - 1):
            success_p = rng.random()
            coord = landmark + bunch_radius * np.array(
                [np.cos(2 * np.pi * j / (landmarks_per_bunch - 1)), np.sin(2 * np.pi * j / (landmarks_per_bunch - 1))],
                dtype=np.float16,
            )
            landmarks.append(Landmark(coord, success_p, rng))
        bunches.append(LandmarkBunch(landmarks, detection_range, landmark))
    return bunches


def set_landmarks(
    landmarks_coords: Sequence[np.ndarray],
    landmarks_per_bunch: int = 1,
    bunch_radius: float = 1,
    rng: Optional[np.random.Generator | int] = None,
) -> list[Landmark]:
    """Creates landmarks at given coordinates and adds landmarks around them.

    Parameters
    ----------
    landmarks_coords : Sequence[np.ndarray]
        Coordinates of the main landmarks.
    landmarks_per_bunch : int, optional
        Number of landmarks to create around each main landmark, including the main landmark, by default 1.
    radius : float, optional
        Radius within which to place additional landmarks around each main landmark, by default 1.
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    list[Landmark]
        List of created landmarks.
    """
    rng = np.random.default_rng(rng)

    landmarks = []
    for landmark in landmarks_coords:
        landmarks.append(Landmark(landmark, rng.random(), None))
        # Add unknown landmarks around known landmark
        for j in range(landmarks_per_bunch - 1):
            success_p = rng.random()
            coord = landmark + bunch_radius * np.array(
                [np.cos(2 * np.pi * j / (landmarks_per_bunch - 1)), np.sin(2 * np.pi * j / (landmarks_per_bunch - 1))],
                dtype=np.float16,
            )
            landmarks.append(Landmark(coord, success_p, rng))
    return landmarks


def step(
    initial_belief: ParticleBelief,
    environment: Environment,
    action_noise: PDF,
    action_space: ActionSpace,
    reward: Reward,
    max_tree_depth: int,
    simplification_factors: Sequence[float],
) -> TimeStepRewardData:
    """Perform a single step in the belief tree planning process.

    Parameters
    ----------
    initial_belief : ParticleBelief
        The initial belief state of the agent.
    environment : Environment
        The known landmarks in the environment.
    action_space : ActionSpace
        The set of possible actions the agent can take.
    reward : Reward
        The reward model used to evaluate actions.
    max_tree_depth : int
        The maximum depth of the belief tree.
    simplification_factors : Sequence[float]
        A sequence of simplification factors to calculate bounds.

    Returns
    -------
    TimeStepRewardData
        The data collected for this time step, including Q-function and bounds.
    """
    root = DANode(initial_belief, np.ndarray([0]))
    root.is_root = True
    print("Building belief tree...")
    tree = BeliefTree(
        root,
        environment,
        action_space,
        action_noise,
        max_tree_depth,
    )
    print(tree)
    print("Belief tree built.\nCalculating Q-function...")

    time_step_data = TimeStepRewardData()

    for simplification_factor in [None] + simplification_factors:
        run_time, nodes_counter = root_q_function(root, reward, simplification_factor)
        if simplification_factor is not None:
            print(
                f"Simplification factor: {simplification_factor}"
                + f"\nNode elimination rate: {nodes_counter.node_elimination_rate}"
                + f"\nFactor elimination rate: {nodes_counter.factor_elimination_rate}"
            )
        # Save data as TimeStepRewardDataSimplified
        simplification_data = TimeStepRewardDataSimplified(
            simplification=simplification_factor,
            time=run_time,
            root_q_function=root.q_function if simplification_factor is None else root.q_function_bounds,
            nodes_counter=nodes_counter,
        )
        time_step_data[simplification_factor] = simplification_data

    return time_step_data


def root_q_function(
    node: TNode,
    reward: Reward,
    simplification_factor: float | None,
    da_node_count: int = 0,
    to_bound_node_count: int = 0,
) -> tuple[float, NodeCounter]:
    """Calculate the Q-function for the root node of the belief tree.

    Parameters
    ----------
    node : TNode
        The current node in the belief tree.
    reward : Reward
        The reward model used to evaluate actions.
    simplification_factor : float | None
        The factor used to simplify the belief tree.

    Returns
    -------
    tuple[float, int, int]
        A tuple containing the runtime, number of nodes removed, and number of factors removed.

    Raises
    ------
    ValueError
        If the node type is unknown.
    """

    run_time = 0
    node_counter = NodeCounter()

    # Downward pass
    for child in node.children:
        if simplification_factor is not None and isinstance(child, ObservationNode):
            subset_size, set_size = child.split_into_subsets(simplification_factor, to_bound_node_count, da_node_count)
            to_bound_node_count += subset_size
            da_node_count += set_size

        child_run_time, child_node_counter = root_q_function(
            child, reward, simplification_factor, da_node_count, to_bound_node_count
        )

        # Upward pass
        node_counter += child_node_counter
        run_time += child_run_time

    t_0 = time.perf_counter()
    for _ in range(1):
        if simplification_factor is None:
            if isinstance(node, ObservationNode):
                node.expectation()
            elif isinstance(node, ActionNode):
                node.reward_calculations(reward)
                node.expectation()
            elif isinstance(node, DANode):
                node_counter.factors += node.beta.sum()
                node_counter.da_nodes += 1
                node.bellman_optimality(reward)
                if not node.is_root:
                    node.reward_calculations(reward)
            else:
                raise ValueError(f"Unknown node type: {type(node)}")
        else:
            if isinstance(node, ObservationNode):
                node.expectation_bounds()
            elif isinstance(node, ActionNode):
                node.reward_calculations(reward)
                node.expectation_bounds()
            elif isinstance(node, DANode):
                node_counter.factors += node.beta.sum()
                node_counter.da_nodes += 1
                node.bellman_optimality_bounds(reward)
                if node.to_bound:
                    factors = node.bound_calculations(reward)
                    # Factors remove is count of observation factors removed, a result of da_diff
                    node_counter.factors_removed += factors
                    # nodes removed is count of how many da nodes are bounded
                    if factors > 0:
                        node_counter.nodes_removed += 1
                elif not node.is_root:
                    node.reward_calculations(reward)

            else:
                raise ValueError(f"Unknown node type: {type(node)}")

    t_1 = time.perf_counter()
    if isinstance(node, TNode):
        run_time += t_1 - t_0
    return run_time, node_counter


def get_q_function(data_dir: str, iter: int, time_step: int) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Get the Q-function data for a specific iteration and time step.

    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    plot_iter : int
        The iteration to plot the Q-function for.
    plot_time_step : int
        The time step to plot the Q-function for.

    Returns
    -------
    tuple[np.ndarray, dict[float, np.ndarray]]
        The Q-function and Q-function bounds.
    """
    # Load the checkpoint
    q_dir = f"{data_dir}/iter_{iter}/checkpoint_{time_step}.pkl"
    with open(q_dir, "rb") as file:
        results = pickle.load(file)
    results = results["results"]
    q_function_bounds = {}

    for simplification_factor, data in results.items():
        if simplification_factor is None:
            q_function = data.root_q_function
        else:
            q_function_bounds[simplification_factor] = data.root_q_function

    return q_function, q_function_bounds


def get_v_function(data_dir: str, iter: int) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Get the V-function data for a specific iteration.

    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    plot_iter : int
        The iteration to plot the V-function for.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The V-function and V-function bounds.
    """
    v_dir = f"{data_dir}/iter_{iter}"
    n = len([file for file in os.listdir(v_dir) if file.endswith(".pkl")])
    v_function = np.empty(n)
    v_function_bounds = defaultdict(lambda: np.empty((n, 2)))

    i = 0
    for file in os.listdir(v_dir):
        if file.endswith(".pkl"):
            time_step = int(file.split("_")[-1].split(".")[0])
            q_function, q_function_bounds = get_q_function(data_dir, iter, time_step)
            v_function[i] = q_function.max()
            for simplification_factor, q_function_bounds in q_function_bounds.items():
                v_function_bounds[simplification_factor][i] = q_function_bounds.max(axis=0)
            i += 1

    return v_function, v_function_bounds
