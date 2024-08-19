from typing import Sequence
import time

import numpy as np

from slicesInference.distributions import MultiVariateGaussianPDF
from slicesInference.factor_graph import SlicesFactorGraph
from slicesInference.factor_graph.factors import (
    R2PriorGaussianFactor,
    R2PairWiseLinearGaussianFactor,
)
from slicesInference.functions import LinearFunction

from scenarios.utils import Landmark

from .basic_classes import Beacon
from .tree import BeliefTree
from .tree_node import ObservationNode
from .agent import LightDarkAgent
from .reward import LightDarkReward
from .post_functions import save_data, load_checkpoint, plot_q_function


def create_prior_factor_graph(landmarks: list[Landmark]) -> SlicesFactorGraph:
    """
    Create a factor graph with priors from the given landmarks.

    Parameters
    ----------
    landmarks : list[Landmark]
        A list of landmarks.

    Returns
    -------
    SlicesFactorGraph
        The created factor graph.

    """
    fg = SlicesFactorGraph()
    add_priors(fg, landmarks)
    return fg


def add_priors(fg: SlicesFactorGraph, landmarks: list[Landmark]) -> None:
    """
    Add prior factors to the factor graph.

    Parameters
    ----------
    fg : SlicesFactorGraph
        The factor graph to add the prior factors to.
    landmarks : list[Landmark]
        The list of landmarks to add prior factors for.
    """
    # Add first pose
    start_point = np.array([0, 0])
    pose_prior_noise = MultiVariateGaussianPDF(mean=start_point, cov=1e-2 * np.eye(2))
    x0 = fg.symbol("x", 0)
    f0 = R2PriorGaussianFactor(x0, pose_prior_noise)
    fg.add(f0)

    # Add landmark observation factors to pose x0
    for landmark in landmarks:
        l_id = fg.symbol("l", landmark.landmark_number)
        cov = define_landmark_covs([landmark])
        landmark.uuid = l_id
        measurement = LinearFunction(np.eye(2), landmark.gt - start_point)
        landmark_observation_noise = MultiVariateGaussianPDF(mean=np.array([0, 0]), cov=cov * np.eye(2))
        f0l = R2PairWiseLinearGaussianFactor(x0, l_id, measurement, landmark_observation_noise)
        fg.add(f0l)


def define_landmark_covs(landmarks: list[Landmark]) -> list[int]:
    """
    Define the covariance of the landmarks.

    Parameters
    ----------
    landmarks : list[Landmark]
        The landmarks.

    Returns
    -------
    list[int]
        The covariance of the landmarks.
    """
    covs = []
    for landmark in landmarks:
        p_i = landmark.success_prob
        covs.append(0.6e-1 / (p_i + 0.2))
    return covs


def simulation_loop(
    true_agent: LightDarkAgent,
    landmarks: Sequence[Landmark],
    action_space: Sequence,
    reward: LightDarkReward,
    stopping_condition: callable,
    N_O: int,
    max_tree_depth: int,
    directory: str,
    checkpoint_name: str = None,
) -> dict[str, list]:
    """
    Run the simulation loop.

    Parameters
    ----------
    true_agent : LightDarkAgent
        The ground truth agent in the environment.
    initial_belief : SlicesBelief
        The initial belief state of the agent.
    landmarks : Sequence[Landmark]
        The landmarks in the environment.
    action_space : Sequence
        The available actions for the agent.
    reward : LightDarkReward
        The reward function.
    stopping_condition : callable
        A function that determines when to stop the simulation loop.
    N_O : int
        The number of observation samples per action node.
    max_tree_depth : int
        The maximum depth of the belief tree.

    Returns
    -------
    dict[str, list]
        A dictionary containing the simulation results.

    Notes
    -----
    The simulation results are returned as a dictionary with the following keys:
    - 'q_function_bounds': A list of lists containing the lower and upper bounds of the Q-function.
    - 'q_function_optimal': A list of lists containing the optimal Q-function values.
    - 'bound_time': A list of floats representing the time taken to compute the bounds.
    - 'reward_time': A list of floats representing the time taken to compute the rewards.
    - 'optimal_actions': A list of NumPy arrays representing the optimal actions at each iteration.
    - 'bound_actions': A list of NumPy arrays representing the actions selected based on the bounds at each iteration.
    """

    q_function_bounds: list[list[tuple[float]]] = []
    q_function_optimal: list[list[float]] = []
    bound_times: list[float] = []
    reward_times: list[float] = []
    optimal_actions: list[np.ndarray] = []
    bound_actions: list[np.ndarray] = []
    factors_eliminated: list[int] = []
    agent_path: list[np.ndarray] = []
    n_da_nodes: list[list[int]] = []
    n_eliminations: list[list[int]] = []
    results = {
        "q_function_bounds": q_function_bounds,
        "q_function_optimal": q_function_optimal,
        "bound_times": bound_times,
        "reward_times": reward_times,
        "optimal_actions": optimal_actions,
        "bound_actions": bound_actions,
        "factors_eliminated": factors_eliminated,
        "agent_path": agent_path,
        "n_da_nodes": n_da_nodes,
        "n_eliminations": n_eliminations,
    }
    if checkpoint_name is not None:
        checkpoint = load_checkpoint(checkpoint_name)
        true_agent = checkpoint["true_agent"]
        results = checkpoint["results"]

    i = 0
    total_time = 0
    while not stopping_condition(i):
        initial_belief = true_agent.belief.copy()
        root_node = ObservationNode(initial_belief, is_root=True)

        virtual_agent = true_agent.copy()
        t_0 = time.perf_counter()
        tree = BeliefTree(
            root_node,
            landmarks,
            virtual_agent,
            action_space,
            n_samples=N_O,
            depth=max_tree_depth,
            reward=reward,
        )
        t_1 = time.perf_counter()

        optimal_actions.append(root_node.optimal_action)
        bound_actions.append(root_node.bound_action)
        q_function_bounds.append(root_node.q_function_bounds)
        q_function_optimal.append(root_node.q_function)
        bound_times.append(tree.bounds_time)
        reward_times.append(tree.reward_time)
        factors_eliminated.append(reward.n_eliminated)
        agent_path = true_agent.path
        n_da_nodes.append([action_node.n_children for action_node in root_node.children])
        n_eliminations.append([action_node.factors_eliminated for action_node in root_node.children])

        print(
            f"\nEnd of step {i+1}"
            + f"\nStep time: {t_1 - t_0} seconds\n"
            + f"\nOptimal action: {optimal_actions[-1]} \nBound action: {bound_actions[-1]}"
            + f"\nBounds calculation time: {bound_times[-1]} \nRewards calculation time: {reward_times[-1]}"
        )

        # move gt agent
        true_agent.move_and_update_agent_belief([optimal_actions[-1]], landmarks, reward)
        plot_q_function(q_function_bounds[-1], q_function_optimal[-1], action_space, directory, i)
        # save results
        save_data(
            {
                "true_agent": true_agent,
                "results": results,
            },
            directory=directory,
            checkpoint_num=i,
        )
        i += 1
        total_time += t_1 - t_0
    return


def create_beacons(n_landmarks: int, rng: np.random.Generator) -> list[Landmark]:
    """
    Create a list of landmarks with random positions.

    Parameters
    ----------
    landmark_spacing : float
        The spacing between landmarks.
    x_max : float
        The maximum x-coordinate for the landmarks.
    y_max : float
        The maximum y-coordinate for the landmarks.
    rng : np.random.Generator
        The random number generator.

    Returns
    -------
    list[Landmark]
        A list of landmarks with random positions.
    """
    landmarks = []
    radius = 2.5
    theta = np.linspace(0, 2 * np.pi, n_landmarks, endpoint=False)
    for i, angle in enumerate(theta):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        p = 0.9 * abs(0.25 * (y / (radius * 1.1) + 1) * (np.sin(angle + np.pi / 8) + 1)) + 0.1
        landmarks.append(Beacon(np.array([x, y]), i, p, rng))
    return landmarks
