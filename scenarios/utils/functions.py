from __future__ import annotations

import numpy as np


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
        euclidean distance
    """
    return np.linalg.norm(np.subtract(point_1, point_2).reshape((-1, 2)), axis=axis)


def actions_to_trajectory(
    actions: list[np.ndarray] | list[list[np.ndarray]], starting_point: np.ndarray
) -> list[np.ndarray] | list[list[np.ndarray]]:
    if isinstance(actions[0], list):
        trajectories = []
        for action_sequence in actions:
            trajectory = actions_to_trajectory(action_sequence, starting_point)
            trajectories.append(trajectory)
        return trajectories
    trajectory = [starting_point]
    for action in actions:
        trajectory.append(trajectory[-1] + action)
    return trajectory
