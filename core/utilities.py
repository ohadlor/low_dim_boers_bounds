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
        euclidean distance between two points
    """

    return np.linalg.norm(np.subtract(point_1, point_2).reshape((-1, 2)), axis=axis)
