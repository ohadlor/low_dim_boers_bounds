import numpy as np

from .environment import Environment
from .functions import distance


class PRM:
    """Generate a probability road map over a given environment"""

    def __init__(
        self,
        environment: Environment | np.ndarray,
        d_lower: float,
        d_upper: float,
        N_nodes: int,
        rng: np.random.Generator,
        start_location: np.ndarray,
        descrete: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        environment : Environment | np.ndarray
            environment object with obstacles or an array describing the environment size
        d_threshold : float
            Threshold distance between connected nodes
        N_nodes : int
            number of nodes generated for map
        start_location : np.ndarray
            start location of the agent
        descrete : bool
            check if the environment is descrete
        """
        self.upper = d_upper
        self.lower = d_lower
        self.N_nodes = N_nodes
        # coords: list of tuples of node coordinates
        self.coords: list[np.ndarray] = []
        # map that is returned as dictionary, key: node index [int],
        #     values: children index w/ weights [list of tuple of int + float]
        self.map: dict[int, list[np.ndarray]] = {}
        self.EPSILON = 1e-2
        self.rng = rng

        if isinstance(environment, Environment):
            # C_obs: set of obstacles given as list of class Obstacles
            self.C_obs = environment.obstacles
            # N: map size [int]
            self.N = environment.STATE_SPACE
            self.dim = environment.DIM
        else:
            self.C_obs = []
            self.N = environment
            self.dim = len(environment)

        self.coords.append(start_location)
        self._update_map(start_location, 0)

        index = 1
        while index < N_nodes:
            coord = self._sample_node()
            if self._is_in_obstacle(coord) or self._is_same(coord):
                pass
            else:
                self.coords.append(coord)
                self._update_map(coord, index)
                index += 1

        self._remove_lone_points()

    def _is_same(self, new_coord: np.ndarray) -> bool:
        return any([distance(new_coord, coord) < self.EPSILON for coord in self.coords])

    def _update_map(self, new_coord: np.ndarray, index: int) -> None:
        edges = self._find_edges(new_coord, index)
        self.map[index] = edges

    def _find_edges(self, new_coord: np.ndarray, index: int) -> list[tuple[int, float]]:
        edges: list[tuple[int, float]] = []
        for n, coord in enumerate(self.coords):
            dist = distance(new_coord, coord)
            if (
                dist < self.upper
                and dist > self.lower
                and not self._edge_obstacle_collision(new_coord, coord)
                and n != index
            ):
                self.map[n].append((index, dist))
                edges.append((n, dist))
        return edges

    def _sample_node(self) -> np.ndarray:
        coord = self.N * self.rng.random(size=self.dim)
        return coord

    def _is_in_obstacle(self, coord: np.ndarray) -> bool:
        return any([obstacle.collision(coord) for obstacle in self.C_obs])

    def _edge_obstacle_collision(
        self, new_coord: np.ndarray, coord: np.ndarray
    ) -> bool:
        return any([obstacle.intersect(new_coord, coord) for obstacle in self.C_obs])

    def _find_closest_point(self, point: np.ndarray) -> tuple[np.ndarray, int]:
        point_idx = np.argmin(distance(np.array(self.coords), point, axis=1))
        prm_point = self.coords[point_idx]
        return prm_point, point_idx

    def get_actions(self, point: np.ndarray) -> list[np.ndarray]:
        prm_point, _ = self._find_closest_point(point)
        index = self.coords.index(prm_point)
        return self.map[index]

    def _remove_lone_points(self) -> None:
        for index in list(self.map.keys()):
            if not self.map[index]:
                del self.map[index]

    def sample_paths(
        self,
        n_paths: int,
        path_length: int,
        path_start: np.ndarray,
        goal_location: np.ndarray,
    ) -> list[list[np.ndarray]]:
        """Sample multiple paths in the PRM.

        This method generates multiple paths in the PRM starting from a given point.
        Each path consists of a sequence of actions that lead from the starting point
        to a randomly selected point in the PRM.

        Parameters
        ----------
        n_paths : int
            The number of paths to sample.
        path_length : int
            The length of each path.
        path_start : np.ndarray
            The starting point for the paths.

        Returns
        -------
        list[list[np.ndarray]]
            A list of paths, where each path is represented as a list of action sequences.
            Each action sequence is a list of numpy arrays representing the actions taken
            to reach the next point in the path as a relative position.
        """
        paths = []
        for _ in range(n_paths):
            _, id = self._find_closest_point(path_start)
            action_sequence = []
            current_dist = distance(path_start, goal_location)

            for _ in range(path_length):
                # Only select node that is closer to the goal
                possible_nodes = []
                for node in self.map[id]:
                    node_dist = distance(self.coords[node[0]], goal_location)
                    if node_dist < current_dist:
                        possible_nodes.append(node)
                if not possible_nodes:
                    raise TypeError("No next steps in the PRM.")

                next_id, _ = self.rng.choice(possible_nodes)
                next_id = int(next_id)
                coords = self.coords[next_id]
                action = coords - self.coords[id]
                action_sequence.append(action)
                current_dist = distance(coords, goal_location)
                id = next_id
            paths.append(action_sequence)
        return paths
