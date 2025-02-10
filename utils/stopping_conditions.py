import numpy as np

from .utilities import distance


class StoppingConditions:
    """Class containing stopping conditions for the simulation loop."""

    def __init__(
        self, max_steps: int = 3, goal_location: np.ndarray = np.array([10, 10]), timeout: float = 300
    ) -> None:
        # Number of steps to run the simulation
        self.max_step = max_steps
        self.timeout = timeout
        self.goal = goal_location
        self.goal_delta: float = 1

        self.stopping_variables = {"max_steps": 0, "time_out": 0, "goal_reached": np.array([0, 0])}

    def is_stopped(self, stopping_condition: str) -> bool:
        stopping_variable = self.stopping_variables[stopping_condition]
        return self.get_stopping_function(stopping_condition)(stopping_variable)

    def update_stopping_variables(self, updates: dict) -> None:
        """Update multiple stopping variables.

        Parameters
        ----------
        updates : dict
            Dictionary containing stopping conditions as keys and their new values.
        """
        for condition, value in updates.items():
            if condition in self.stopping_variables:
                self._update_stopping_variable(condition, value)
            else:
                raise ValueError(f"Invalid stopping condition: {condition}")

    def _update_stopping_variable(self, stopping_condition: str, value: float) -> None:
        self.stopping_variables[stopping_condition] = value

    def max_steps(self, step: int) -> bool:
        """Check that the current step is the maximum step.

        Parameters
        ----------
        step : int
            current time step

        Returns
        -------
        bool
            true if current step is max step
        """
        return step == self.max_step

    def goal_reached(self, loc: np.ndarray) -> bool:
        """check if the agent has reached the goal location.

        Parameters
        ----------
        loc : np.ndarray
            current location of the agent

        Returns
        -------
        bool
            true if the current location is within delta of the goal location
        """
        return distance(loc, self.goal, axis=1) <= self.goal_delta

    def time_out(self, time_taken: float) -> bool:
        """Check if the simulation has run for a certain amount of time.

        Parameters
        ----------
        time_taken : float
            time taken to run the simulation

        Returns
        -------
        bool
            true if the time taken is greater than the timeout
        """
        return time_taken > self.timeout

    def get_stopping_function(self, str: str) -> callable:
        """Get the stopping function from the string.

        Parameters
        ----------
        str : str
            the name of the stopping function

        Returns
        -------
        callable
            the stopping function
        """
        if str == "max_steps":
            return self.max_steps
        elif str == "goal_reached":
            return self.goal_reached
        elif str == "time_out":
            return self.time_out
        else:
            raise ValueError("Invalid stopping condition")
