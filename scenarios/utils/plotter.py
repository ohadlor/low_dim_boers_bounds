import matplotlib.pyplot as plt
import numpy as np

from scenarios.utils.basic_classes import Landmark


class ProblemPlotter:
    def __init__(self):
        self.axes = plt.subplot()
        # plt.rc("text", usetex=True)
        # plt.rc("font", family="serif")
        # plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xticklabels([])
        self.axes.set_yticklabels([])
        self.axes.set_aspect("equal", adjustable="box")

        self.axes.set_facecolor("dimgrey")
        self.limits(np.array([[-4.5, 4.5], [-4.5, 4.5]]))

    def plot_landmarks(
        self, landmarks: list[Landmark], observation_radius: float, cov: list[int], start_location: np.ndarray
    ) -> None:
        """Plot the environment.

        Parameters
        ----------
        env : Environment
            The environment to plot.
        """
        cov = cov / max(cov)
        for i, landmark in enumerate(landmarks):
            landmark.plot(axes=self.axes, radius=observation_radius)
            line = np.array([start_location, start_location + landmark.gt]).squeeze()
            plt.plot(line[:, 0], line[:, 1], "y", markersize=4, alpha=cov[i])

    def plot_actions(
        self,
        gt_locations: np.ndarray,
        bound_actions: np.ndarray,
    ) -> None:
        """Plot the trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            The trajectory to plot.
        """
        for i in range(len(bound_actions)):
            self.axes.arrow(
                *gt_locations[i],
                *bound_actions[i],
                length_includes_head=True,
                head_width=0.15,
                zorder=10,
                width=0.035,
                color="k",
            )
        plt.plot(
            gt_locations[:, 0], gt_locations[:, 1], c="r", linewidth=1.5, linestyle="-", marker="o", markersize=2.5
        )

    def plot_start_goal(self, start: np.ndarray, goal: np.ndarray = None) -> None:
        self.axes.plot(start[0, 0], start[0, 1], "go", markersize=10)
        if goal is not None:
            self.axes.plot(goal[0], goal[1], "ro", markersize=10)

    def plot_belief_particles(self, belief_particles: dict[int, np.ndarray], symbols: dict[int, str]) -> None:
        marker_size = 4
        for key, value in belief_particles.items():
            if isinstance(key, str):
                continue
            plt.scatter(value[:, 0], value[:, 1], label=rf"${symbols[key]}$", s=marker_size)
        plt.legend()

    def show(self):
        plt.show()

    def limits(self, limits: tuple[np.ndarray]) -> None:
        plt.xlim(limits[0])
        plt.ylim(limits[1])

    def save(self, plot_name: str):
        plt.savefig(f"{plot_name}.png", dpi=1200, bbox_inches="tight")
        plt.close()
