import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from core.action_space import ActionSpace
from core.environment import Environment
from core.beliefs import ParticleBelief

from utils.data_classes import ResultsData


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

        self.xlim = [0, 0]
        self.ylim = [0, 0]

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key == "environment":
                self.add_environment(value)
            elif key == "location":
                self.add_location(value)
            elif key == "goal":
                self.add_location(value, is_goal=True)
            elif key == "start":
                self.add_location(value, is_start=True)
            elif key == "action":
                self.add_action(value)
            elif key == "belief":
                self.add_belief(value)
            else:
                raise ValueError(f"Invalid key {key}")

    def add_environment(self, environment: Environment):
        landmarks = environment.landmarks
        detection_range = environment.detection_range
        for landmark in landmarks:
            circle = patches.Circle(
                xy=landmark.loc, radius=detection_range, color="blue", fill=True, alpha=landmark.success_prob
            )
            self.update_limits(landmark.loc, detection_range)
            self.axes.add_patch(circle)

    def add_belief(self, belief: ParticleBelief):
        particles = belief.particles
        weights = belief.weights

        for i in range(belief.n_particles):
            self.axes.plot(*particles[i], "ko", alpha=10 * weights[i], zorder=3)

    def add_location(self, location: np.ndarray, is_start: bool = False, is_goal: bool = False):
        self.update_limits(location)
        if is_start:
            self.axes.plot(*location, "rs", markersize=10, zorder=5)
        elif is_goal:
            self.axes.plot(*location, "y*", markersize=10, zorder=5)
        else:
            self.axes.plot(*location, "yo", markersize=10, zorder=5)

    def add_action(self, action: np.ndarray):
        point_1, point_2 = action
        self.axes.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], "r-", alpha=0.8, zorder=1)

    def show(self):
        plt.show()

    def update_limits(self, point: np.ndarray, radius: float = 0) -> None:
        self.xlim = [min(self.xlim[0], point[0] - radius), max(self.xlim[1], point[0] + radius)]
        self.ylim = [min(self.ylim[0], point[1] - radius), max(self.ylim[1], point[1] + radius)]

    def set_limits(self) -> None:
        delta = 1 * np.array([-1, 1])
        plt.xlim(self.xlim + delta)
        plt.ylim(self.ylim + delta)

    def save(self, plot_dir: str):
        self.set_limits()
        plt.savefig(os.path.join(plot_dir, "env.png"), dpi=1200, bbox_inches="tight")
        plt.close()


def plot_q_function(
    q_function: np.ndarray,
    q_function_bounds: dict[float, np.ndarray],
    dir: str,
    action_space: ActionSpace,
    time_step: int,
):
    plt.figure()
    x_range = np.arange(len(q_function))

    cmap = plt.cm.get_cmap("rainbow")
    indices = np.linspace(0, 1, len(q_function_bounds) + 1)
    colors = [cmap(i) for i in indices]

    # iterate over simplification factors
    for i, (simplification_factor, bounds) in enumerate(q_function_bounds.items()):
        plt.errorbar(
            x_range,
            q_function,
            yerr=np.abs(bounds - q_function.reshape(-1, 1)).T,
            linestyle="",
            capsize=3,
            color=colors[i],
            alpha=0.5,
            label=f"Simplification Factor: {simplification_factor}",
        )
    plt.plot(x_range, q_function, "bs", markersize=5)

    plt.xlabel("Action")
    plt.ylabel("$V^*(b_0)$")
    plt.xticks(range(len(q_function)), [str(act) for act in action_space])
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(dir, f"q_function_{time_step}.png"), bbox_inches="tight", dpi=1200)
    plt.close()


def plot_v_function(value_functions: np.ndarray, value_function_bounds: dict[float, np.ndarray], dir: str):
    plt.figure()

    x_range = np.arange(len(value_functions))

    cmap = plt.cm.get_cmap("rainbow")
    indices = np.linspace(0, 1, len(value_function_bounds) + 1)
    colors = [cmap(i) for i in indices]

    # iterate over simplification factors
    for i, (simplification_factor, bounds) in enumerate(value_function_bounds.items()):
        plt.fill_between(
            x_range,
            y1=bounds[:, 0],
            y2=bounds[:, 1],
            alpha=0.3,
            color=colors[i],
            label=f"Simplification Factor: {simplification_factor}",
        )
    plt.plot(x_range, value_functions, "bs", markersize=5)

    plt.xlabel("Time Step")
    plt.ylabel("$V^*(b_0)$")
    plt.xticks(range(len(value_functions)), range(len(value_functions)))
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(dir, "v_function.png"), bbox_inches="tight", dpi=1200)


def plot_speed_up(results: ResultsData, dir: str):
    mean_speed_up = results.speed_up[0]
    std_speed_up = results.speed_up[1]
    # simplification_factors = results.simplification
    plt.figure()
    plt.errorbar(
        results.node_elimination_rate[0],
        mean_speed_up,
        xerr=results.node_elimination_rate[1],
        yerr=std_speed_up,
        fmt="bs",
        markersize=5,
        capsize=3,
    )
    plt.plot(results.node_elimination_rate[0], mean_speed_up, "b-", alpha=0.7)
    plt.xlabel("Simplification Factor")
    plt.ylabel("Speed Up")
    plt.title("Speed Up vs. Simplification Factor")

    # Save the figure
    plt.savefig(os.path.join(dir, "speed_up.png"), bbox_inches="tight", dpi=1200)
    plt.close()
