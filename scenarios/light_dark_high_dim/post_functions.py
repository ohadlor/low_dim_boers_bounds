import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from .basic_classes import ActionSpace


def save_data(data: dict, directory: str, checkpoint_num: int) -> None:
    """
    Save the checkpoint to "log/scenario_name/run_time/iteration_num.pkl".

    Parameters
    ----------
    data : dict
        The data to save in the checkpoint.
    """
    log_dir = os.path.join(directory)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_path = os.path.join(log_dir, f"checkpoint_{checkpoint_num}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(path: str, checkpoint_name: str) -> dict:
    """
    Load the checkpoint from "path/file_name.pkl".

    Parameters
    ----------
    path : str
        The path to load the checkpoint from.

    Returns
    -------
    dict
        The loaded checkpoint data.
    """
    file_path = os.path.join(path, checkpoint_name + ".pkl")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def results_to_text(results: dict, directory: str) -> None:
    """
    Save the results to a text file.

    Parameters
    ----------
    results : dict
        The results to save.
    directory : str
        The directory to save the results to.
    """
    file_path = os.path.join(directory, "results.txt")
    with open(file_path, "w") as f:
        f.write("\nBound Times\n")
        for i, time in enumerate(results["bound_times"]):
            f.write(f"Iter {i+1}: {time}\n")
        total_bound_time = np.sum(np.array(results["bound_times"]), axis=1)
        average_bound_time = np.mean(total_bound_time)
        std_bound_time = np.std(total_bound_time)
        f.write(f"Total: {average_bound_time}+-{std_bound_time}\n")
        f.write("\nReward Times\n")
        for i, time in enumerate(results["reward_times"]):
            f.write(f"Iter {i+1}: {time}\n")
        total_reward_time = np.sum(np.array(results["reward_times"]), axis=1)
        average_reward_time = np.mean(total_reward_time)
        std_reward_time = np.std(total_reward_time)
        f.write(f"Total: {average_reward_time}+-{std_reward_time}\n")
        total_reward_time = np.sum(np.array(results["reward_times"]), axis=1)
        speedup = total_reward_time / total_bound_time
        average_speedup = np.mean(speedup)
        std_speedup = np.std(speedup)
        f.write(f"\nSpeedup: {average_speedup}+-{std_speedup}\n")
        total_factors_elim = np.sum(np.array(results["factors_eliminated"]), axis=1)
        average_elimination = np.mean(total_factors_elim)
        std_elimination = np.std(total_factors_elim)
        f.write(f"\nN eliminated factors: {average_elimination}+-{std_elimination}\n")


def results_to_result(results: list[dict]) -> dict:
    combined_results = {}
    for result in results:
        for key, value in result.items():
            combined_results[key] = combined_results.get(key, []) + [value]

    return combined_results


def plot_q_function(
    q_function_bounds: np.ndarray,
    q_function_optimal: np.ndarray,
    action_space: ActionSpace,
    dir: str,
    time_i: int,
):
    plt.figure()
    x_range = np.arange(len(q_function_optimal))
    for j in range(len(q_function_bounds)):
        plt.vlines(
            x=x_range[j], ymin=q_function_bounds[j][0], ymax=q_function_bounds[j][1], color="grey", linewidth=3.5
        )
    plt.plot(x_range, q_function_optimal, "bs", markersize=5)
    bound_label = ["Lower Bound", "Upper Bound"]
    bound_marker = ["bv", "b^"]
    for k in reversed(range(2)):
        plt.plot(x_range, q_function_bounds[:, k], bound_marker[k], markersize=4, label=bound_label[k])

    plt.xlabel("Action")
    plt.ylabel("Value Function")
    plt.title("Q-Function")
    plt.xticks(range(len(q_function_optimal[0])), [str(act) for act in action_space])
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(dir, f"q_function_{time_i}.png"), bbox_inches="tight", dpi=1200)
    plt.close()


def plot_q_functions(
    q_function_bounds: dict[int, np.ndarray],
    q_function_optimal: dict[int, np.ndarray],
    action_space: ActionSpace,
    dir: str,
):
    # Convert q_function_bounds and q_function_optimal to numpy arrays
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    shapes = ["s", "o", "*", "D", "P"]
    plt.figure()
    for i, (kappa, bound) in enumerate(q_function_bounds.items()):
        optimal = q_function_optimal[kappa]
        x_range = np.arange(len(bound)) + 0.1 * (i - len(q_function_bounds) / 2 + 0.5)
        for j in range(len(bound)):
            plt.vlines(
                x=x_range[j], ymin=bound[j][0], ymax=bound[j][1], color="grey", linewidth=3.5, label="_nolegend_"
            )
        plt.plot(x_range, optimal, colors[i] + shapes[i], markersize=5)
        bound_marker = [colors[i] + "v", colors[i] + "^"]
        for k in reversed(range(2)):
            plt.plot(x_range, bound[:, k], bound_marker[k], markersize=4, label="_nolegend_")

    plt.xlabel("Action")
    plt.ylabel("Value Function")
    plt.title("Q-Function")
    plt.xticks(range(len(q_function_optimal[0])), [str(act.astype(int)) for act in action_space])
    plt.legend()
    plt.legend(q_function_bounds.keys())
    # Save the figure
    plt.savefig(os.path.join(dir, "q_function.png"), bbox_inches="tight", dpi=1200)
    plt.close()


def plot_v_function(v_function_bounds: list[list[np.ndarray]], v_function_optimal: list[list[float]], dir: str):
    # Convert q_function_bounds and q_function_optimal to numpy arrays
    v_function_bounds = np.array(v_function_bounds)
    v_function_optimal = np.array(v_function_optimal)

    bound_mean = np.mean(v_function_bounds, axis=0)
    bound_error = np.std(v_function_bounds, axis=0)
    optimal_mean = np.mean(v_function_optimal, axis=0)
    optimal_error = np.std(v_function_optimal, axis=0)

    plt.figure()
    for i in range(len(bound_mean)):
        plt.vlines(x=i, ymin=bound_mean[i][0], ymax=bound_mean[i][1], color="grey", linewidth=3.5)
    plt.errorbar(
        range(len(optimal_mean)),
        optimal_mean,
        yerr=optimal_error,
        color="k",
        linestyle="none",
        capsize=3,
        marker=None,
    )
    plt.plot(range(len(optimal_mean)), optimal_mean, "bs", markersize=5)
    bound_label = ["Lower Bound", "Upper Bound"]
    bound_marker = ["bv", "b^"]
    for i in reversed(range(2)):
        plt.errorbar(
            range(len(optimal_mean)),
            bound_mean[:, i],
            yerr=bound_error[:, i],
            color="k",
            linestyle="none",
            capsize=3,
            marker=None,
        )
        plt.plot(range(len(bound_mean[:, i])), bound_mean[:, i], bound_marker[i], markersize=4, label=bound_label[i])

    plt.xlabel("Time Step")
    plt.ylabel("Value Function")
    plt.title("Optimal Value Function")
    plt.xticks(range(len(optimal_mean)), range(len(optimal_mean)))
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(dir, "v_function.png"), bbox_inches="tight", dpi=1200)
