import os
import pickle

import numpy as np

from .data_classes import ResultsData, TimeStepRewardData


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


def load_results(results_path: str, time_step: int) -> ResultsData:
    results = []
    for file in os.listdir(results_path):
        if not file.startswith("iter"):
            continue
        iter_path = os.path.join(results_path, file)
        checkpoint_data = load_checkpoint(iter_path, f"checkpoint_{time_step}")
        results.append(checkpoint_data["results"])
    results = process_data(results)
    return results


def process_data(total_data: list[TimeStepRewardData]) -> ResultsData:
    # number of iterations
    n = len(total_data)
    # number of simplifications
    m = len(total_data[0])
    times = np.empty((n, m), dtype=float)
    factors = np.empty(m, dtype=int)
    factors_removed = np.empty((n, m), dtype=int)
    n_da = np.empty(m, dtype=int)
    nodes_removed = np.empty((n, m), dtype=int)
    node_elimination_rates = np.empty((n, m), dtype=float)
    factor_elimination_rates = np.empty((n, m), dtype=float)

    simplifications = total_data[0].keys()
    for i, iter_data in enumerate(total_data):
        for j, data in enumerate(iter_data.values()):
            nodes_counter = data.nodes_counter
            times[i, j] = data.time
            factors_removed[i, j] = nodes_counter.factors_eliminated
            nodes_removed[i, j] = nodes_counter.nodes_removed
            node_elimination_rates[i, j] = nodes_counter.node_elimination_rate
            factor_elimination_rates[i, j] = nodes_counter.factor_elimination_rate

            if i == 0:
                factors[j] = data.nodes_counter.factors_eliminated
                n_da[j] = data.n_da_nodes

    time_data = (times.mean(axis=0), times.std(axis=0))
    speed_up = times[:, 0] / times
    speed_up = (speed_up.mean(axis=0), speed_up.std(axis=0))
    factors_removed = (factors_removed.mean(axis=0), factors_removed.std(axis=0))
    nodes_removed = (nodes_removed.mean(axis=0), nodes_removed.std(axis=0))
    node_elimination_rates = (node_elimination_rates.mean(axis=0), node_elimination_rates.std(axis=0))
    factor_elimination_rates = (factor_elimination_rates.mean(axis=0), factor_elimination_rates.std(axis=0))

    results = ResultsData(
        simplifications,
        time_data,
        speed_up,
        factors,
        n_da,
        factors_removed,
        nodes_removed,
        node_elimination_rates,
        factor_elimination_rates,
    )
    return results
