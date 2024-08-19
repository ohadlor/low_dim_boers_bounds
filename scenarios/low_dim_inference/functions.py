import pickle
import os
from omegaconf import OmegaConf
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from scenarios.utils import Truncated2DGaussianPDF

from .beliefs import ParticleBelief
from .reward import LightDarkReward
from .factors import Truncated2DGaussianPairWiseLinearFactor, Truncated2DGaussianPriorFactor


def initial_belief(
    prior_mean: np.ndarray, prior_sigma: float, samples: int, rng: np.random.Generator
) -> ParticleBelief:
    """
    Initialize the agent's belief.

    Returns
    -------
    ParticleBelief
        The initial belief of the agent.
    """
    particles = rng.multivariate_normal(prior_mean, prior_sigma**2 * np.eye(len(prior_mean)), samples)
    weights = np.ones(samples) / samples
    return ParticleBelief(particles, weights, rng)


def single_step(
    belief: ParticleBelief,
    action: np.ndarray,
    action_noise: Truncated2DGaussianPDF,
    obs_noise: Truncated2DGaussianPDF,
    partial_simplifiction: float,
    sith_simplification: float | None,
) -> None:
    """
    Perform a single belief update step of the agent in the environment.

    Parameters
    ----------
    belief : ParticleBelief
        The current belief state of the agent.
    action : np.ndarray
        The action taken by the agent.
    action_noise : Truncated2DGaussianPDF
        The noise associated with the action.
    observation : np.ndarray
        The observation received by the agent.
    obs_noise : Truncated2DGaussianPDF
        The noise associated with the observation.
    simplification_factor : float
        A factor used to simplify the reward calculation.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "estimator": The Boers estimator of the reward.
        - "partial_bounds": A tuple containing the lower and upper bounds of the partial expectation.
        - "sith_bounds": A tuple containing the lower and upper bounds of the SITH expectation.
    belief : ParticleBelief
        The updated belief state of the agent.
    """
    belief.propogate(action, action_noise)
    belief.update(obs_noise)
    transition_factor = Truncated2DGaussianPairWiseLinearFactor(action, action_noise)
    observation_factor = Truncated2DGaussianPriorFactor(obs_noise)
    reward = LightDarkReward(
        transition_factor,
        observation_factor,
        belief=belief,
        partial_simplification=partial_simplifiction,
        sith_simplification=sith_simplification,
    )
    boers_estimator, boers_time = reward.boers_estimator()
    partial_bounds, partial_time = reward.partial_expectation_bounds()
    sith_bounds, sith_time = reward.sith_bounds()
    return {
        "estimator": {"value": boers_estimator, "time": boers_time},
        "partial": {"value": partial_bounds, "time": partial_time},
        "sith": {"value": sith_bounds, "time": sith_time},
    }, belief


def simulation_loop(
    action_sequence: np.ndarray,
    samples: int,
    simplification_factors: tuple[float, float | None],
    prior_mean: np.ndarray,
    prior_sigma: float,
    action_noise: Truncated2DGaussianPDF,
    rng: np.random.Generator,
    results_path: str,
    resample: bool = False,
) -> None:
    """
    Performs a specified action sequence with given parameters and returns the result.
    Saves results to pickle files.

    Parameters
    ----------
    action_sequence : np.ndarray
        An array representing the sequence of actions to be performed.
    samples : int
        The number of samples in the belief.
    simplification_factors : tuple[float, float | None]
        Simplification factors for partial bounds and sith bounds respectively. If sith simplification is
        None, it is set to the partial simplification.
    prior_mean : np.ndarray
        The mean values for the prior distribution.
    prior_sigma : float
        The standard deviation for the prior distribution.
    action_noise : Truncated2DGaussianPDF
        A truncated 2D Gaussian probability density function representing the action noise.
    rng : np.random.Generator
        A random number generator instance for reproducibility.
    resample : bool, optional
        A flag indicating whether to resample the belief, by default False
    """
    belief = initial_belief(prior_mean, prior_sigma, samples, rng)
    observation = prior_mean
    for t, action in enumerate(action_sequence):
        observation += action
        action_std = action_noise.std
        obs_std = action_noise.std * 0.7 + 0.01 * t
        obs_noise = Truncated2DGaussianPDF(observation, obs_std, 1.5 * (1 + action_std / obs_std) * obs_std, rng)
        assert obs_noise.Min > 0
        results, belief = single_step(belief, action, action_noise, obs_noise, *simplification_factors)
        belief = belief.copy()
        if resample:
            belief.resample(N=samples)
        log_results(results, t, results_path)


def log_results(results: dict[str, dict], time_step: int, path: str) -> None:
    """Logs the results to a pickle file named with the given time step.

    Parameters
    ----------
    results : dict
        The results to be logged.
    time_step : int
        The current time step used to name the pickle file.
    """
    with open(path + f"/results_{time_step}.pkl", "ab") as file:
        pickle.dump(results, file)


def load_results(path: str) -> list[list[dict]]:
    """Loads the results of a given run from the given path.

    Parameters
    ----------
    path : str
        The path to the pickle files.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the results.
    """
    run_results = []
    for dirpath, _, filenames in os.walk(path):
        if not filenames:
            continue
        iter_results = [None] * len(filenames)
        for filename in filenames:
            if filename.endswith(".pkl"):
                with open(os.path.join(dirpath, filename), "rb") as file:
                    t = int(file.name.split("_")[-1].split(".")[0])
                    iter_results[t] = pickle.load(file)
        run_results.append(iter_results)
    return run_results


def plot_results(results: Sequence[Sequence[dict]], path: str) -> None:
    iters = len(results)
    steps = len(results[0])

    boers_estimator = np.zeros((iters, steps))
    partial_lower = np.zeros((iters, steps))
    partial_upper = np.zeros((iters, steps))
    sith_lower = np.zeros((iters, steps))
    sith_upper = np.zeros((iters, steps))
    boers_time = np.zeros((iters, steps))
    partial_time = np.zeros((iters, steps))
    sith_time = np.zeros((iters, steps))

    for i, result in enumerate(results):
        for t, values in enumerate(result):
            boers_estimator[i, t] = values["estimator"]["value"]
            partial_lower[i, t], partial_upper[i, t] = values["partial"]["value"]
            sith_lower[i, t], sith_upper[i, t] = values["sith"]["value"]
            boers_time[i, t] = values["estimator"]["time"]
            partial_time[i, t] = values["estimator"]["time"] / values["partial"]["time"]
            sith_time[i, t] = values["estimator"]["time"] / values["sith"]["time"]

    boers_estimator_std = np.std(boers_estimator, axis=0)
    partial_lower_std = np.std(partial_lower - boers_estimator, axis=0)
    partial_upper_std = np.std(partial_upper - boers_estimator, axis=0)
    sith_lower_std = np.std(sith_lower - boers_estimator, axis=0)
    sith_upper_std = np.std(sith_upper - boers_estimator, axis=0)
    boers_time_std = np.std(boers_time)
    partial_time_std = np.std(partial_time)
    sith_time_std = np.std(sith_time)

    boers_estimator = np.mean(boers_estimator, axis=0)
    partial_lower = np.mean(partial_lower, axis=0)
    partial_upper = np.mean(partial_upper, axis=0)
    sith_lower = np.mean(sith_lower, axis=0)
    sith_upper = np.mean(sith_upper, axis=0)
    boers_time = np.mean(boers_time)
    partial_time = np.mean(partial_time)
    sith_time = np.mean(sith_time)

    # Plot the results
    plt.figure()
    time_steps = np.arange(steps)

    # Plot Boers estimator with std margins
    plt.plot(time_steps, boers_estimator, label="Boers Estimator", color="black")
    plt.fill_between(
        time_steps,
        boers_estimator - boers_estimator_std,
        boers_estimator + boers_estimator_std,
        color="black",
        alpha=0.2,
    )

    # Plot partial expectation bounds with std margins
    plt.plot(time_steps, partial_lower, label="Partial Expectation Lower Bound", color="orange")
    plt.plot(time_steps, partial_upper, label="Partial Expectation Upper Bound", color="red")
    plt.fill_between(
        time_steps,
        partial_lower - partial_lower_std,
        partial_lower + partial_lower_std,
        color="orange",
        alpha=0.2,
    )
    plt.fill_between(
        time_steps,
        partial_upper - partial_upper_std,
        partial_upper + partial_upper_std,
        color="red",
        alpha=0.2,
    )

    # Plot SITH bounds with std margins
    plt.plot(time_steps, sith_lower, label="SITH Lower Bound", color="green")
    plt.plot(time_steps, sith_upper, label="SITH Upper Bound", color="blue")
    plt.fill_between(
        time_steps,
        sith_lower - sith_lower_std,
        sith_lower + sith_lower_std,
        color="green",
        alpha=0.2,
    )
    plt.fill_between(
        time_steps,
        sith_upper - sith_upper_std,
        sith_upper + sith_upper_std,
        color="blue",
        alpha=0.2,
    )

    plt.xlabel("Time Step")
    plt.ylabel("Reward value")
    # plt.title("Boers Estimator")
    plt.legend()
    # plt.grid(True)
    plt.savefig(path + "/results_plot.png", dpi=1200)

    with open(path + "/times.txt", "w") as file:
        file.write("Average time per time step:\n")
        file.write(f"Boers Time: {boers_time} ± {boers_time_std}\n")
        file.write(f"Partial Speed up: {partial_time} ± {partial_time_std}\n")
        file.write(f"SITH Speed up: {sith_time} ± {sith_time_std}\n")


def run_with_parameters(run_function, parameter_path=None):
    seed = 163
    rng = np.random.default_rng(seed=seed)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    if parameter_path is not None:
        absolute_path = os.path.join(file_dir + parameter_path, "params.yaml")
    else:
        absolute_path = os.path.join(file_dir, "params.yaml")
    params = OmegaConf.load(absolute_path)

    experiment_sets = params.experiment_sets
    global_common_params = params.global_common_parameters

    for set_content in experiment_sets.values():
        set_common_params = OmegaConf.merge(global_common_params, set_content.common_parameters)

        for run in set_content.runs:
            run_params = OmegaConf.merge(set_common_params, run)
            run_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            N = run_params.n_particles
            p_s = run_params.partial_simplification
            s_s = run_params.sith_simplification
            run_folder = f"/results/{N}_{p_s}_{s_s}".replace(".", "")
            results_path = os.path.realpath(run_path + run_folder)
            if os.path.exists(results_path):
                continue

            for _ in range(run_params["iterations"]):
                run_function(**run_params, rng=rng)
            results = load_results(results_path)
            plot_results(results, results_path)
