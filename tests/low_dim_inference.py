import numpy as np

from scenarios.utils import Truncated2DGaussianPDF
from scenarios.low_dim_inference import simulation_loop
import os


def run(
    n_steps: int = 30,
    n_particles: int = 100,
    partial_simplification: float = 0.5,
    sith_simplification: float = 0.5,
    resample=False,
    rng: np.random.Generator = np.random.default_rng(),
    **kwargs,
) -> None:
    rng = rng

    # Parameters
    action_sequence = np.array([[1, 1]]).repeat(n_steps, axis=0)
    action_std = 0.8
    action_noise = Truncated2DGaussianPDF(np.zeros(2), action_std, 3 * action_std, rng)
    prior_mean = np.array([0, 0])
    prior_sigma = 1.0

    # Create results folder
    run_folder = f"{n_particles}_{partial_simplification}_{sith_simplification}".replace(".", "")
    results_dir = "c:/Users/Ohad/Documents/Masters/Repositories/Partial_Expectation/Thesis/src/results"
    results_path = f"{results_dir}/{run_folder}"
    i = 0
    while os.path.exists(f"{results_path}/{i}"):
        i += 1
    results_path = f"{results_path}/{i}"
    os.makedirs(results_path)

    # Run simulation
    simulation_loop(
        action_sequence=action_sequence,
        samples=n_particles,
        simplification_factors=(partial_simplification, sith_simplification),
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
        action_noise=action_noise,
        rng=rng,
        results_path=results_path,
        resample=resample,
    )


def post_process():
    pass
