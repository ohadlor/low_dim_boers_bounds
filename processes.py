import os
import time

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from core import (
    Reward,
    Agent,
    CircleActions,
    ParticleBelief,
    Truncated2DGaussianPDF,
    Environment,
    MultiVariateGaussianPDF,
)
from utils.utilities import set_landmarks, step, get_q_function, get_v_function
from utils.logging import save_data, load_results
from utils.stopping_conditions import StoppingConditions

from utils.visualization import ProblemPlotter, plot_q_function, plot_v_function, plot_speed_up


@hydra.main(version_base=None, config_path="conf", config_name="config")
def simulator(cfg: DictConfig) -> None:
    """run the simulation with the given hyperparameters, logs results to directory
    results/kappa/current_time/iter_i

    Parameters
    ----------
    inference_samples : int, optional
        number of state samples used in the inference engine, by default 150
    reward_samples : int, optional
        number of state samples used for the empirical reward, by default 100
    observation_samples : int, optional
        number of observation nodes opened per action node, by default 150
    kappa : float, optional
        simplification factor of the data association space, by default 0.7
    discount_factor : float, optional
        reward discount factor, in [0,1], by default 0.95
    information_weight : float, optional
        weight of the information reward relative to state reward, in [0,1], by default 1.0
    iterations : int, optional
        number of experiments run, by default 3
    planning_horizon : int, optional
        planning horizon of belief tree, by default 1
    obs_std : float, optional
        observation model std before truncation, by default 0.2
    obs_range : float, optional
        observation model truncation radius, by default 1.5
    agent_obs_range : float, optional
        range at which agent will get observation, by default 1.5
    action_std : float, optional
        transition model std, by default 0.3
    stopping_condition : callable, optional
        simulation stopping condition, by default None
    """
    # get hyperparameters from config
    cfg = cfg.process

    seed = cfg.simulation_params.seed
    iterations = cfg.simulation_params.iterations

    prior_landmarks_coords = np.array(cfg.landmarks.prior_landmarks)
    landmarks_per_bunch = cfg.landmarks.landmarks_per_bunch

    # detection range should be much smaller than observation range such that most particles will share observations
    detection_range = cfg.noise.detection_range
    obs_std = cfg.noise.obs_std
    obs_range = cfg.noise.obs_range
    action_std = cfg.noise.action_std
    initial_std = cfg.noise.initial_std

    planning_horizon = cfg.planning.planning_horizon
    state_samples = cfg.planning.state_samples
    kappas = cfg.planning.kappa
    discount_factor = cfg.planning.discount_factor
    information_weight = cfg.planning.information_weight
    action_partitions = cfg.planning.action_partitions
    action_radius = cfg.planning.action_radius

    stopping_condition = cfg.stopping.stopping_condition
    goal_location = np.array(cfg.stopping.goal)
    max_steps = cfg.stopping.max_steps
    timeout = cfg.stopping.timeout

    start_location = np.array([0, 0])

    # set simulation parameters
    rng = np.random.default_rng(seed)

    # set landmarks
    landmarks = set_landmarks(prior_landmarks_coords, landmarks_per_bunch)

    # set action space
    action_space = CircleActions(action_partitions, action_radius)

    # define initial particle belief
    particles = rng.multivariate_normal(mean=np.array([0, 0]), cov=initial_std**2 * np.eye(2), size=state_samples)
    initial_belief = ParticleBelief(particles=particles, weights=np.ones(state_samples) / state_samples, rng=rng)
    setattr(ParticleBelief, "full_n", state_samples)

    # define noise
    observation_noise = Truncated2DGaussianPDF(mean=np.array([0, 0]), std=obs_std, range=obs_range, rng=rng)
    action_noise = MultiVariateGaussianPDF(mean=np.array([0, 0]), cov=action_std**2 * np.eye(2), rng=rng)

    # Define environment
    environment = Environment(landmarks, observation_noise, detection_range, rng)

    # set reward function
    reward = Reward(
        goal_location=goal_location,
        transition_noise=action_noise,
        obs_noise=observation_noise,
        discount_factor=discount_factor,
        information_weight=information_weight,
    )

    # Create directory for results
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("results", current_time)
    # Save the hydra config file to log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    # start iterations of simulation
    for iteration in range(iterations):
        iter_dir = os.path.join(log_dir, f"iter_{iteration}")
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)
        agent = Agent(
            action_noise=action_noise,
            observation_noise=observation_noise,
            belief=initial_belief.copy(),
            start_location=start_location,
            agent_observation_range=detection_range,
            rng=rng,
        )

        stopper = StoppingConditions(max_steps=max_steps, goal_location=goal_location, timeout=timeout)
        i = 0
        total_time = 0

        problem_plotter = ProblemPlotter()
        problem_plotter.add(environment=environment, goal=goal_location, belief=initial_belief, start=start_location)

        while not stopper.is_stopped(stopping_condition):
            t_0 = time.perf_counter()

            step_data = step(
                agent.belief.copy(),
                environment,
                action_noise,
                action_space,
                reward,
                planning_horizon,
                simplification_factors=kappas,
            )

            # move gt agent
            optimal_action = action_space[np.argmax(step_data[None].root_q_function)]
            agent.move_and_update_agent_belief(optimal_action, landmarks)
            print(step_data)
            problem_plotter.add(action=agent.path[-2:], belief=agent.belief, location=agent.path[-1])

            # update stopping variables
            agent_location = agent.path[-1]
            i += 1
            t_1 = time.perf_counter()
            delta_t = t_1 - t_0
            total_time += delta_t

            stopper.update_stopping_variables({"max_steps": i, "time_out": total_time, "goal_reached": agent_location})
            # save results
            save_data(
                {
                    "groud truth location": agent_location,
                    "belief": agent.belief,
                    "results": step_data,
                },
                directory=iter_dir,
                checkpoint_num=i,
            )
        problem_plotter.save(iter_dir)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def post_proccess(cfg: DictConfig) -> None:
    action_space = CircleActions(cfg.process.planning.action_partitions, cfg.process.planning.action_radius)
    cfg = cfg.post_process

    data_dir = cfg.data_dir

    # iteration to plot q_function for
    iter = cfg.iter
    # time step to plot q_function for
    time_step = cfg.time_step

    # plot data
    q_function, q_function_bounds = get_q_function(data_dir, iter, time_step)
    plot_q_function(q_function, q_function_bounds, data_dir, action_space, time_step)

    v_function, v_function_bounds = get_v_function(data_dir, iter)
    plot_v_function(v_function, v_function_bounds, data_dir)

    results = load_results(data_dir, time_step)
    plot_speed_up(results, data_dir)


if __name__ == "__main__":
    simulator()
    # post_proccess()
