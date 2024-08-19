import time
import os

import numpy as np

from slicesInference.distributions import MultiVariateGaussianPDF

from scenarios.light_dark_high_dim import (
    LightDarkReward,
    LightDarkAgent,
    UnitCircleActions,
    SlicesBelief,
    create_prior_factor_graph,
    simulation_loop,
    create_beacons,
    load_checkpoint,
    define_landmark_covs,
    results_to_result,
    results_to_text,
    plot_q_functions,
)
from scenarios.utils import (
    # distance,
    Truncated2DGaussianPDF,
    ProblemPlotter,
)


def run():
    # set simulation parameters
    seed = 6314
    rng = np.random.default_rng(seed)

    start_location = np.array([[0, 0]])
    # Between 0 and 1, 0 means no simplification
    action_partitions = 4

    # set action space
    action_space = UnitCircleActions(action_partitions)
    # action_space = [np.array([0, 1])]

    # set landmarks
    n_landmarks = 9
    landmarks = create_beacons(n_landmarks, rng)

    # plotter = ProblemPlotter()
    # plotter.plot_landmarks(landmarks, 1.3)
    # plotter.plot_start_goal(start_location, goal_location)
    # plotter.show()

    # Create factor graph belief
    slices_fg = create_prior_factor_graph(landmarks)
    # samples for inference
    slices_fg.n_samples = 150
    initial_belief = SlicesBelief(slices_fg)
    initial_belief.inference()
    # initial_belief is now a bayes net with marginals

    # define noise
    mean = np.array([0, 0])
    obs_std = 0.2
    action_std = 0.3
    observation_noise = Truncated2DGaussianPDF(mean=mean, std=obs_std, range=obs_std * 3.5, rng=rng)
    action_noise = MultiVariateGaussianPDF(mean=np.array([0, 0]), cov=action_std**2 * np.eye(2))
    if observation_noise.Normalizer == 1:
        raise ValueError("Observation noise is not truncated")

    # set reward function
    kappa = 0.7
    reward = LightDarkReward(
        goal_location=None,
        observation_noise=observation_noise,
        discount_factor=0.95,
        bound_simplification_factor=kappa,
        information_weight=1,
        reward_samples=100,
    )

    # Create directory for results
    # dir = os.path.basename(__file__)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("results", f"{kappa}".replace(".", "_"), current_time)

    ITERS = 3
    for iteration in range(ITERS):
        iter_dir = os.path.join(log_dir, f"iter_{iteration}")
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)
        true_agent = LightDarkAgent(
            action_noise=action_noise,
            observation_noise=observation_noise,
            belief=initial_belief.copy(),
            start_location=start_location,
            agent_observation_range=1.3,
        )

        # delta = 1
        # def stopping_condition(loc):
        #     return distance(loc, goal_location) <= delta

        # timeout = 60 * 5

        # def stopping_condition(time_taken):
        #     return time_taken > timeout

        max_step = 3

        def stopping_condition(step):
            return step == max_step

        # simulation loop
        simulation_loop(
            true_agent=true_agent,
            landmarks=landmarks,
            action_space=action_space,
            reward=reward,
            stopping_condition=stopping_condition,
            N_O=150,
            max_tree_depth=1,
            directory=iter_dir,
        )


def post_process():
    if False:
        paths = {}
        paths[0] = r"results/0/2024-06-12_10-01-02"
        paths[0.5] = r"results/0_5/2024-06-12_09-10-23"
        paths[0.7] = r"results/0_7/2024-06-12_15-36-00"
        paths[1] = r"results/1/2024-06-12_10-47-25"
        iter_check_for_plot = {}
        iter_check_for_plot[0] = (0, 0)
        iter_check_for_plot[0.5] = (0, 1)
        iter_check_for_plot[0.7] = (0, 0)
        iter_check_for_plot[1] = (0, 0)
        q_function = {}
        q_function_bounds = {}
        for kappa, path in paths.items():
            iters, check = iter_check_for_plot[kappa]
            file = f"iter_{iters}/checkpoint_{check}"
            results = load_checkpoint(path, file)
            results = results["results"]
            q_function[kappa] = results["q_function_optimal"][-1]
            q_function_bounds[kappa] = results["q_function_bounds"][-1]
            # results_to_text(result, path)
        plot_q_functions(q_function_bounds, q_function, UnitCircleActions(4), "results")

    # Plot problem
    if True:
        path = r"results/0_5/2024-06-12_09-10-23/iter_2"
        file = "checkpoint_2"
        results = load_checkpoint(path, file)
        agent = results["true_agent"]
        results = results["results"]
        seed = 6314
        rng = np.random.default_rng(seed)
        agent_path = np.array([])
        for loc in agent.path:
            agent_path = np.append(agent_path, loc.flatten())
        agent_path = agent_path.reshape((-1, 2))
        bound_actions = results["bound_actions"]
        start_location = np.array([[0, 0]])

        # set landmarks
        n_landmarks = 9
        landmarks = create_beacons(n_landmarks, rng)
        covs = define_landmark_covs(landmarks)

        plotter = ProblemPlotter()
        plotter.plot_landmarks(landmarks, 1.3, covs, start_location)
        plotter.plot_start_goal(start_location)
        plotter.plot_actions(agent_path, bound_actions)
        plotter.save(os.path.join(path, "actions"))
