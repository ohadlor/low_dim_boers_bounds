import numpy as np

from code.factor_graph.factor_graph_slices import SlicesFactorGraph

from scenarios.light_dark_low_dim.environment import LightDark2D
from scenarios.light_dark_low_dim.agent import LightDarkAgent
from scenarios.light_dark_low_dim.basic_classes import LightDarkActions
from scenarios.light_dark_low_dim.planner import Planner
from scenarios.light_dark_low_dim.reward import LightDarkReward
from scenarios.light_dark_low_dim.beliefs import SlicesBelief
from scenarios.light_dark_low_dim.tree_node import ObservationNode
from scenarios.light_dark_low_dim.noise import ActionNoise, ObservationNoise
from scenarios.light_dark_low_dim.functions import spare_sampling_tree
from scenarios.utils.functions import distance


def run():
    def create_factor_graph() -> SlicesFactorGraph:
        fg = SlicesFactorGraph()
        fg.add_node()
        return fg

    rng = np.random.Generator
    action_disances = [0.2, 0.4, 0.6, 0.8, 1.0]
    action_partions = 10
    N_BEACONS = 10
    STATE_SPACE = np.array((20, 20))
    N_OBSTACLES = 0
    start_location = np.array((0.5, 0.5))
    goal_location = np.array((19, 19))
    max_tree_depth = 3
    delta = 0.5

    env = LightDark2D(rng, STATE_SPACE, N_BEACONS, N_OBSTACLES)
    actions = LightDarkActions(distances=action_disances, partitions=action_partions)
    initial_pose = start_location
    pose_prior = None
    planner = Planner(LightDarkReward())
    initial_belief = SlicesBelief(create_factor_graph())

    true_agent = LightDarkAgent(
        action_noise=ActionNoise,
        observation_noise=ObservationNoise,
        action_space=LightDarkActions,
        belief=initial_belief,
        start_location=initial_pose,
    )

    root_node = ObservationNode(initial_belief, depth=0, is_root=True)
    root_node.agent_path = [true_agent.path[-1]]
    while distance(true_agent.path[-1], goal_location) > delta:
        virtual_agent = true_agent.copy()
        spare_sampling_tree(
            env, virtual_agent, actions, max_depth=max_tree_depth, n_observations=100
        )
        planner(root_node)
        next_action = planner.best_action
        true_agent.move(next_action)
