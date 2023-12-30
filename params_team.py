import numpy as np
import os

# Frontiers21 environments (allows exits)
# mdp_class = 'augmented_taxi'
# mdp_class = 'two_goal'
# mdp_class = 'skateboard'

# S22/IJCAI22 environments (doesn't allow exits)
mdp_class = 'augmented_taxi2'
# mdp_class = 'colored_tiles'
# mdp_class = 'skateboard2'
# mdp_class = 'two_goal2'

# misc (allows exits)
# mdp_class = 'taxi'
# mdp_class = 'cookie_crumb'

# mdp_class = 'augmented_navigation'

w_norm_order = 2
if mdp_class == 'augmented_taxi':
    w = np.array([[8.5, -3, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 4, 'y': 1, 'has_passenger': 0},
        'walls': [{'x': 1, 'y': 3}, {'x': 1, 'y': 2}],
        'passengers': [{'x': 4, 'y': 1, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}],
        'tolls': [{'x': 3, 'y': 1}],
        'available_tolls': [{"x": 2, "y": 3}, {"x": 3, "y": 3}, {"x": 4, "y": 3},
                   {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2},
                   {"x": 2, "y": 1}, {"x": 3, "y": 1}],
        'traffic': [],  # probability that you're stuck
        'fuel_station': [],
        'width': 4,
        'height': 3,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[1, 0, 0]]), np.array([[0, 0, -1]])]
    posterior = [np.array([[1, 0, 0]]), np.array([[0, -1, 0]]), np.array([[0, 0, -1]])]
    # posterior = [np.array([[1, 0, 4]]), np.array([[-1, 0, -13]]), np.array([[0, -1, 1]]), np.array([[0, 1, -5]])]  # +/- 50%
elif mdp_class == 'two_goal':
    w = np.array([[7.25, 10.5, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 3, 'y': 5},
        'goals': [{'x': 1, 'y': 1}, {'x': 5, 'y': 2}],
        'walls': [],
        'available_walls': [{'x': 1, 'y': 4}, {'x': 2, 'y': 4}, {'x': 3, 'y': 4}, {'x': 3, 'y': 2}, {'x': 4, 'y': 2},
                            {'x': 5, 'y': 3}],
        'width': 5,
        'height': 5,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, -1]])]
    posterior = [np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, -1]])]
    # posterior = [np.array([[1, 0, 3]]), np.array([[-1, 0, -11]]), np.array([[0, 1, 3]]),
    #              np.array([[0, -1, -16]])]  # +/- 50%
elif mdp_class == 'skateboard':
    w = np.array([[9, -0.3, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 4, 'y': 4, 'has_skateboard': 0},
        'skateboard': [{'x': 2, 'y': 3, 'on_agent': 0}],
        'goal': {'x': 6, 'y': 4},
        'walls': [],
        'available_walls': [{'x': 3, 'y': 4}, {'x': 3, 'y': 3}, {'x': 3, 'y': 2}, {'x': 2, 'y': 2}],
        'width': 7,
        'height': 4,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[1, 0, 0]]), np.array([[0, 0, -1]])]
    posterior = [np.array([[1, 0, 0]]), np.array([[0, -1, 0]]), np.array([[0, 0, -1]])]
    # posterior = [np.array([[1, 0, 4]]), np.array([[-1, 0, -14]]), np.array([[0, 1, -1]]),
    #              np.array([[0, -1, 0]])]  # +/- 50%
elif mdp_class == 'augmented_taxi2':
    w = np.array([[-3, 3.5, -1]]) # toll, hotswap station, step cost
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)
    # w_normalized = [[-0.63599873  0.74199852 -0.21199958]] # actual values fpr reference

    mdp_parameters = {
        'agent': {'x': 4, 'y': 1, 'has_passenger': 0},
        'walls': [{'x': 1, 'y': 3}, {'x': 1, 'y': 2}],
        'passengers': [{'x': 4, 'y': 1, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}],
        'tolls': [{'x': 3, 'y': 1}],
        'available_tolls': [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}],
        'traffic': [],  # probability that you're stuck
        'fuel_station': [],
        'hotswap_station': [{'x': 4, 'y': 3}],
        'available_hotswap_stations': [{'x': 4, 'y': 3}],
        'width': 4,
        'height': 3,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[0, 0, -1]])]
    posterior = [np.array([[-1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, -1]])]
elif mdp_class == 'two_goal2':
    w = np.array([[7.25, 10.5, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 3, 'y': 5},
        'goals': [{'x': 1, 'y': 1}, {'x': 5, 'y': 2}],
        'walls': [],
        'available_walls': [{'x': 1, 'y': 4}, {'x': 2, 'y': 4}, {'x': 3, 'y': 4}, {'x': 3, 'y': 2}, {'x': 4, 'y': 2},
                            {'x': 5, 'y': 3}],
        'width': 5,
        'height': 5,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, -1]])]
    posterior = [np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, -1]])]
elif mdp_class == 'skateboard2':
    w = np.array([[-0.175, -0.5125, -1]]) # skateboard (you might want to go backward to retrieve it), path
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 4, 'y': 4, 'has_skateboard': 0},
        'skateboard': [{'x': 1, 'y': 4, 'on_agent': 0}],
        'goal': {'x': 6, 'y': 4},
        'walls': [],
        'available_paths': [{'x': 1, 'y': 1}, {'x': 2, 'y': 1}, {'x': 3, 'y': 1}, {'x': 4, 'y': 1}, {'x': 5, 'y': 1}, {'x': 6, 'y': 1},
                 {'x': 6, 'y': 2}, {'x': 6, 'y': 3}],
        'paths': [{'x': 1, 'y': 1}, {'x': 2, 'y': 1}, {'x': 3, 'y': 1}, {'x': 4, 'y': 1}, {'x': 5, 'y': 1}, {'x': 6, 'y': 1},
                 {'x': 6, 'y': 2}, {'x': 6, 'y': 3}],
        'width': 6,
        'height': 4,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[0, 0, -1]])]
    posterior = [np.array([[-1, 0, 0]]), np.array([[0, -1, 0]]), np.array([[0, 0, -1]])]
elif mdp_class == 'colored_tiles':
    w = np.array([[-6.5, -5.25, -1]]) # A_tile (square), B_tile (ring), step cost
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 1, 'y': 1},
        'goal': {'x': 5, 'y': 1},
        'walls': [],
        'A_tiles': [],
        'available_A_tiles': [{'x': 2, 'y': 2}, {'x': 3, 'y': 2}, {'x': 4, 'y': 2}, {'x': 4, 'y': 1}],
        'B_tiles': [],
        'available_B_tiles': [{'x': 2, 'y': 4}, {'x': 3, 'y': 4}, {'x': 4, 'y': 4}, {'x': 5, 'y': 4}, {'x': 5, 'y': 3}],
        'width': 5,
        'height': 5,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[0, 0, -1]])]
    posterior = [np.array([[-1, 0, 0]]), np.array([[0, -1, 0]]), np.array([[0, 0, -1]])]
elif mdp_class == 'augmented_navigation':
    w = np.array([[-3, -7, 0.7, 3.5, 0.5, 0.8, -1]]) # gravel, grass, road, recharge, skateboard, car, step
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 1, 'y': 3, 'has_skateboard': 0, 'has_car': 0},
        'skateboard': [{'x': 1, 'y': 5, 'on_agent': 0}],
        'cars': [{'x': 1, 'y': 4, 'on_agent': 0}],
        'goal': {'x': 5, 'y': 1},
        'walls': [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}],
        'available_roads': [{'x': 1, 'y': 1}, {'x': 2, 'y': 1}, {'x': 3, 'y': 1}, {'x': 3, 'y': 5}, {'x': 4, 'y': 5},
                            {'x': 5, 'y': 5}],
        'roads': [{'x': 1, 'y': 1}, {'x': 2, 'y': 1}, {'x': 3, 'y': 1}, {'x': 3, 'y': 5}, {'x': 4, 'y': 5},
                            {'x': 5, 'y': 5}],
        'available_gravel': [{'x': 2, 'y': 2}, {'x': 2, 'y': 3}, {'x': 2, 'y': 4}, {'x': 2, 'y': 5}],
        'gravel': [{'x': 2, 'y': 2}, {'x': 2, 'y': 3}, {'x': 2, 'y': 4}, {'x': 2, 'y': 5}],
        'available_grass': [{'x': 4, 'y': 1}, {'x': 4, 'y': 2}, {'x': 4, 'y': 3}, {'x': 4, 'y': 4}],
        'grass': [{'x': 4, 'y': 1}, {'x': 4, 'y': 2}, {'x': 4, 'y': 3}, {'x': 4, 'y': 4}],
        'hotswap_station': [{'x': 3, 'y': 3}],
        'available_hotswap_stations': [{'x': 3, 'y': 3}],
        'width': 5,
        'height': 5,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    prior = [np.array([[0, 0, 0, 0, 0, 0, -1]])]
    posterior = [np.array([[-1, 0, 0, 0, 0, 0, 0]]), np.array([[0, -1, 0, 0, 0, 0, 0]]), np.array([[0, 0, 1, 0, 0, 0, 0]]),
                 np.array([[0, 0, 0, 1, 0, 0, 0]]), np.array([[0, 0, 0, 0, 1, 0, 0]]), np.array([[0, 0, 0, 0, 0, 1, 0]]),
                 np.array([[0, 0, 0, 0, 0, 0, -1]])]
elif mdp_class == 'cookie_crumb':
    w = np.array([[2.5, 1.7, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order),

    mdp_parameters = {
        'agent': {'x': 4, 'y': 1},
        'goals': [{'x': 4, 'y': 4}],
        'walls': [],
        'crumbs': [],
        'available_crumbs': [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 4}, {'x': 2, 'y': 2}, {'x': 2, 'y': 3}, {'x': 2, 'y': 4}],
        'width': 4,
        'height': 4,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }
# currently only compatible with making a single MDP through make_custom_MDP of make_mdp.py, whereas others can support
# making many MDPs by varying the available environment features (e.g. tolls, walls, crumbs)
elif mdp_class == 'taxi':
    # drop off reward, none, step cost
    w = np.array([[15, 0, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order)

    mdp_parameters = {
        'agent': {'x': 4, 'y': 1, 'has_passenger': 0},
        'walls': [{'x': 1, 'y': 3}, {'x': 2, 'y': 3}, {'x': 3, 'y': 3}],
        'passengers': [{'x': 1, 'y': 2, 'dest_x': 1, 'dest_y': 4, 'in_taxi': 0}],
        'width': 4,
        'height': 4,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }
else:
    raise Exception("Unknown MDP class.")

# Based on Pygame's key constants
keys_map = ['K_UP', 'K_DOWN', 'K_LEFT', 'K_RIGHT', 'K_p', 'K_d', 'K_r', 'K_u', 'K_9', 'K_0']

# reward weight parameters (on the goal with the passenger, on a toll, step cost).
# assume the L1 norm of the weights is equal 1. WLOG
weights = {
    'lb': np.array([-1., -1., -0.03125]),
    'ub': np.array([1., 1., -0.03125]),
    'val': w / np.linalg.norm(w[0, :], ord=w_norm_order)
}

weights_human = {
    'lb': np.array([-1., -1., -0.03125]),
    'ub': np.array([1., 1., -0.03125]),
    'val': np.array([[0.875, -0.5, -0.03125]])
}

# If true, code assumes that the last weight of a 1x3 weight vector is a known step cost and works in 2D Euclidean space
# If false, code assumes 1x3 weight vector (no weights are known) and works on the 2-sphere (in 3D)
step_cost_flag = False


# BEC parameters
BEC = {
    'summary_variant': 'proposed',            # [proposed, counterfactual_only, feature_only, baseline]
                                              # proposed: counterfactual (Y), feature scaffolding (Y)
                                              # feature_only: counterfactual (N), feature scaffolding (Y)
                                              # counterfactual_only: counterfactual (Y), feature scaffolding (N)
                                              # (Frontiers21) baseline: counterfactual (N), feature scaffolding (N)

    'n_train_demos': 8,                       # number of desired training demonstrations

    'n_test_demos': 30,                       # number of desired test demonstration

    'test_difficulty': 'high',                # expected ease for human to correctly predict the agent's actions in this test environment (low, medium, high)

    'n_human_models': 24,                     # number of human beliefs to actively consider; ensure that this is a multiple of the team size parameter below; needed for sampling models from joint team knowledge

    'n_particles': 500,                      # number of particles in particle filter

    'obj_func_proportion': 1,                 # proportion of the max objective function (i.e. info gain) to aim for
                                              # when selecting the next demonstration (range: 0 - 1). selecting a value
                                              # less than 1 may yield a greater number of demonstrations

    'BEC_depth': 1,                           # number of action deviations to consider when extracting BEC constraints

    'n_human_models_precomputed': 2500,         # number of human beliefs to precompute for future quick, real-time inference

}

data_loc = {
    'base': 'base',
    'BEC': mdp_class,
}

n_cpu = os.cpu_count()

# environment and trajectory indices of the tests used in Lee et al. 2022 Reasoning about Counterfactuals to Improve Human Inverse Reinforcement Learning
test_env_traj_tracers = {
    'augmented_taxi2':
        {
            'low': [(16, 17), (45, 110), (3, 81), (19, 176), (1, 149), (12, 58)],
            'medium': [(25, 82), (7, 7), (25, 88), (11, 105), (25, 13), (35, 105)],
            'high': [(13, 131), (31, 27), (10, 14), (13, 17), (13, 31), (47, 177)],
        },
    'colored_tiles':
        {
            'low': [(187, 16), (254, 7), (199, 6), (155, 14), (98, 20), (183, 21)],
            'medium': [(237, 8), (461, 18), (255, 4), (381, 8), (481, 8), (97, 19)],
            'high': [(455, 19), (75, 17), (335, 17), (431, 19), (382, 14), (349, 14)],
        },
    'skateboard2':
        {
            'low': [(7, 345), (5, 236), (5, 5), (1, 165), (3, 377), (1, 220)],
            'medium': [(5, 444), (7, 360), (5, 185), (1, 161), (5, 420), (1, 397)],
            'high': [(3, 363), (7, 74), (5, 96), (7, 84), (3, 29), (3, 394)],
        }
}


#######################################################

### Added for robot teaching to team situation

# for debugging
debug_flag = False
debug_kl_calc = False
debug_hm_sampling = False
plot_sampled_models_flag = False

team_size = 3

# human_learning_models = {'p1': np.array([0.9, 0.9, 0.9]), \
#                          'p2': np.array([0.8, 0.8, 0.8]), \
#                          'p3': np.array([0.7, 0.7, 0.7]) }


team_prior = {}
for i in range(team_size):
    member_id = 'p' + str(i+1)
    team_prior[member_id] = [prior]



# knowledge = 'common_knowledge'
# knowledge = 'joint_knowledge'
# knowledge = 'individual_knowledge'

# learning_goal = {'common_knowledge': np.array([0.5, 0.5, 0.5]), \
#                   'joint_knowledge': np.array([0.95, 0.95, 0.95]), \
#                   'individual_knowledge': np.array([0.7, 0.7, 0.7])}

# demo_strategy = 'common_knowledge'
# demo_strategy = 'joint_knowledge'
# demo_strategy = 'individual_knowledge_low'
# demo_strategy = 'individual_knowledge_high'

max_KC_loops = 20
max_loops = 50
learning_goal = 1.0
loop_threshold_demo_simplification = 100
default_learning_factor_teacher = 0.8
max_learning_factor = 1.0
