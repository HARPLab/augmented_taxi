import numpy as np
import os

# mdp_class = 'augmented_taxi'
# mdp_class = 'two_goal'
# mdp_class = 'skateboard'
# mdp_class = 'augmented_taxi2'
# mdp_class = 'two_goal2'
mdp_class = 'skateboard2'
# mdp_class = 'taxi'
# mdp_class = 'cookie_crumb'

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
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order),

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
    w = np.array([[-0.15, -0.5, -1]]) # skateboard (you might want to go backward to retrieve it), path
    w_normalized = w / np.linalg.norm(w[0, :], ord=w_norm_order),

    mdp_parameters = {
        'agent': {'x': 4, 'y': 4, 'has_skateboard': 0},
        'skateboard': [{'x': 1, 'y': 4, 'on_agent': 0}],
        'goal': {'x': 6, 'y': 4},
        'walls': [],
        'available_walls': [{'x': 2, 'y': 2}, {'x': 2, 'y': 3}, {'x': 4, 'y': 2}, {'x': 4, 'y': 3}],
        'paths': [{'x': 2, 'y': 1}, {'x': 3, 'y': 1}, {'x': 4, 'y': 1}, {'x': 5, 'y': 1}, {'x': 6, 'y': 1},
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
keys_map = ['K_UP', 'K_DOWN', 'K_LEFT', 'K_RIGHT', 'K_p', 'K_d', 'K_7', 'K_8', 'K_9', 'K_0']

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
    'summary_variant': 'proposed',            # [proposed, human_only, feature_only, baseline]
                                              # proposed: counterfactual (Y), feature scaffolding (Y)
                                              # feature_only: counterfactual (N), feature scaffolding (Y)
                                              # human_only: counterfactual (Y), feature scaffolding (N)
                                              # (Frontiers21) baseline: counterfactual (N), feature scaffolding (N)

    'n_train_demos': 15,                       # number of desired training demonstrations

    'n_test_demos': 30,                        # number of desired test demonstration

    'test_difficulty': 'medium',               # expected ease for human to correctly predict the agent's actions in this test environment (low, medium, high)

    'n_human_models': 8
}

data_loc = {
    'base': 'base',
    'BEC': mdp_class,
}

n_cpu = os.cpu_count()