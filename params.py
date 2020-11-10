import numpy as np

mdp_class = 'augmented_taxi'
# mdp_class = 'two_goal'
# mdp_class = 'skateboard'
# mdp_class = 'taxi'
# mdp_class = 'cookie_crumb'

if mdp_class == 'augmented_taxi':
    w = np.array([[26, -3, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=1)

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
elif mdp_class == 'two_goal':
    w = np.array([[7.25, 10.5, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=1)

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
elif mdp_class == 'skateboard':
    w = np.array([[9, -0.3, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=1),

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
elif mdp_class == 'cookie_crumb':
    w = np.array([[2.5, 1.7, -1]])
    w_normalized = w / np.linalg.norm(w[0, :], ord=1),

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
    w_normalized = w / np.linalg.norm(w[0, :], ord=1)

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
    'val': w / np.linalg.norm(w[0, :], ord=1)
}

weights_human = {
    'lb': np.array([-1., -1., -0.03125]),
    'ub': np.array([1., 1., -0.03125]),
    'val': np.array([[0.875, -0.5, -0.03125]])
}


step_cost_flag = True    # indicates that the last weight element is a known step cost. code currently assumes a 2D
                         # weight vector if step_cost_flag = False, and a 3D weight vector if step_cost_flag = True


# BEC parameters
BEC = {
    'summary_type': 'policy',                 # demo or policy: whether constratints are extraced from just the optimal demo from the
                                              # starting state or from all possible states from the full policy

    'summary_variant': ['medium', 'high'],   # [{'low', 'medium', 'high', 'highest', 'forward', 'backward}, {low', 'high'}]
                                              # [{expected information transfer to a perfect IRL agent, or scaffolding},
                                              # {ease metrics (visual similarity, visual simplicity, etc)}]

    'n_train_demos': 5,                       # number of desired training demonstrations

    'n_test_demos': 30,                        # number of desired test demonstration

    'depth': 1,                               # number of suboptimal actions to take before following the optimal policy to obtain the
                                              # suboptimal trajectory (and the corresponding suboptimal expected feature counts)

    'test_difficulty': 'medium'                 # expected ease for human to correctly predict the agent's actions in this test environment (low, medium, high)
}

# BIRL parameters
n_wt = 1                                      # total number of weight candidates (including the ground truth)
                                              # tip: select n_wt to such that n_wt_partitions is an int to ensure that the exact
                                              # number of desired weight candidates is actually incorporated. see ps_helpers.discretize_wt_candidates()
                                              # also note that n_wt = n_wt_partitions ** (# of weights you're discretizing over) + 1
iter_idx = None                               # weight dimension to discretize over. If None, discretize uniformly over all dimensions


if iter_idx == None:
    data_loc_BIRL = str(n_wt) + '_wt_' + 'uniform'
    if step_cost_flag:
        n_wt_partitions = int((n_wt - 1) ** (1.0 / (weights['val'].shape[1] - 1)))
    else:
        n_wt_partitions = int((n_wt - 1) ** (1.0 / weights['val'].shape[1]))
else:
    data_loc_BIRL = str(n_wt) + '_wt_' + 'iter_idx_' + str(iter_idx)
    n_wt_partitions = n_wt - 1

BIRL = {
    'n_wt': n_wt,
    'iter_idx': iter_idx,
    'eval_fn': 'approx_MP',             # desired likelihood function for computing the posterior probability of weight candidates
    'n_demonstrations': 10,             # total number of demonstrations sought, in order of decreasing effectiveness
    'n_wt_partitions': n_wt_partitions
}

data_loc = {
    'base': 'base',
    'BEC': mdp_class,
    'BIRL': mdp_class + '/' + data_loc_BIRL
}