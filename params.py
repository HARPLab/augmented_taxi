import numpy as np

aug_taxi = {
    'agent': {'x': 4, 'y': 1, 'has_passenger': 0},
    'walls': [{'x': 1, 'y': 3, 'fee': 1}, {'x': 1, 'y': 2, 'fee': 1}],
    'passengers': [{'x': 4, 'y': 1, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}],
    'tolls': [{'x': 3, 'y': 1, 'fee': 1}],
    'traffic': [],  # probability that you're stuck
    'fuel_station': [],
    'width': 4,
    'height': 3,
    'gamma': 1
}

# w = np.array([[26, -5, -1]])
# w = np.array([[26, -4, -1]])
w = np.array([[26, -3, -1]])
# w = np.array([[26, -2, -1]])
# w = np.array([[26, -1, -1]])

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

# Joint BIRL and BEC parameters
n_env = 256                           # number of environments to consider
                                      # tip: select so that np.log(n_env) / np.log(2) yields an int for predictable
                                      # behavior see ps_helpers.obtain_env_policies()

# BEC parameters
BEC = {
    'summary_type': 'policy',           # demo or policy: whether constratints are extraced from just the optimal demo from the
                                      # starting state or from all possible states from the full policy
    'depth': 1,                       # number of suboptimal actions to take before following the optimal policy to obtain the
                                      # suboptimal trajectory (and the corresponding suboptimal expected feature counts)
    'n_test_demos': 1,                # number of desired test environments

    'n_train_demos': 4,

    'test_difficulty': 'hard',

    'min_BEC_set_only': False
}

# BIRL parameters
n_wt = 1                              # total number of weight candidates (including the ground truth)
                                      # tip: select n_wt to such that n_wt_partitions is an int to ensure that the exact
                                      # number of desired weight candidates is actually incorporated. see ps_helpers.discretize_wt_candidates()
                                      # also note that n_wt = n_wt_partitions ** (# of weights you're discretizing over) + 1
iter_idx = None                       # weight dimension to discretize over. If None, discretize uniformly over all dimensions

if iter_idx == None:
    data_loc_BIRL = str(n_env) + '_env/' + str(n_wt) + '_wt_' + 'uniform'
    if step_cost_flag:
        n_wt_partitions = int((n_wt - 1) ** (1.0 / (weights['val'].shape[1] - 1)))
    else:
        n_wt_partitions = int((n_wt - 1) ** (1.0 / weights['val'].shape[1]))
else:
    data_loc_BIRL = str(n_env) + '_env/' + str(n_wt) + '_wt_' + 'iter_idx_' + str(iter_idx)
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
    'BEC': str(n_env) + '_env',
    'BIRL': data_loc_BIRL
}