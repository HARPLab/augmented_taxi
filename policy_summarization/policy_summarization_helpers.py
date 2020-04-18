import numpy as np

def discretize_wt_candidates(weights, weights_lb, weights_ub, n_wt_partitions, iter_idx=None):
    '''
    Args:
        weights (numpy array)
        weights_lb (numpy array)
        weights_ub (numpy array)

    Returns:
        wt_uniform_sample (numpy array)

    Summary:
        Return the discretized reward weight candidates
    '''
    # if a specific reward weight wasn't specified for discretization, uniformly discretize over all reward weights
    if iter_idx is None:
        mesh = np.array(np.meshgrid(*[np.linspace(weights_lb[x], weights_ub[x], n_wt_partitions) for x in np.arange(len(weights[0]))]))
        wt_uniform_sampling = np.hstack([mesh[x].reshape(-1, 1) for x in np.arange(len(weights[0]))])
        wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
        wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1, wt_uniform_sampling.shape[1]) # for future dot products
    # uniformly discretize only over the desired reward weight dimension
    else:
        discretized_weights = np.linspace(weights_lb[iter_idx], weights_ub[iter_idx], n_wt_partitions)
        wt_uniform_sampling = np.tile(weights, (len(discretized_weights), 1))
        wt_uniform_sampling[:, iter_idx] = discretized_weights
        wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
        wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1, wt_uniform_sampling.shape[1])

    return wt_uniform_sampling

