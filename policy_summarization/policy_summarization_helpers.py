import numpy as np

def discretize_wt_candidates(weights, weights_lb, weights_ub, n_wt_partitions):
    '''
    Args:
        weights (numpy array)
        weights_lb (numpy array)
        weights_ub (numpy array)

    Returns:
        wt_uniform_sample (numpy array)

    Summary:
        Return the uniformly discretized reward weight candidates
    '''
    mesh = np.array(np.meshgrid(np.linspace(weights_lb[0], weights_ub[0], n_wt_partitions),
                    np.linspace(weights_lb[1], weights_ub[1], n_wt_partitions),
                    np.linspace(weights_lb[2], weights_ub[2], n_wt_partitions)))
    wt_uniform_sampling = np.hstack((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1), mesh[2].reshape(-1, 1)))
    wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
    wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1, wt_uniform_sampling.shape[1]) # for future dot products

    return wt_uniform_sampling

