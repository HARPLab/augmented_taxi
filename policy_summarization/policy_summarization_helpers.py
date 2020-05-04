import numpy as np
import dill as pickle
import itertools
import shutil

# Other imports
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_helpers


def obtain_wt_candidates(data_loc, weights, weights_lb, weights_ub, n_wt_partitions=0, iter_idx=None):
    '''
    Args:
        weights: Ground truth weights (numpy array)
        weights_lb: Lowerbound for weights (numpy array)
        weights_ub: Upperbound for weights (numpy array)
        n_wt_partitions: Number of partitions to discretize weights into. 0 corresponds to only considering the ground truth weight
        iter_idx: Weight dimension to discretize over. If None, discretize uniformly over all dimensions

    Returns:
        wt_uniform_sample (numpy array)

    Summary:
        Return the discretized reward weight candidates
    '''

    try:
        with open('models/' + data_loc + '/wt_candidates.pickle', 'rb') as f:
            wt_uniform_sampling = pickle.load(f)
    except:
        # if a specific reward weight wasn't specified for discretization, uniformly discretize over all reward weights
        if iter_idx is None:
            mesh = np.array(np.meshgrid(
                *[np.linspace(weights_lb[x], weights_ub[x], n_wt_partitions) for x in np.arange(len(weights[0]))]))
            wt_uniform_sampling = np.hstack([mesh[x].reshape(-1, 1) for x in np.arange(len(weights[0]))])
            wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
            wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1,
                                                              wt_uniform_sampling.shape[1])  # for future dot products
        # uniformly discretize only over the desired reward weight dimension
        else:
            discretized_weights = np.linspace(weights_lb[iter_idx], weights_ub[iter_idx], n_wt_partitions)
            wt_uniform_sampling = np.tile(weights, (len(discretized_weights), 1))
            wt_uniform_sampling[:, iter_idx] = discretized_weights
            wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
            wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1,
                                                              wt_uniform_sampling.shape[1])

        with open('models/' + data_loc + '/wt_candidates.pickle', 'wb') as f:
            pickle.dump(wt_uniform_sampling, f)

    return wt_uniform_sampling

def generate_env(env_code):
    '''
    :param env_code: Vector representation of an augmented taxi environment (list of binary values)
    :return: Corresponding passenger and toll objects
    '''

    # first entry currently dictates where the passenger begins
    if env_code[0] == 0:
        requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
    else:
        requested_passenger = [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]

    # the last eight entries currently dictate the presence of tolls
    available_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
               {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 4, "y": 2, "fee": 1},
               {"x": 2, "y": 1, "fee": 1}, {"x": 3, "y": 1, "fee": 1}]

    requested_tolls = []

    offset = 1
    for x in range(offset, len(env_code)):
        entry = env_code[x]
        if entry:
            requested_tolls.append(available_tolls[x - offset])

    return requested_passenger, requested_tolls

def obtain_env_policies(data_loc, n_env, wt_candidates, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a):
    '''
    Summary: come up with an optimal policy for each of the candidates
    '''

    # generate codes that govern passenger's initial position and status of the eight tolls in the 4x3 environment
    env_codes = list(map(list, itertools.product([0, 1], repeat=int(np.log(n_env) / np.log(2)))))

    save_mark = 50
    try:
        with open('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'rb') as f:
            wt_vi_traj_candidates = pickle.load(f)

        if len(wt_vi_traj_candidates) == len(env_codes) and len(env_codes[-1]) == len(wt_candidates):
            # all environments and weights have been processed
            n_processed_envs = len(env_codes)
        else:
            # a portion of the environments and weights have been processed
            n_processed_envs = len(wt_vi_traj_candidates)
    except:
        wt_vi_traj_candidates = []
        n_processed_envs = 0

    # enumeration of all possible optimal policies from possible environments x weight candidates
    # if there are environments and weights yet to be processed
    if n_processed_envs < len(env_codes):
        for env_idx in range(n_processed_envs, len(env_codes)):
            env_code = env_codes[env_idx]
            passengers_a, tolls_a = generate_env(env_code)
            wt_counter = 0
            # a per-environment tuple of corresponding reward weight, optimal policy, and optimal trajectory
            wt_vi_traj_env = []
            for wt_candidate in wt_candidates:
                mdp_candidate = AugmentedTaxiOOMDP(width=width_a, height=height_a, agent=agent_a, walls=walls_a,
                                               passengers=passengers_a, tolls=tolls_a, traffic=traffic_a,
                                               fuel_stations=fuel_station_a, gamma=gamma_a, weights=wt_candidate)
                vi_candidate = ValueIteration(mdp_candidate, sample_rate=1)
                iterations, value_of_init_state = vi_candidate.run_vi()
                trajectory = mdp_helpers.rollout_policy(mdp_candidate, vi_candidate)
                wt_vi_traj_env.append((wt_candidate, vi_candidate, trajectory))

                wt_counter += 1
                print('wt_counter: {}, iterations: {}, init_val: {}, wt_candidate: {}'.format(wt_counter, iterations,
                                                                                       value_of_init_state,
                                                                                       wt_candidate))
            wt_vi_traj_candidates.append(wt_vi_traj_env)
            n_processed_envs += 1
            print('Finished analyzing environment {}'.format(n_processed_envs))

            if n_processed_envs % save_mark == 0:
                with open('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'wb') as f:
                    pickle.dump(wt_vi_traj_candidates, f)

                # make a backup in case the overwriting in the code above fails
                shutil.copy2('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'models/' + data_loc + '/wt_vi_traj_candidates_backup.pickle')

                print("Saved!")

        with open('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'wb') as f:
            pickle.dump(wt_vi_traj_candidates, f)

        # make a backup in case the overwriting in the code above fails
        shutil.copy2('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'models/' + data_loc + '/wt_vi_traj_candidates_backup.pickle')

    return wt_vi_traj_candidates