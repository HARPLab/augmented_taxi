import numpy as np
import dill as pickle
import itertools
import shutil
import random
from termcolor import colored

# Other imports
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_helpers

def sample_wt_candidates(data_loc, weights, step_cost_flag, n_samples, sample_radius):
    '''
    :param data_loc: location to load and save data from
    :param weights: ground truth weight
    :param step_cost_flag: whether the final element of the reward weight vector is a step cost (and should not be estimated)
    :param n_samples: desired number of test weight candidates
    :param sample_radius: radius around the ground truth weight from which you will uniformly sample from
    :return:

    Summary: Sample test weight candidates. The radius from which you sample will dictate how difficult difficult
    the test environments will be
    '''
    try:
        with open('models/' + data_loc + '/test_wt_candidates.pickle', 'rb') as f:
            wt_candidates = pickle.load(f)
    except:
        d = weights.shape[1] - step_cost_flag
        wt_candidates = np.zeros((n_samples, d))
        count = 0

        # Muller method for sampling from a d-dimensional ball
        while count < n_samples:
            u = np.random.normal(0, 1, d)  # an array of d normally distributed random variables
            norm = np.sum(u**2) ** (0.5)
            r = sample_radius * random.random() ** (1.0 / d)
            x = r * u / norm
            shifted = x + weights[0, 0:d]

            # ensure that the samples aren't out of bounds
            if (shifted > 1.).any() or (shifted < -1.).any():
                continue
            else:
                wt_candidates[count, :] = shifted
                count += 1

        if step_cost_flag:
            wt_candidates = np.hstack((wt_candidates, np.ones((n_samples, 1)) * weights[0, -1]))

        wt_candidates = wt_candidates.reshape((wt_candidates.shape[0], 1, wt_candidates.shape[1]))

        with open('models/' + data_loc + '/test_wt_candidates.pickle', 'wb') as f:
            pickle.dump(wt_candidates, f)

    return wt_candidates

def discretize_wt_candidates(data_loc, weights, weights_lb, weights_ub, step_cost_flag, n_wt_partitions=0, iter_idx=None):
    '''
    Args:
        weights: Ground truth weights (numpy array)
        weights_lb: Lowerbound for weights (numpy array)
        weights_ub: Upperbound for weights (numpy array)
        step_cost_flag: Whether the final element of the reward weight vector is a step cost (and should not be estimated)
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
            # if the last element of the reward weight vector is the step cost,
            if step_cost_flag:
                mesh = np.array(np.meshgrid(
                    *[np.linspace(weights_lb[x], weights_ub[x], n_wt_partitions) for x in np.arange(len(weights[0]) - 1)]))
                wt_uniform_sampling = np.hstack([mesh[x].reshape(-1, 1) for x in np.arange(len(weights[0]) - 1)])
                wt_uniform_sampling = np.hstack((wt_uniform_sampling, weights[0, -1] * np.ones((wt_uniform_sampling.shape[0], 1))))
            else:
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
            wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1, wt_uniform_sampling.shape[1])

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

def obtain_env_policies(data_loc, n_env, wt_candidates, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, save_type):
    '''
    Summary: come up with an optimal policy for each of the candidates
    '''

    # generate codes that govern passenger's initial position and status of the eight tolls in the 4x3 environment
    # todo: can't n_env be recovered by the size of wt_candidates?
    env_codes = list(map(list, itertools.product([0, 1], repeat=int(np.log(n_env) / np.log(2)))))

    save_mark = 750
    if save_type == 'ground_truth':
        filename = 'models/' + data_loc + '/gt_wt_vi_traj_candidates.pickle'
        backup_filename = 'models/' + data_loc + '/gt_wt_vi_traj_candidates_backup.pickle'
    elif save_type == 'test':
        filename = 'models/' + data_loc + '/test_wt_vi_traj_candidates.pickle'
        backup_filename = 'models/' + data_loc + '/test_wt_vi_traj_candidates_backup.pickle'
    else:
        filename = 'models/' + data_loc + '/wt_vi_traj_candidates.pickle'
        backup_filename = 'models/' + data_loc + '/wt_vi_traj_candidates_backup.pickle'

    try:
        with open(filename, 'rb') as f:
            wt_vi_traj_candidates = pickle.load(f)

        if len(wt_vi_traj_candidates) == len(env_codes) and len(wt_vi_traj_candidates[-1]) == len(wt_candidates):
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
                # parameters tailored to the 4x3 Augmented Taxi Domain
                vi_candidate = ValueIteration(mdp_candidate, sample_rate=1, max_iterations=50)
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
                with open(filename, 'wb') as f:
                    pickle.dump(wt_vi_traj_candidates, f)

                # make a backup in case the overwriting in the code above fails
                shutil.copy2(filename, backup_filename)

                print("Saved!")

        with open(filename, 'wb') as f:
            pickle.dump(wt_vi_traj_candidates, f)

        # make a backup in case the overwriting in the code above fails
        shutil.copy2(filename, backup_filename)

    return wt_vi_traj_candidates

def obtain_test_environments(data_loc, weights, wt_candidates, n_env, n_desired_test_env, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, visualize=False):
    '''
    Summary: Select the environments with the most disagreement between the optimal polices of the ground truth weight
    and the test weights as the test environments
    '''
    # obtain optimal policies for each of the test candidate weights and the ground truth candidate weight. It's assumed
    # that the environments appear in the same order for both the test and ground truth weights.
    test_wt_vi_traj_candidates = obtain_env_policies(data_loc, n_env, wt_candidates, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, save_type='test')

    gt_wt_vi_traj_candidates = obtain_env_policies(data_loc, n_env, np.expand_dims(weights, axis=0), agent_a,
                                                           walls_a, traffic_a, fuel_station_a, gamma_a, width_a,
                                                           height_a, 'ground_truth')

    discrepancy = np.zeros(n_env)

    # compare the difference in reward between optimal demonstrations of the candidate weights to the optimal
    # demonstration of the ground truth weight in each environment
    for env_idx in range(n_env):
        gt_weight = gt_wt_vi_traj_candidates[env_idx][0][0]
        gt_mdp = gt_wt_vi_traj_candidates[env_idx][0][1].mdp
        gt_trajectory= gt_wt_vi_traj_candidates[env_idx][0][2]
        test_wt_vi_traj_candidates_tuples = test_wt_vi_traj_candidates[env_idx]

        reward_diff = 0

        for test_wt_vi_traj_candidates_tuple in test_wt_vi_traj_candidates_tuples:
            wt_candidate = test_wt_vi_traj_candidates_tuple[0]
            vi_candidate = test_wt_vi_traj_candidates_tuple[1]
            trajectory_candidate = test_wt_vi_traj_candidates_tuple[2]

            # note that this is different from the reward comparison in BIRL (see bayesian_IRL.py). Euclidean distance
            # between the trajectories could also be used instead
            reward_diff += abs(
                (wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate, discount=True).T) \
                 - gt_weight.dot(gt_mdp.accumulate_reward_features(gt_trajectory, discount=True).T))[0][0])

        discrepancy[env_idx] = reward_diff

    print(discrepancy)
    # sort from greatest discrepancy to least discrepancy, taking the first n_desired_test_env as the test environments
    test_idxs = (-discrepancy).argsort()[:n_desired_test_env]
    print(test_idxs)
    print(discrepancy[test_idxs])

    # visualize each test environment, and the optimal trajectories
    if visualize:
        test_env_count = 1
        for test_idx in test_idxs:
            print(colored('Visualizing test environment {}'.format(test_env_count), 'red'))

            # visualize the ground truth weight trajectory
            print(colored('Ground truth weight: {}'.format(gt_wt_vi_traj_candidates[test_idx][0][0]), 'blue'))
            gt_trajectory = gt_wt_vi_traj_candidates[test_idx][0][2]
            gt_wt_vi_traj_candidates[test_idx][0][1].mdp.visualize_trajectory(gt_trajectory)

            # visualize the candidate weight trajectories
            test_wt_vi_traj_candidates_tuples = test_wt_vi_traj_candidates[test_idx]
            test_wt_count = 1
            for test_wt_vi_traj_candidates_tuple in test_wt_vi_traj_candidates_tuples:
                wt_candidate = test_wt_vi_traj_candidates_tuple[0]
                vi_candidate = test_wt_vi_traj_candidates_tuple[1]
                trajectory_candidate = test_wt_vi_traj_candidates_tuple[2]

                print(colored('#{} candidate weight: {}'.format(test_wt_count, wt_candidate), 'blue'))
                vi_candidate.mdp.visualize_trajectory(trajectory_candidate)

                test_wt_count += 1
            test_env_count += 1

    # return a list of [gt_wt_vi_traj_candidates, test_wt_vi_traj_candidates] that correspond to the selected test
    # environments
    test_wt_vi_traj_tuples = []
    for test_idx in test_idxs:
        test_wt_vi_traj_tuples.append([gt_wt_vi_traj_candidates[test_idx], test_wt_vi_traj_candidates[test_idx]])

    return test_wt_vi_traj_tuples