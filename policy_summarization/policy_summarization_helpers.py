import numpy as np
import dill as pickle
import itertools
import shutil
import random
from termcolor import colored
import copy
from sklearn.cluster import KMeans
import os
from tqdm import tqdm

# Other imports
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_helpers
from policy_summarization import BEC_helpers
from simple_rl.utils import make_mdp
from policy_summarization import multiprocessing_helpers as mp_helpers

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
                    *[np.linspace(weights_lb[0][x], weights_ub[0][x], n_wt_partitions) for x in np.arange(len(weights[0]) - 1)]))
                wt_uniform_sampling = np.hstack([mesh[x].reshape(-1, 1) for x in np.arange(len(weights[0]) - 1)])
                wt_uniform_sampling = np.hstack((wt_uniform_sampling, weights[0, -1] * np.ones((wt_uniform_sampling.shape[0], 1))))
            else:
                mesh = np.array(np.meshgrid(
                    *[np.linspace(weights_lb[0][x], weights_ub[0][x], n_wt_partitions) for x in np.arange(len(weights[0]))]))
                wt_uniform_sampling = np.hstack([mesh[x].reshape(-1, 1) for x in np.arange(len(weights[0]))])
            wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
            wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1,
                                                              wt_uniform_sampling.shape[1])  # for future dot products
        # uniformly discretize only over the desired reward weight dimension
        else:
            discretized_weights = np.linspace(weights_lb[0][iter_idx], weights_ub[0][iter_idx], n_wt_partitions)
            wt_uniform_sampling = np.tile(weights, (len(discretized_weights), 1))
            wt_uniform_sampling[:, iter_idx] = discretized_weights
            wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
            wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1, wt_uniform_sampling.shape[1])

        with open('models/' + data_loc + '/wt_candidates.pickle', 'wb') as f:
            pickle.dump(wt_uniform_sampling, f)

    return wt_uniform_sampling

def solve_policy(args):
    env_idx, mdp_code, mdp_class, hardcode_envs, mdp_parameters, wt_candidates, data_loc = args

    if mdp_class == 'augmented_taxi':
        # note that this is specially accommodates the four hand-designed environments
        if hardcode_envs:
            passengers, tolls, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            passengers, tolls, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['passengers'] = passengers
        mdp_parameters['tolls'] = tolls
        mdp_parameters['env_code'] = env_code
    elif mdp_class == 'two_goal':
        if hardcode_envs:
            walls, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            walls, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['walls'] = walls
        mdp_parameters['env_code'] = env_code
    elif mdp_class == 'skateboard':
        if hardcode_envs:
            walls, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            walls, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['walls'] = walls
        mdp_parameters['env_code'] = env_code
    elif mdp_class == 'cookie_crumb':
        if hardcode_envs:
            crumbs, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            crumbs, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['crumbs'] = crumbs
        mdp_parameters['env_code'] = env_code
    else:
        raise Exception("Unknown MDP class.")

    # a per-environment tuple of corresponding reward weight, optimal policy, and optimal trajectory
    wt_vi_traj_env = []
    wt_counter = 0
    for wt_candidate in wt_candidates:
        mdp_parameters['weights'] = wt_candidate
        if mdp_class == 'augmented_taxi':
            mdp_candidate = make_mdp.make_custom_mdp('augmented_taxi', mdp_parameters)
        elif mdp_class == 'two_goal':
            mdp_candidate = make_mdp.make_custom_mdp('two_goal', mdp_parameters)
        elif mdp_class == 'skateboard':
            mdp_candidate = make_mdp.make_custom_mdp('skateboard', mdp_parameters)
        elif mdp_class == 'cookie_crumb':
            mdp_candidate = make_mdp.make_custom_mdp('cookie_crumb', mdp_parameters)
        else:
            raise Exception("Unknown MDP class.")

        # parameters tailored to the 4x3 Augmented Taxi Domain
        vi_candidate = ValueIteration(mdp_candidate, sample_rate=1, max_iterations=25)
        iterations, value_of_init_state = vi_candidate.run_vi()
        trajectory = mdp_helpers.rollout_policy(mdp_candidate, vi_candidate)
        wt_vi_traj_env.append([wt_candidate, vi_candidate, trajectory, mdp_parameters.copy()])

        wt_counter += 1

    with open(mp_helpers.lookup_env_filename(data_loc, env_idx), 'wb') as f:
        pickle.dump(wt_vi_traj_env, f)

def obtain_env_policies(mdp_class, data_loc, wt_candidates, mdp_parameters, pool, hardcode_envs=False):
    '''
    Summary: come up with an optimal policy for each of the candidates
    '''
    if hardcode_envs:
        # each of the domains has four possible hard-coded environments
        mdp_codes = list(map(list, itertools.product([0, 1], repeat=2)))
    else:
        # generate codes that govern the binary status of available tolls, walls, or crumbs
        if mdp_class == 'augmented_taxi':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_tolls']))))
        elif mdp_class == 'two_goal':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_walls']))))
        elif mdp_class == 'skateboard':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_walls']))))
        elif mdp_class == 'cookie_crumb':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_crumbs']))))
        else:
            raise Exception("Unknown MDP class.")

    n_processed_envs = len(os.listdir('models/' + data_loc + '/gt_policies/'))

    print("Solving for the optimal policy in each environment:")
    pool.restart()
    args = [(i, mdp_codes[i], mdp_class, hardcode_envs, mdp_parameters, wt_candidates, data_loc) for i in range(n_processed_envs, len(mdp_codes))]
    list(tqdm(pool.imap(solve_policy, args), total=len(args)))
    pool.close()
    pool.join()
    pool.terminate()

def _in_summary(mdp, summary, initial_state):
    '''
    Summary: Check if this MDP (and trajectory, if summary type is policy BEC) is already in the BEC summary. If so,
    do not consider it as a potential test environment.
    '''
    if summary is None:
        return False

    for summary_idx in range(len(summary)):
        if (mdp.env_code == summary[summary_idx][0].env_code) and (summary[summary_idx][1][0][0] == initial_state):
            return True
    return False

def select_test_demos(cluster_idx, n_desired_test_env, wt_vi_traj_candidates, env_idxs , traj_opts, BEC_lengths, BEC_constraints, n_clusters=6):
    kmeans = KMeans(n_clusters=n_clusters).fit(np.array(BEC_lengths).reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    ordering = np.arange(0, n_clusters)
    sorted_zipped = sorted(zip(cluster_centers, ordering))
    cluster_centers_sorted, ordering_sorted = list(zip(*sorted_zipped))

    # checking out partitions one at a time
    partition_idx = ordering_sorted[cluster_idx]

    covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
    print(len(covering_demo_idxs))

    test_demo_idxs = random.sample(covering_demo_idxs, min(n_desired_test_env, len(covering_demo_idxs)))

    selected_env_idxs = [env_idxs[k] for k in test_demo_idxs]

    test_wt_vi_traj_tuples = [copy.deepcopy(wt_vi_traj_candidates[k][0]) for k in
                              selected_env_idxs]

    test_traj_opts = [traj_opts[k] for k in test_demo_idxs]

    for k, test_traj_opt in enumerate(test_traj_opts):
        test_wt_vi_traj_tuples[k][1].mdp.set_init_state(test_traj_opt[0][0])
        test_wt_vi_traj_tuples[k][2] = test_traj_opt

    test_BEC_lengths = [BEC_lengths[k] for k in test_demo_idxs]
    test_BEC_constraints = [BEC_constraints[k] for k in test_demo_idxs]

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints


def obtain_test_environments(wt_vi_traj_candidates, min_subset_constraints_record, env_record, traj_record, weights, n_desired_test_env, difficulty, step_cost_flag, summary=None, BEC_summary_type=None):
    '''
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    '''
    env_idxs = []
    env_complexities = []
    BEC_lengths = []
    BEC_constraints = []
    traj_opts = []

    # go through the BEC constraints of each possible optimal demo in each environment and store the corresponding
    # BEC lengths
    for j, constraints in enumerate(min_subset_constraints_record):
        # todo: this may be redundant since the BEC lengths are already calculated for generating the summary
        if j % 1000 == 0:
            print('{}/{} of constraints checked to see if they\'re in the summary'.format(j, len(min_subset_constraints_record)))
        if not _in_summary(wt_vi_traj_candidates[env_record[j]][0][1].mdp, summary, traj_record[j][0][0]):
            BEC_length = BEC_helpers.calculate_BEC_length(constraints, weights, step_cost_flag)[0]
            BEC_lengths.append(BEC_length)
            env_complexities.append(wt_vi_traj_candidates[env_record[j]][0][1].mdp.measure_env_complexity())
            env_idxs.append(env_record[j])
            BEC_constraints.append(constraints)
            traj_opts.append(traj_record[j])

    if BEC_summary_type == 'demo':
        # demo test environment generation is outdated
        if difficulty == 'high':
            test_wt_vi_traj_tuples = [wt_vi_traj_candidates[k][0] for k in
                                            env_record_sorted[unique_idxs[:n_desired_test_env]]]
            test_BEC_lengths = [BEC_lengths_sorted[k] for k in unique_idxs[:n_desired_test_env]]
            test_BEC_constraints = [BEC_constraints_sorted[k] for k in unique_idxs[:n_desired_test_env]]
        else:
            test_wt_vi_traj_tuples = [wt_vi_traj_candidates[k][0] for k in
                                            env_record_sorted[unique_idxs[-n_desired_test_env:]]]
            test_BEC_lengths = [BEC_lengths_sorted[k] for k in unique_idxs[-n_desired_test_env:]]
            test_BEC_constraints = [BEC_constraints_sorted[k] for k in unique_idxs[-n_desired_test_env:]]
    else:
        if difficulty == 'high':
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = select_test_demos(0, n_desired_test_env, wt_vi_traj_candidates, env_idxs, traj_opts, BEC_lengths, BEC_constraints)
        if difficulty == 'medium':
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = select_test_demos(2, n_desired_test_env,
                                                                                               wt_vi_traj_candidates,
                                                                                               env_idxs, traj_opts,
                                                                                               BEC_lengths, BEC_constraints)

        if difficulty == 'low':
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = select_test_demos(4, n_desired_test_env,
                                                                                               wt_vi_traj_candidates,
                                                                                               env_idxs, traj_opts,
                                                                                               BEC_lengths, BEC_constraints)

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints