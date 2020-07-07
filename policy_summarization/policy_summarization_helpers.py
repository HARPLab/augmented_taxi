import numpy as np
import dill as pickle
import itertools
import shutil
import random
from termcolor import colored
import copy

# Other imports
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from simple_rl.agents import FixedPolicyAgent
from simple_rl.utils import mdp_helpers
from policy_summarization import BEC

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

def generate_mdp_obj(mdp_code):
    '''
    :param mdp_code: Vector representation of an augmented taxi environment (list of binary values)
    :return: Corresponding passenger and toll objects and code that specifically only concerns the environment (and not
    the initial state) of the MDP
    '''

    # first entry currently dictates where the passenger begins
    if mdp_code[0] == 0:
        requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
    else:
        requested_passenger = [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]

    # the last eight entries currently dictate the presence of tolls
    available_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
               {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 4, "y": 2, "fee": 1},
               {"x": 2, "y": 1, "fee": 1}, {"x": 3, "y": 1, "fee": 1}]

    requested_tolls = []

    offset = 1
    for x in range(offset, len(mdp_code)):
        entry = mdp_code[x]
        if entry:
            requested_tolls.append(available_tolls[x - offset])

    # note that what's considered mdp_code (potentially includes both initial state and environment info) and env_code
    # (only includes environment info) will always need to be manually defined
    return requested_passenger, requested_tolls, mdp_code[1:]

# hard-coded in order to evaluate hand-designed environments
def hand_generate_mdp_obj(mdp_code):

    # a) for resolving weight of toll
    if mdp_code == [0, 0]:
        requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
        requested_tolls = [{"x": 3, "y": 1, "fee": 1}, {"x": 3, "y": 2, "fee": 1}]  # upperbound
    elif mdp_code == [0, 1]:
        requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
        requested_tolls = [{"x": 3, "y": 1, "fee": 1}]                              # lowerbound
    # b) for resolving weight of dropping off passenger
    elif mdp_code == [1, 0]:
        requested_passenger = [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
        requested_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
                   {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 2, "y": 1, "fee": 1},
                   {"x": 3, "y": 1, "fee": 1}]                              # lowerbound
    else:
        requested_passenger = [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
        requested_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
                   {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 4, "y": 2, "fee": 1},
                   {"x": 2, "y": 1, "fee": 1}, {"x": 3, "y": 1, "fee": 1}]  # upperbound

    return requested_passenger, requested_tolls, mdp_code

def obtain_env_policies(data_loc, n_env, wt_candidates, aug_taxi, save_type):
    '''
    Summary: come up with an optimal policy for each of the candidates
    '''

    # generate codes that govern passenger's initial position and status of the eight tolls in the 4x3 environment
    mdp_codes = list(map(list, itertools.product([0, 1], repeat=int(np.log(n_env) / np.log(2)))))

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

        if len(wt_vi_traj_candidates) == len(mdp_codes) and len(wt_vi_traj_candidates[-1]) == len(wt_candidates):
            # all environments and weights have been processed
            n_processed_envs = len(mdp_codes)
        else:
            # a portion of the environments and weights have been processed
            n_processed_envs = len(wt_vi_traj_candidates)
    except:
        wt_vi_traj_candidates = []
        n_processed_envs = 0

    # enumeration of all possible optimal policies from possible environments x weight candidates
    # if there are environments and weights yet to be processed
    if n_processed_envs < len(mdp_codes):
        for env_idx in range(n_processed_envs, len(mdp_codes)):
            mdp_code = mdp_codes[env_idx]
            # note that this is specially accommodates the four hand-designed environments
            if len(mdp_codes) == 4:
                passengers, tolls, env_code = hand_generate_mdp_obj(mdp_code)
            else:
                passengers, tolls, env_code = generate_mdp_obj(mdp_code)
            wt_counter = 0
            # a per-environment tuple of corresponding reward weight, optimal policy, and optimal trajectory
            wt_vi_traj_env = []
            for wt_candidate in wt_candidates:
                mdp_candidate = AugmentedTaxiOOMDP(width=aug_taxi['width'], height=aug_taxi['height'], agent=aug_taxi['agent'],
                                   walls=aug_taxi['walls'], passengers=passengers, tolls=tolls,
                                   traffic=aug_taxi['traffic'], fuel_stations=aug_taxi['fuel_station'],
                                                   gamma=aug_taxi['gamma'], weights=wt_candidate, env_code=env_code)

                # parameters tailored to the 4x3 Augmented Taxi Domain
                vi_candidate = ValueIteration(mdp_candidate, sample_rate=1, max_iterations=50)
                iterations, value_of_init_state = vi_candidate.run_vi()
                trajectory = mdp_helpers.rollout_policy(mdp_candidate, vi_candidate)
                wt_vi_traj_env.append([wt_candidate, vi_candidate, trajectory])

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
        if not _in_summary(wt_vi_traj_candidates[env_record[j]][0][1].mdp, summary, traj_record[j][0][0]):
            BEC_length = BEC.calculate_BEC_length(constraints, weights, step_cost_flag)
            BEC_lengths.append(BEC_length)
            env_complexities.append(wt_vi_traj_candidates[env_record[j]][0][1].mdp.measure_env_complexity())
            env_idxs.append(env_record[j])
            BEC_constraints.append(constraints)
            traj_opts.append(traj_record[j])

    # sorted from smallest to largest BEC lengths (i.e. most to least challenging)
    tie_breaker = [i for i in range(len(BEC_lengths))]
    sorted_zipped = sorted(zip(BEC_lengths, env_complexities, tie_breaker, env_idxs, BEC_constraints, traj_opts))
    BEC_lengths_sorted, env_complexities_sorted, _, env_record_sorted, BEC_constraints_sorted, traj_opts_sorted = list(zip(*sorted_zipped))
    env_record_sorted = np.array(env_record_sorted)

    # again sorted in order from smallest to largest, only selecting unique BEC lengths (as demos with the same
    # BEC lengths are often quite similar)
    BEC_lengths_unique, unique_idxs = np.unique(BEC_lengths_sorted, return_index=True)

    if BEC_summary_type == 'demo':
        if difficulty == 'hard':
            test_wt_vi_traj_tuples = [wt_vi_traj_candidates[k] for k in
                                            env_record_sorted[unique_idxs[:n_desired_test_env]]]
            test_BEC_lengths = [BEC_lengths_sorted[k] for k in unique_idxs[:n_desired_test_env]]
            test_BEC_constraints = [BEC_constraints_sorted[k] for k in unique_idxs[:n_desired_test_env]]
        else:
            test_wt_vi_traj_tuples = [wt_vi_traj_candidates[k] for k in
                                            env_record_sorted[unique_idxs[-n_desired_test_env:]]]
            test_BEC_lengths = [BEC_lengths_sorted[k] for k in unique_idxs[-n_desired_test_env:]]
            test_BEC_constraints = [BEC_constraints_sorted[k] for k in unique_idxs[-n_desired_test_env:]]
    else:
        # must update the wt_vi_traj_candidate with the right initial state and trajectory
        if difficulty == 'hard':
            test_wt_vi_traj_tuples = [copy.deepcopy(wt_vi_traj_candidates[k]) for k in
                                      env_record_sorted[unique_idxs[:n_desired_test_env]]]

            test_traj_opts = [traj_opts_sorted[k] for k in unique_idxs[:n_desired_test_env]]

            for k, test_traj_opt in enumerate(test_traj_opts):
                test_wt_vi_traj_tuples[k][0][1].mdp.set_init_state(test_traj_opt[0][0])
                test_wt_vi_traj_tuples[k][0][2] = test_traj_opt

            test_BEC_lengths = [BEC_lengths_sorted[k] for k in unique_idxs[:n_desired_test_env]]
            test_BEC_constraints = [BEC_constraints_sorted[k] for k in unique_idxs[:n_desired_test_env]]
        else:
            test_wt_vi_traj_tuples = [copy.deepcopy(wt_vi_traj_candidates[k]) for k in
                                      env_record_sorted[unique_idxs[-n_desired_test_env:]]]

            test_traj_opts = [traj_opts_sorted[k] for k in unique_idxs[-n_desired_test_env:]]

            for k, test_traj_opt in enumerate(test_traj_opts):
                test_wt_vi_traj_tuples[k][0][1].mdp.set_init_state(test_traj_opt[0][0])
                test_wt_vi_traj_tuples[k][0][2] = test_traj_opt

            test_BEC_lengths = [BEC_lengths_sorted[k] for k in unique_idxs[-n_desired_test_env:]]
            test_BEC_constraints = [BEC_constraints_sorted[k] for k in unique_idxs[-n_desired_test_env:]]

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints