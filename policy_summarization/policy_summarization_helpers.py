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
from collections import defaultdict

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
    elif mdp_class == 'two_goal' or mdp_class == 'two_goal2':
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
    elif mdp_class == 'skateboard2':
        if hardcode_envs:
            skateboard, paths, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            skateboard, paths, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['skateboard'] = skateboard
        mdp_parameters['paths'] = paths
        mdp_parameters['env_code'] = env_code
    elif mdp_class == 'augmented_taxi2':
        # note that this is specially accommodates the four hand-designed environments
        if hardcode_envs:
            passengers, tolls, hotswap_stations, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            passengers, tolls, hotswap_stations, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['passengers'] = passengers
        mdp_parameters['tolls'] = tolls
        mdp_parameters['hotswap_station'] = hotswap_stations
        mdp_parameters['env_code'] = env_code
    elif mdp_class == 'cookie_crumb':
        if hardcode_envs:
            crumbs, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            crumbs, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['crumbs'] = crumbs
        mdp_parameters['env_code'] = env_code
    elif mdp_class == 'colored_tiles':
        if hardcode_envs:
            A_tiles, B_tiles, env_code = make_mdp.hardcode_mdp_obj(mdp_class, mdp_code)
        else:
            A_tiles, B_tiles, env_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['A_tiles'] = A_tiles
        mdp_parameters['B_tiles'] = B_tiles
        mdp_parameters['env_code'] = env_code
    elif mdp_class == 'augmented_navigation':
        requested_gravel, requested_grass, requested_road, requested_hotswap_stations, requested_skateboard, requested_car, mdp_code = make_mdp.make_mdp_obj(mdp_class, mdp_code, mdp_parameters)
        mdp_parameters['gravel'] = requested_gravel
        mdp_parameters['grass'] = requested_grass
        mdp_parameters['roads'] = requested_road
        mdp_parameters['hotswap_station'] = requested_hotswap_stations
        mdp_parameters['skateboard'] = requested_skateboard
        mdp_parameters['cars'] = requested_car
    else:
        raise Exception("Unknown MDP class.")

    # a per-environment tuple of corresponding reward weight, optimal policy, and optimal trajectory
    wt_vi_traj_env = []
    wt_counter = 0
    for wt_candidate in wt_candidates:
        mdp_parameters['weights'] = wt_candidate
        mdp_candidate = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)

        # parameters tailored to the 4x3 Augmented Taxi Domain
        vi_candidate = ValueIteration(mdp_candidate, sample_rate=1)
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
        elif mdp_class == 'two_goal' or mdp_class == 'two_goal2':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_walls']))))
        elif mdp_class == 'skateboard':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_walls']))))
        elif mdp_class == 'skateboard2':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=int((len(mdp_parameters['available_paths'])/4)+1))))
        elif mdp_class == 'cookie_crumb':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_crumbs']))))
        elif mdp_class == 'augmented_taxi2':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_tolls'])+1)))
        elif mdp_class == 'colored_tiles':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=len(mdp_parameters['available_A_tiles'])+len(mdp_parameters['available_B_tiles']))))
        elif mdp_class == 'augmented_navigation':
            mdp_codes = list(map(list, itertools.product([0, 1], repeat=6)))  # todo: make this a variable
        else:
            raise Exception("Unknown MDP class.")

    policy_dir = 'models/' + data_loc + '/gt_policies/'
    os.makedirs(policy_dir, exist_ok=True)
    n_processed_envs = len(os.listdir(policy_dir))

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

def optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, chunked_traj_record, summary):
    visual_dissimilarities = np.zeros(len(best_env_idxs))
    complexities = np.zeros(len(best_env_idxs))

    prev_env_idx = None
    for j, best_env_idx in enumerate(best_env_idxs):

        # assuming that environments are provided in order of monotonically increasing indexes
        if prev_env_idx != best_env_idx:
            # reset the visual dissimilarity dictionary for a new MDP
            average_dissimilarity_dict = {}

            filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
            with open(filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)
            best_mdp = wt_vi_traj_env[0][1].mdp

        # compare the visual dissimilarity to the most recent summary (only if it's not the first summary
        # and you didn't recently switch which variable you wished to convey
        if len(summary) > 0:
            # get similar demos (todo: consider various scenarios for similarity and dissimilarity later)
            visual_dissimilarities[j] = best_mdp.measure_visual_dissimilarity(
                chunked_traj_record[best_env_idx][best_traj_idxs[j]][0][0], summary[-1][0], summary[-1][1][0][0])
        else:
            first_state = chunked_traj_record[best_env_idx][best_traj_idxs[j]][0][0]

            # compare visual dissimilarity of this state to other states in this MDP, trying to minimize dissimilarity.
            # the rationale behind this is that you want to have a starting demonstration that can be easily followed
            # up by visually similar demonstrations
            if first_state in average_dissimilarity_dict:
                visual_dissimilarities[j] = average_dissimilarity_dict[first_state]
            else:
                average_dissimilarity = 0
                for other_state_idx, other_state in enumerate(best_mdp.states):
                    if first_state != other_state:
                        average_dissimilarity += best_mdp.measure_visual_dissimilarity(first_state, best_mdp, other_state)

                average_dissimilarity = average_dissimilarity / (len(best_mdp.states) - 1)
                average_dissimilarity_dict[first_state] = average_dissimilarity

                visual_dissimilarities[j] = average_dissimilarity

        # get demos of low visual complexity
        complexities[j] = best_mdp.measure_env_complexity()

        prev_env_idx = best_env_idx

    tie_breaker = np.arange(len(best_env_idxs))
    np.random.shuffle(tie_breaker)
    # sorts from small to large values

    # sort first for visual simplicity, then visual similarity
    sorted_zipped = sorted(zip(complexities, visual_dissimilarities, tie_breaker, best_env_idxs, best_traj_idxs))
    complexities_sorted, visual_dissimilarities_sorted, _, best_env_idxs_sorted, best_traj_idxs_sorted = list(
        zip(*sorted_zipped))

    best_env_idx = best_env_idxs_sorted[0]
    best_traj_idx = best_traj_idxs_sorted[0]

    return best_env_idx, best_traj_idx

def select_test_demos(cluster_idx, data_loc, n_desired_test_env, env_idxs, traj_opts, BEC_lengths, BEC_constraints, env_traj_tracer_flattened, n_clusters=5):
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

    selected_env_traj_tracers = [env_traj_tracer_flattened[k] for k in test_demo_idxs]

    selected_env_idxs = [env_idxs[k] for k in test_demo_idxs]

    test_wt_vi_traj_tuples = []
    for k in selected_env_idxs:
        env_filename = mp_helpers.lookup_env_filename(data_loc, k)

        with open(env_filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)

        test_wt_vi_traj_tuples.append(copy.deepcopy(wt_vi_traj_env[0]))

    test_traj_opts = [traj_opts[k] for k in test_demo_idxs]

    for k, test_traj_opt in enumerate(test_traj_opts):
        test_wt_vi_traj_tuples[k][1].mdp.set_init_state(test_traj_opt[0][0])
        test_wt_vi_traj_tuples[k][2] = test_traj_opt

    test_BEC_lengths = [BEC_lengths[k] for k in test_demo_idxs]
    test_BEC_constraints = [BEC_constraints[k] for k in test_demo_idxs]

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers

def obtain_test_environments(data_loc, min_subset_constraints_record, env_record, traj_record, weights, BEC_lengths_record, n_desired_test_env, difficulty, step_cost_flag, summary=None, overlap_in_trajs_avg=None, c=0.001):
    '''
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    '''
    traj_record_filtered = copy.deepcopy(traj_record)
    min_subset_constraints_record_filtered = copy.deepcopy(min_subset_constraints_record)
    BEC_lengths_record_filtered = copy.deepcopy(BEC_lengths_record)
    if overlap_in_trajs_avg is not None:
        overlap_in_trajs_avg_filtered = copy.deepcopy(overlap_in_trajs_avg)


    env_traj_tracer = []
    for j, env_idx in enumerate(env_record):
        env_traj_tracer.append(list(zip([env_idx] * len(traj_record[j]), list(np.arange(len(traj_record[env_idx]))))))


    # remove environment and trajectory indices that comprise the summary
    summary_idxs = defaultdict(lambda: [])
    for summary_tuple in summary:
        best_env_idx = summary_tuple[2][0]
        best_traj_idx = summary_tuple[2][1]

        summary_idxs[best_env_idx].append(best_traj_idx)

    for env_idx in summary_idxs.keys():
        traj_idxs = summary_idxs[env_idx].copy()
        traj_idxs = list(set(traj_idxs)) # remove redundant trajectory idxs
        for traj_idx in sorted(traj_idxs, reverse=True):
            del traj_record_filtered[env_idx][traj_idx]
            del min_subset_constraints_record_filtered[env_idx][traj_idx]
            del BEC_lengths_record_filtered[env_idx][traj_idx]
            if overlap_in_trajs_avg is not None:
                del overlap_in_trajs_avg_filtered[env_idx][traj_idx]
            del env_traj_tracer[env_idx][traj_idx]

    # flatten relevant lists for easy sorting
    envs_record_flattened = []
    for j, env_idx in enumerate(env_record):
        envs_record_flattened.extend([env_idx] * len(traj_record_filtered[j]))

    env_traj_tracer_flattened = [item for sublist in env_traj_tracer for item in sublist]


    traj_record_flattened = [item for sublist in traj_record_filtered for item in sublist]
    min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_filtered for item in sublist]
    if overlap_in_trajs_avg is None:
        obj_func_flattened = [item for sublist in BEC_lengths_record_filtered for item in sublist]
    else:
        BEC_lengths_record_flattened = [item for sublist in BEC_lengths_record_filtered for item in sublist]
        overlap_in_trajs_avg_flattened = [item for sublist in overlap_in_trajs_avg_filtered for item in sublist]
        obj_func_flattened = np.array(BEC_lengths_record_flattened) * (np.array(overlap_in_trajs_avg_flattened) + c)

    if difficulty == 'high':
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = select_test_demos(0, data_loc, n_desired_test_env, envs_record_flattened, traj_record_flattened, obj_func_flattened, min_subset_constraints_record_flattened, env_traj_tracer_flattened)
    if difficulty == 'medium':
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = select_test_demos(2, data_loc, n_desired_test_env,
                                                                                           envs_record_flattened,
                                                                                           traj_record_flattened,
                                                                                           obj_func_flattened, min_subset_constraints_record_flattened, env_traj_tracer_flattened)

    if difficulty == 'low':
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = select_test_demos(4, data_loc, n_desired_test_env,
                                                                                           envs_record_flattened,
                                                                                           traj_record_flattened,
                                                                                           obj_func_flattened, min_subset_constraints_record_flattened, env_traj_tracer_flattened)

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers