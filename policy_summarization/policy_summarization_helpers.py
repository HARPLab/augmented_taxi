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

def select_test_demos(cluster_idx, data_loc, n_desired_test_env, env_idxs , traj_opts, BEC_lengths, BEC_constraints, human_counterfactual_trajs, n_clusters=6):
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
    if human_counterfactual_trajs is not None:
        test_human_counterfactual_trajs = [human_counterfactual_trajs[k] for k in test_demo_idxs]
    else:
        test_human_counterfactual_trajs = None

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, test_human_counterfactual_trajs


def optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, chunked_traj_record, summary):
    visual_dissimilarities = np.zeros(len(best_env_idxs))
    complexities = np.zeros(len(best_env_idxs))

    prev_env_idx = None
    for j, best_env_idx in enumerate(best_env_idxs):

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

            if first_state in average_dissimilarity_dict:
                # compare visual dissimilarity of this state to other states in this MDP, trying to minimize dissimilarity
                visual_dissimilarities[j] = average_dissimilarity_dict[first_state]
            else:
                # measure and store how dissimilar this first state is to other states in this MDP
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

def obtain_test_environments(data_loc, min_subset_constraints_record, env_record, traj_record, reward_record, weights, BEC_lengths_record, n_desired_test_env, difficulty, step_cost_flag, counterfactual_folder_idx, summary=None, overlap_in_trajs_avg=None, c=0.001, human_counterfactual_trajs=None, n_clusters=12):
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
    if human_counterfactual_trajs is not None:
        human_counterfactual_trajs_filtered = copy.deepcopy(human_counterfactual_trajs)

    # remove environment and trajectory indices that comprise the summary
    summary_idxs = defaultdict(lambda: [])
    for summary_tuple in summary:
        best_env_idx = summary_tuple[4]
        best_traj_idx = summary_tuple[5]

        summary_idxs[best_env_idx].append(best_traj_idx)

    env_traj_tracer = []
    # todo: 8 and -1 are currently hardcoded. I can also probably just move this to when the overlap between
    #  human posterior and demo BEC are calculated
    human_exit_demo = []
    human_traj_overlap = []
    for env_idx in range(len(traj_record_filtered)):
        print(env_idx)
        best_human_trajs_record_env_across_models = []
        human_rewards_env_across_models = []
        env_traj_tracer_env = []
        human_exit_demo_env = []
        human_traj_overlap_env = []
        for model_idx in range(8):
            with open('models/' + data_loc + '/counterfactual_data_' + str(counterfactual_folder_idx) + '/model' + str(
                    model_idx) + '/cf_data_env' + str(
                env_idx).zfill(5) + '.pickle', 'rb') as f:
                best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)

            # only consider the first best human trajectory (no need to consider partial trajectories)
            first_best_human_trajs_record = [best_human_trajs_record[0] for best_human_trajs_record in
                                             best_human_trajs_record_env]
            best_human_trajs_record_env_across_models.append(first_best_human_trajs_record)
            human_rewards_env_across_models.append(human_rewards_env)

        # reorder such that each subarray is a comparison amongst the models
        human_rewards_env_across_models_per_traj = [list(itertools.chain.from_iterable(i)) for i in
                                                    zip(*human_rewards_env_across_models)]
        best_human_trajs_record_env_across_models_per_traj = [i for i in
                                                              zip(*best_human_trajs_record_env_across_models)]

        for traj_idx in range(len(human_rewards_env_across_models_per_traj)):
            human_rewards_across_models = np.array(human_rewards_env_across_models_per_traj[traj_idx]).flatten()
            best_human_counterfactual_trajs = best_human_trajs_record_env_across_models_per_traj[traj_idx]
            avg_overlap_pct = 0
            overlap_counter = 0
            max_overlap_pct = 0
            min_overlap_pct = 1
            for model_idx, reward in enumerate(human_rewards_across_models):
                # if reward == reward_record[env_idx][traj_idx]:
                #     human_rewards_across_models[model_idx] = 0
                # else:
                #     # method 1: get the average traj overlap amongst suboptimal trajectories
                #     # if the human considered a suboptimal trajectory, consider the overlap
                #     # avg_overlap_pct += BEC_helpers.calculate_counterfactual_overlap_pct(
                #     #     best_human_counterfactual_trajs[model_idx], traj_record_filtered[env_idx][traj_idx])
                #     # overlap_counter += 1
                #
                #     # method 2: get the max traj overlap amongst non-exit trajectories
                #     overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(
                #         best_human_counterfactual_trajs[model_idx], traj_record_filtered[env_idx][traj_idx])
                #     if overlap_pct > max_overlap_pct:
                #         max_overlap_pct = overlap_pct

                # method 3: get the average traj overlap amongst non exit-human trajectories
                # if best_human_counterfactual_trajs[model_idx][0][1] != 'exit':
                #     avg_overlap_pct += BEC_helpers.calculate_counterfactual_overlap_pct(
                #         best_human_counterfactual_trajs[model_idx], traj_record_filtered[env_idx][traj_idx])
                #     overlap_counter += 1

                # method 4: get the min traj overlap amongst non-exit trajectories
                if best_human_counterfactual_trajs[model_idx][0][1] != 'exit':
                    overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(
                        best_human_counterfactual_trajs[model_idx], traj_record_filtered[env_idx][traj_idx])
                    if overlap_pct < min_overlap_pct:
                        min_overlap_pct = overlap_pct

            # method 1: get the average traj overlap amongst non-exit trajectories
            # if overlap_counter > 0:
            #     avg_overlap_pct = avg_overlap_pct / overlap_counter
            # human_traj_overlap_env.append(avg_overlap_pct)
            # method 2: get the max traj overlap amongst non-exit trajectories
            # human_traj_overlap_env.append(max_overlap_pct)
            # method 3: get the average traj overlap amongst non-exit trajectories
            # if overlap_counter > 0:
            #     avg_overlap_pct = avg_overlap_pct / overlap_counter
            # human_traj_overlap_env.append(avg_overlap_pct)
            # method 4: get the min traj overlap amongst non-exit trajectories
            human_traj_overlap_env.append(min_overlap_pct)

            # the code right below is outdated and human_exit_demo_env actually isn't being used
            # if 0, then every trajectory is either correct or is an exit.
            if np.sum(human_rewards_across_models) == 0:
                human_exit_demo_env.append(True)
            else:
                human_exit_demo_env.append(False)

            env_traj_tracer_env.append((env_idx, traj_idx))

        env_traj_tracer.append(env_traj_tracer_env)
        human_exit_demo.append(human_exit_demo_env)
        human_traj_overlap.append(human_traj_overlap_env)

    for env_idx in summary_idxs.keys():
        traj_idxs = summary_idxs[env_idx]
        for traj_idx in sorted(traj_idxs, reverse=True):
            del traj_record_filtered[env_idx][traj_idx]
            del min_subset_constraints_record_filtered[env_idx][traj_idx]
            del BEC_lengths_record_filtered[env_idx][traj_idx]
            if overlap_in_trajs_avg is not None:
                del overlap_in_trajs_avg_filtered[env_idx][traj_idx]
            if human_counterfactual_trajs is not None:
                del human_counterfactual_trajs_filtered[env_idx][traj_idx]
            del env_traj_tracer[env_idx][traj_idx]
            del human_exit_demo[env_idx][traj_idx]
            del human_traj_overlap[env_idx][traj_idx]

    # flatten relevant lists for easy sorting
    envs_record_flattened = []
    for j, env_idx in enumerate(env_record):
        envs_record_flattened.extend([env_idx] * len(traj_record_filtered[j]))

    traj_record_flattened = [item for sublist in traj_record_filtered for item in sublist]
    min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_filtered for item in sublist]
    if human_counterfactual_trajs is not None:
        human_counterfactual_trajs_flattened = [item for sublist in human_counterfactual_trajs_filtered for item in sublist]
    else:
        human_counterfactual_trajs_flattened = None
    human_traj_overlap_flattened = [item for sublist in human_traj_overlap for item in sublist]

    # if overlap_in_trajs_avg is None:
    #     obj_func_flattened = [item for sublist in BEC_lengths_record_filtered for item in sublist]
    # else:
    #     BEC_lengths_record_flattened = [item for sublist in BEC_lengths_record_filtered for item in sublist]
    #     overlap_in_trajs_avg_flattened = [item for sublist in overlap_in_trajs_avg_filtered for item in sublist]
    #     obj_func_flattened = np.array(BEC_lengths_record_flattened) * (np.array(overlap_in_trajs_avg_flattened) + c)

    BEC_lengths_record_flattened = [item for sublist in BEC_lengths_record_filtered for item in sublist]
    obj_func_flattened = np.array(BEC_lengths_record_flattened) * (np.array(human_traj_overlap_flattened) + c)

    # todo: I can probably just do this above while im flattening
    exit_demo = []
    for traj_opt in traj_record_flattened:
        exit_demo.append(traj_opt[0][1] == 'exit')

    human_exit_demo_flattened = [item for sublist in human_exit_demo for item in sublist]
    env_traj_tracer_flattened = [item for sublist in env_traj_tracer for item in sublist]

    envs_record_flattened_exit = []
    traj_record_flattened_exit = []
    obj_func_flattened_exit = []
    min_subset_constraints_record_flattened_exit = []
    human_counterfactual_trajs_flattened_exit = []
    overlap_in_trajs_avg_flattened_exit = []
    env_traj_tracer_flattened_exit = []

    envs_record_flattened_nonexit = []
    traj_record_flattened_nonexit = []
    obj_func_flattened_nonexit = []
    min_subset_constraints_record_flattened_nonexit = []
    human_counterfactual_trajs_flattened_nonexit = []
    overlap_in_trajs_avg_flattened_nonexit = []
    env_traj_tracer_flattened_nonexit = []

    for j, exit in enumerate(exit_demo):
        if exit:
            envs_record_flattened_exit.append(envs_record_flattened[j])
            traj_record_flattened_exit.append(traj_record_flattened[j])
            obj_func_flattened_exit.append(obj_func_flattened[j])
            min_subset_constraints_record_flattened_exit.append(min_subset_constraints_record_flattened[j])
            if human_counterfactual_trajs_flattened is not None:
                human_counterfactual_trajs_flattened_exit.append(human_counterfactual_trajs_flattened[j])
            else:
                human_counterfactual_trajs_flattened_exit = None
            if overlap_in_trajs_avg is not None:
                overlap_in_trajs_avg_flattened_exit.append(overlap_in_trajs_avg_flattened[j])
            env_traj_tracer_flattened_exit.append(env_traj_tracer_flattened[j])
        else:
            envs_record_flattened_nonexit.append(envs_record_flattened[j])
            traj_record_flattened_nonexit.append(traj_record_flattened[j])
            obj_func_flattened_nonexit.append(obj_func_flattened[j])
            min_subset_constraints_record_flattened_nonexit.append(min_subset_constraints_record_flattened[j])
            if human_counterfactual_trajs_flattened is not None:
                human_counterfactual_trajs_flattened_nonexit.append(human_counterfactual_trajs_flattened[j])
            else:
                human_counterfactual_trajs_flattened_nonexit = None
            if overlap_in_trajs_avg is not None:
                overlap_in_trajs_avg_flattened_nonexit.append(overlap_in_trajs_avg_flattened[j])
            env_traj_tracer_flattened_nonexit.append(env_traj_tracer_flattened[j])

    consider_exits = False
    if consider_exits:
        envs_record_flattened_select = envs_record_flattened_exit
        traj_record_flattened_select = traj_record_flattened_exit
        obj_func_flattened_select = obj_func_flattened_exit
        min_subset_constraints_record_flattened_select = min_subset_constraints_record_flattened_exit
        human_counterfactual_trajs_flattened_select = human_counterfactual_trajs_flattened_exit
        overlap_in_trajs_avg_flattened_select = overlap_in_trajs_avg_flattened_exit
    else:
        envs_record_flattened_select = envs_record_flattened_nonexit
        traj_record_flattened_select = traj_record_flattened_nonexit
        obj_func_flattened_select = obj_func_flattened_nonexit
        min_subset_constraints_record_flattened_select = min_subset_constraints_record_flattened_nonexit
        human_counterfactual_trajs_flattened_select = human_counterfactual_trajs_flattened_nonexit
        overlap_in_trajs_avg_flattened_select = overlap_in_trajs_avg_flattened_nonexit

    if difficulty == 'high':
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, test_human_counterfactual_trajs = select_test_demos(
            0, data_loc, n_desired_test_env, envs_record_flattened_select, traj_record_flattened_select, obj_func_flattened_select,
            min_subset_constraints_record_flattened_select, human_counterfactual_trajs_flattened_select, n_clusters=n_clusters)
    if difficulty == 'medium':
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, test_human_counterfactual_trajs = select_test_demos(int(np.ceil(n_clusters / 2)), data_loc, n_desired_test_env,
                                                                                           envs_record_flattened_select,
                                                                                           traj_record_flattened_select,
                                                                                           obj_func_flattened_select, min_subset_constraints_record_flattened_select, human_counterfactual_trajs_flattened_select, n_clusters=n_clusters)

    if difficulty == 'low':
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, test_human_counterfactual_trajs = select_test_demos(n_clusters-1, data_loc, n_desired_test_env,
                                                                                           envs_record_flattened_select,
                                                                                           traj_record_flattened_select,
                                                                                           obj_func_flattened_select, min_subset_constraints_record_flattened_select, human_counterfactual_trajs_flattened_select, n_clusters=n_clusters)

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, test_human_counterfactual_trajs