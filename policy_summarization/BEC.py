import matplotlib.pyplot as plt
from simple_rl.utils import mdp_helpers
from simple_rl.agents import FixedPolicyAgent
import policy_summarization.BEC_helpers as BEC_helpers
import numpy as np
import itertools
import random
import copy
from simple_rl.planning import ValueIteration
from sklearn.cluster import KMeans
import dill as pickle
from termcolor import colored
import time
from tqdm import tqdm
import os
import policy_summarization.multiprocessing_helpers as mp_helpers

def extract_constraints_policy(args):
    env_idx, data_loc, step_cost_flag = args
    with open(mp_helpers.lookup_env_filename(data_loc, env_idx), 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    mdp = wt_vi_traj_env[0][1].mdp
    agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
    weights = mdp.weights

    min_subset_constraints_record = []    # minimum BEC constraints conveyed by a trajectory
    env_record = []
    policy_constraints = []               # BEC constraints that define a policy (i.e. constraints arising from one action
                                          # deviations from every possible starting state and the corresponding optimal trajectories)
    traj_record = []


    for state in mdp.states:
        constraints = []
        traj_opt = mdp_helpers.rollout_policy(mdp, agent, cur_state=state)

        for sas_idx in range(len(traj_opt)):
            # reward features of optimal action
            mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

            sas = traj_opt[sas_idx]
            cur_state = sas[0]

            # currently assumes that all actions are executable from all states. only considering
            # action depth of 1 currently
            for action in mdp.actions:
                if action != sas[1]:
                    traj_hyp = mdp_helpers.rollout_policy(mdp, agent, cur_state=cur_state, action_seq=[action])
                    mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)

                    constraints.append(mu_sa - mu_sb)

            # if considering only suboptimal actions of the first sas, put the corresponding constraints
            # toward the BEC of the policy (per definition)
            if sas_idx == 0:
                policy_constraints.append(
                    BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag))

        # also store the BEC constraints for optimal trajectory in each state, along with the associated
        # demo and environment number
        min_subset_constraints_record.append(
            BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag))
        traj_record.append(traj_opt)
        env_record.append(env_idx)

    return min_subset_constraints_record, traj_record, env_record, policy_constraints

def extract_constraints_demonstration(args):
    env_idx, vi, traj_opt, step_cost_flag = args

    min_subset_constraints_record = []    # minimum BEC constraints conveyed by a trajectory
    env_record = []
    policy_constraints = []               # BEC constraints that define a policy (i.e. constraints arising from one action
                                          # deviations from every possible starting state and the corresponding optimal trajectories)
    traj_record = []

    mdp = vi.mdp
    agent = FixedPolicyAgent(vi.policy)
    weights = mdp.weights

    constraints = []
    # BEC constraints are obtained by ensuring that the optimal actions accumulate at least as much reward as
    # all other possible actions along a trajectory (only considering an action depth of 1 currently)
    action_seq_list = list(itertools.product(mdp.actions, repeat=1))

    for sas_idx in range(len(traj_opt)):
        # reward features of optimal action
        mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

        sas = traj_opt[sas_idx]
        cur_state = sas[0]

        # currently assumes that all actions are executable from all states
        for action_seq in action_seq_list:
            traj_hyp = mdp_helpers.rollout_policy(mdp, agent, cur_state, action_seq)
            mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)

            constraints.append(mu_sa - mu_sb)

    # store the BEC constraints for each environment, along with the associated demo and environment number
    min_subset_constraints = BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag)
    min_subset_constraints_record.append(min_subset_constraints)
    traj_record.append(traj_opt)
    env_record.append(env_idx)
    # slightly abusing the term 'policy' here since I'm only considering a subset of possible trajectories (i.e.
    # demos) that the policy can generate in these environments
    policy_constraints.append(min_subset_constraints)

    return min_subset_constraints_record, traj_record, env_record, policy_constraints


def extract_constraints(data_loc, step_cost_flag, pool, vi_traj_pairs=None, print_flag=False):
    '''
    :param wt_vi_traj_candidates: Nested list of [weight, value iteration object, trajectory]
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :return: min_subset_constraints: List of constraints

    Summary: Obtain the minimum BEC constraints for each environment
    '''
    min_subset_constraints_record = []    # minimum BEC constraints conveyed by a trajectory
    env_record = []
    policy_constraints = []               # BEC constraints that define a policy (i.e. constraints arising from one action
                                          # deviations from every possible starting state and the corresponding optimal trajectories)
    traj_record = []

    n_envs = len(os.listdir('models/' + data_loc + '/gt_policies/'))

    print("Extracting the BEC constraints in each environment:")
    pool.restart()
    if vi_traj_pairs is None:
        # a) policy-driven BEC: generate constraints by considering the expected feature counts after taking one
        # suboptimal action in every possible state in the state space, then acting optimally afterward. see eq 13, 14
        # of Brown et al. 'Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications' 2019
        args = [(i, data_loc, step_cost_flag) for i in range(n_envs)]
        results = list(tqdm(pool.imap(extract_constraints_policy, args), total=len(args)))
        pool.close()
        pool.join()
        pool.terminate()

        for result in results:
            min_subset_constraints_record.extend(result[0])
            traj_record.extend(result[1])
            env_record.extend(result[2])
            policy_constraints.extend(result[3])
    else:
        # b) demonstration-driven BEC: generate constraints by considering the expected feature counts after taking one
        # suboptimal action in every state along a trajectory (demonstration), then acting optimally afterward.
        # see eq 16 of Brown et al. 'Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications' 2019
        # need to specify the environment, environment idx, and corresponding optimal trajectories (second, first, and
        # third elements of vi_traj_pair, respectively) that you want to extract constraints from
        args = [(vi_traj_pair[0], vi_traj_pair[1], vi_traj_pair[2], step_cost_flag) for vi_traj_pair in enumerate(vi_traj_pairs)]
        results = list(tqdm(pool.imap(extract_constraints_demonstration, args), total=len(args)))
        pool.close()
        pool.join()
        pool.terminate()

        for result in results:
            min_subset_constraints_record.extend(result[0])
            traj_record.extend(result[1])
            env_record.extend(result[2])
            policy_constraints.extend(result[3])

    return policy_constraints, min_subset_constraints_record, env_record, traj_record

def extract_BEC_constraints(policy_constraints, min_subset_constraints_record, weights, step_cost_flag):
    '''
    Summary: Obtain the minimum BEC constraints across all environments
    '''
    constraints_record = [item for sublist in policy_constraints for item in sublist]

    # first obtain the absolute min BEC constraint
    min_BEC_constraints = BEC_helpers.clean_up_constraints(constraints_record, weights, step_cost_flag)

    # then determine the BEC lengths of all other potential demos that could be shown
    BEC_lengths_record = []

    for j, min_subset_constraints in enumerate(min_subset_constraints_record):
        BEC_lengths_record.append(BEC_helpers.calculate_BEC_length(min_subset_constraints, weights, step_cost_flag)[0])

    # ordered from most constraining to least constraining
    return min_BEC_constraints, BEC_lengths_record

def obtain_potential_summary_demos(BEC_lengths_record, n_demos, n_clusters=6, type='scaffolding', sample_count=25):
    covering_demos_idxs = []

    kmeans = KMeans(n_clusters=n_clusters).fit(np.array(BEC_lengths_record).reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    ordering = np.arange(0, n_clusters)
    sorted_zipped = sorted(zip(cluster_centers, ordering))
    cluster_centers_sorted, ordering_sorted = list(zip(*sorted_zipped))

    if type[0] == 'low':
        cluster_idx = 4
        partition_idx = ordering_sorted[cluster_idx]

        for j in range(n_demos):
            covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
            print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)
    elif type[0] == 'medium':
        cluster_idx = 2
        partition_idx = ordering_sorted[cluster_idx]

        for j in range(n_demos):
            covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
            print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)
    elif type[0] == 'high':
        cluster_idx = 0
        partition_idx = ordering_sorted[cluster_idx]

        for j in range(n_demos):
            covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
            print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)
    else:
        # employ scaffolding

        # 0 is the cluster with the smallest BEC lengths
        if n_demos == 3:
            cluster_idxs = [0, 2, 4]
        else:
            cluster_idxs = [2, 4]

        for j in range(n_demos):
            # checking out partitions one at a time
            partition_idx = ordering_sorted[cluster_idxs[j]]

            covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
            print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)

        # filled this from hardest to easiest demos, so flip
        covering_demos_idxs.reverse()

    return covering_demos_idxs

def compute_counterfactuals(args):
    data_loc, model_idx, env_idx, w_human_normalized, env_filename, trajs_opt, min_BEC_constraints_running, feature_flag, step_cost_flag = args

    with open(env_filename, 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    agent = wt_vi_traj_env[0][1]
    mdp = agent.mdp
    weights = mdp.weights

    human = copy.deepcopy(agent)
    mdp = human.mdp
    mdp.weights = w_human_normalized
    vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=25)
    vi_human.run_vi()

    best_human_trajs_record_env = []
    constraints_env = []
    info_gain_env = []

    for traj_opt in trajs_opt:
        constraints = []
        info_gain = 0

        # # a) accumulate the reward features and generate a single constraint
        # mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
        # traj_hyp = mdp_helpers.rollout_policy(vi_human.mdp, vi_human)
        # mu_sb = vi_human.mdp.accumulate_reward_features(traj_hyp, discount=True)
        # constraints.append(mu_sa - mu_sb)

        # b) contrast differing expected feature counts for each state-action pair along the agent's optimal trajectory
        best_human_trajs_record = []
        for sas_idx in range(len(traj_opt)):
            # reward features of optimal action
            mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

            sas = traj_opt[sas_idx]
            cur_state = sas[0]

            # obtain all optimal trajectory rollouts according to the human's model, if it has reasonable policy that converged
            if vi_human.stabilized:
                human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
            # simply use a single roll out (which will likely just be noise anyway)
            else:
                human_opt_traj = mdp_helpers.rollout_policy(vi_human.mdp, vi_human, cur_state)
                human_opt_trajs = [human_opt_traj]

            cur_best_reward = np.float('-inf')
            best_reward_features = []
            best_human_traj = []
            # select the human's possible trajectory that has the highest true reward (i.e. give the human's policy the benefit of the doubt)
            for traj in human_opt_trajs:
                mu_sb = mdp.accumulate_reward_features(traj,
                                                       discount=True)  # the human and agent should be working with identical mdps
                reward_hyp = weights.dot(mu_sb.T)
                if reward_hyp > cur_best_reward:
                    cur_best_reward = reward_hyp
                    best_reward_features = mu_sb
                    best_human_traj = traj

            constraints.append(mu_sa - best_reward_features)
            best_human_trajs_record.append(best_human_traj)

        # the hypothetical constraints that will be in the human's mind after viewing this demonstrations
        hypothetical_constraints = []
        hypothetical_constraints.extend(min_BEC_constraints_running)
        # try conveying one feature at a time
        try:
            constraints = BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag)
            if feature_flag == 0:
                # keep only horizontal constraints (i.e. info on only the tolls)
                filtered_constraints = []
                for constraint in constraints:
                    if constraint[0, 0] == 0:
                        filtered_constraints.append(constraint)
            elif feature_flag == 1:
                # keep only vertical constraints (i.e. info on only the dropoff)
                filtered_constraints = []
                for constraint in constraints:
                    if constraint[0, 1] == 0:
                        filtered_constraints.append(constraint)
            else:
                filtered_constraints = constraints

            hypothetical_constraints.extend(filtered_constraints)
            info_gain = \
            BEC_helpers.calculate_BEC_length(min_BEC_constraints_running, weights, step_cost_flag)[0] - \
            BEC_helpers.calculate_BEC_length(hypothetical_constraints, weights, step_cost_flag)[0]

        except:
            # print("No valid constraints")
            pass
        # mdp.visualize_trajectory(traj_opt)
        # vi_human.mdp.visualize_trajectory(traj_hyp)

        best_human_trajs_record_env.append(best_human_trajs_record)
        constraints_env.append(constraints)
        info_gain_env.append(info_gain)

    with open('models/' + data_loc + '/counterfactual_data/model' + str(model_idx) + 'cf_data_env' + str(env_idx).zfill(5) + '.pickle', 'wb') as f:
        pickle.dump((best_human_trajs_record_env, constraints_env), f)

    return info_gain_env


def obtain_summary_counterfactual(data_loc, summary_variant, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool,
                       n_train_demos=3, downsample_threshold=float("inf")):

    summary = []
    feature_flag = 0

    # assuming that traj_record / env_record are sorted properly by env order, chunk via environment for faster multiprocessing
    current_env = 0
    chunked_traj_record = []
    chunked_env_record = [0]
    chunked_trajs = []
    for idx, env in enumerate(env_record):
        if env != current_env:
            # record the old chunk
            chunked_traj_record.append(chunked_trajs)

            # start a new chunk
            current_env = env
            chunked_env_record.append(env)
            chunked_trajs = [traj_record[idx]]
        else:
            # continue the chunk
            chunked_trajs.append(traj_record[idx])
    chunked_traj_record.append(chunked_trajs)

    # assume that the human starts off knowing which quadrant the agent's weight belongs to
    min_BEC_constraints_running = [np.array([[0, -1, 0]]), np.array([[1, 0, 0]])]

    while len(summary) < n_train_demos:
        visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, fig_name=str(len(summary)) + '.png', just_save=True)

        # use extreme vertices (vertices comprising the convex hull that have a extreme value in at least a single axis) as the human model
        extreme_human_models = BEC_helpers.obtain_extreme_vertices(min_BEC_constraints_running, weights, step_cost_flag)

        info_gains_record = []

        print("Length of summary: {}".format(len(summary)))
        for model_idx, human_model in enumerate(extreme_human_models):
            print(colored('Model #: {}'.format(model_idx), 'red'))
            print(colored('Model val: {}'.format(human_model), 'red'))

            w_human_normalized = human_model
            info_gains_model = []

            # based on the human's current model, obtain the information gain generated when comparing to the agent's
            # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
            # are saved for reference later)
            print("Obtaining counterfactual information gains:")

            cf_data_dir = 'models/' + data_loc + '/counterfactual_data/model' + str(model_idx)
            os.makedirs(cf_data_dir, exist_ok=True)

            pool.restart()
            args = [(data_loc, model_idx, i, w_human_normalized, mp_helpers.lookup_env_filename(data_loc, chunked_env_record[i]), chunked_traj_record[i], min_BEC_constraints_running, feature_flag, step_cost_flag) for i in range(len(chunked_traj_record))]
            info_gain_envs = list(tqdm(pool.imap(compute_counterfactuals, args), total=len(args)))
            pool.close()
            pool.join()
            pool.terminate()

            for info_gain_env in info_gain_envs:
                info_gains_model.append(info_gain_env)

            info_gains_record.append(info_gains_model)

        # no need to continue search for demonstrations if none of them will improve the human's understanding
        info_gains = np.array(info_gains_record)
        if np.sum(info_gains) == 0:
            # when feature_flag == n_features - 1, you have already considered constraints that aren't solely informative
            # regard one feature (i.e. constraints that are axis-parallel). at this point, is no more information to gain
            if feature_flag == 2:
                break
            else:
                # move onto conveying the next feature
                feature_flag += 1
                print(colored('Incrementing feature flag, which is now at {}'.format(feature_flag), 'red'))
                continue

        # todo: let's just try returning the smallest information gain for now (so that we can have incremental demonstrations)
        #  there often isn't enough demonstrations to generate 6 clusters, as I initially tried to do below
        info_gains[info_gains <= 0] = np.float('inf')
        select_model, best_env_idx, best_traj_idx = np.unravel_index(np.argmin(info_gains), info_gains.shape)
        print(colored('Max infogain: {}'.format(np.max(info_gains)), 'blue'))
        print(colored('Max Min infogain: {}'.format(np.min(info_gains)), 'blue')) # smallest infogain above zero

        best_traj = chunked_traj_record[best_env_idx][best_traj_idx]
        with open('models/' + data_loc + '/counterfactual_data/model' + str(select_model) + 'cf_data_env' + str(
                best_env_idx).zfill(5) + '.pickle', 'rb') as f:
            best_human_trajs_record_env, constraints_env = pickle.load(f)
        best_human_trajs = best_human_trajs_record_env[best_traj_idx]

        filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
        with open(filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)
        best_mdp = wt_vi_traj_env[0][1].mdp
        best_mdp.set_init_state(best_traj[0][0]) # for completeness
        min_BEC_constraints_running.extend(constraints_env[best_traj_idx])
        min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)[0]

        summary.append([best_mdp, best_traj, constraints_env[best_traj_idx], best_human_trajs])

        # this method doesn't always finish, so save the summary along the way
        with open('models/augmented_taxi/BEC_summary.pickle', 'wb') as f:
            pickle.dump(summary, f)

    return summary


def obtain_summary(summary_variant, wt_vi_traj_candidates, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=3, downsample_threshold=float("inf")):
    '''
    :param wt_vi_traj_candidates: Nested list of [weight, value iteration object, trajectory]
    :param BEC_constraints: Minimum set of constraints defining the BEC of a set of demos / policy (list of constraints)
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost

    :return: summary: Nested list of [mdp, trajectory]

    Summary: Obtain a set of efficient demonstrations that recovers the behavioral equivalence class (BEC) of a set of demos / policy.
    An modified implementation of 'Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications' (Brown et al. AAAI 2019).
    '''
    min_BEC_summary = []
    summary = []

    if summary_variant[0] == 'random':
        # select a random set of training examples
        best_idxs = []
        while len(best_idxs) < n_train_demos:
            r = random.randint(0, len(min_subset_constraints_record) - 1)
            if r not in best_idxs: best_idxs.append(r)

        for best_idx in best_idxs:
            best_env_idx = env_record[best_idx]

            # record information associated with the best selected summary demo
            best_traj = traj_record[best_idx]
            best_mdp = wt_vi_traj_candidates[best_env_idx][0][1].mdp
            constraints_added = min_subset_constraints_record[best_idx]
            summary.append([best_mdp, best_traj, constraints_added])
        return summary

    # if you're looking for demonstrations that will convey the most constraining BEC region or will be employing scaffolding,
    # obtain the demos needed to convey the most constraining BEC region
    if summary_variant[0] == 'highest' or summary_variant[0] == 'high' or summary_variant[0] == 'forward' or summary_variant[0] == 'backward':
        BEC_constraints = min_BEC_constraints
        BEC_constraint_bookkeeping = BEC_helpers.perform_BEC_constraint_bookkeeping(BEC_constraints,
                                                                                    min_subset_constraints_record)

        # extract sets of demos+environments pairs that can cover each BEC constraint
        sets = []
        for constraint_idx in range(BEC_constraint_bookkeeping.shape[1]):
            sets.append(np.argwhere(BEC_constraint_bookkeeping[:, constraint_idx] == 1).flatten().tolist())

        # downsample some sets with too many members for computational feasibility
        for j, set in enumerate(sets):
            if len(set) > downsample_threshold:
                sets[j] = random.sample(set, downsample_threshold)

        # find all combination of demos+environments pairs that cover all BEC constraints
        filtered_combo = []
        for combination in itertools.product(*sets):
            filtered_combo.append(np.unique(combination))

        # consider the visual similarity between environments, and the complexity and BEC length of individual demonstrations
        # when constructing the set of summaries
        visual_dissimilarities = np.zeros(len(filtered_combo))
        complexities = np.zeros(len(filtered_combo))
        BEC_lengths = np.zeros(len(filtered_combo))  # an average of BEC lengths of constituent demos of a summary
        for j, combo in enumerate(filtered_combo):
            visual_dissimilarity = 0
            if len(combo) >= 2:
                # compare every possible pairing of this set of demos
                pairs = list(itertools.combinations(combo, 2))
                for pair in pairs:
                    # get similar demos
                    if summary_variant[1] == 'high':
                        visual_dissimilarity += wt_vi_traj_candidates[
                            env_record[pair[0]]][0][1].mdp.measure_visual_dissimilarity(traj_record[pair[0]][0][0],
                                                                                        wt_vi_traj_candidates[
                                                                                            env_record[pair[1]]][0][1].mdp,
                                                                                        traj_record[pair[1]][0][0])
                    else:
                        # get dissimilar demos
                        visual_dissimilarity += -wt_vi_traj_candidates[
                            env_record[pair[0]]][0][1].mdp.measure_visual_dissimilarity(traj_record[pair[0]][0][0],
                                                                                        wt_vi_traj_candidates[
                                                                                            env_record[pair[1]]][0][1].mdp,
                                                                                        traj_record[pair[1]][0][0])
                visual_dissimilarities[j] = visual_dissimilarity / len(pairs)

            complexity = 0
            BEC_length = 0

            for env in combo:
                if summary_variant[1] == 'high':
                    # get demos of low visual complexity
                    complexity += wt_vi_traj_candidates[env_record[env]][0][1].mdp.measure_env_complexity()
                    # get simple demos
                    # BEC_length += \
                    # BEC_helpers.calculate_BEC_length(min_subset_constraints_record[env], weights, step_cost_flag)[0]
                else:
                    # get demos of high visual complexity
                    complexity += -wt_vi_traj_candidates[env_record[env]][0][1].mdp.measure_env_complexity()
                    # get complex demos
                    # BEC_length += \
                    # -BEC_helpers.calculate_BEC_length(min_subset_constraints_record[env], weights, step_cost_flag)[0]

            complexities[j] = complexity / len(combo)
            # large BEC length correlates to simplicity
            BEC_lengths[j] = -BEC_length / len(combo)

        tie_breaker = np.arange(len(filtered_combo))
        sorted_zipped = sorted(zip(visual_dissimilarities, complexities, BEC_lengths, tie_breaker, filtered_combo))
        visual_dissimilarities_sorted, complexities_sorted, BEC_lengths_sorted, _, filtered_combo_sorted = list(
            zip(*sorted_zipped))

        best_idxs = filtered_combo_sorted[0]

        for best_idx in best_idxs:
            best_env_idx = env_record[best_idx]

            # record information associated with the best selected summary demo
            best_traj = traj_record[best_idx]
            # todo: I should also change the initial state of the mdp to be the first state of the trajectory for completeness
            # (as I have already done for the test environments / demonstrations)
            best_mdp = wt_vi_traj_candidates[best_env_idx][0][1].mdp
            constraints_added = min_subset_constraints_record[best_idx]
            min_BEC_summary.append([best_mdp, best_traj, constraints_added])

        for best_idx in sorted(best_idxs, reverse=True):
            del min_subset_constraints_record[best_idx]
            del traj_record[best_idx]
            del env_record[best_idx]
            del BEC_lengths_record[best_idx]

    # fill out the rest of the vacant demo slots
    included_demo_idxs = []
    if len(min_BEC_summary) < n_train_demos:
        covering_demos_idxs = obtain_potential_summary_demos(BEC_lengths_record, n_train_demos - len(min_BEC_summary), type=summary_variant)

        for covering_demo_idxs in covering_demos_idxs:
            #remove demos that have already been included
            duplicate_idxs = []
            for covering_idx, covering_demo_idx in enumerate(covering_demo_idxs):
                for included_demo_idx in included_demo_idxs:
                    if included_demo_idx == covering_demo_idx:
                        duplicate_idxs.append(covering_idx)

            for duplicate_idx in sorted(duplicate_idxs, reverse=True):
                del covering_demo_idxs[duplicate_idx]

            visual_dissimilarities = np.zeros(len(covering_demo_idxs))
            complexities = np.zeros(len(covering_demo_idxs))
            for j, covering_demo_idx in enumerate(covering_demo_idxs):

                # only compare the visual dissimilarity to the most recent summary
                if len(summary) > 0:
                    if summary_variant[0] == 'forward' or summary_variant[0] == 'backward':
                        if summary_variant[1] == 'high':
                            # get similar demos
                            visual_dissimilarities[j] = wt_vi_traj_candidates[env_record[covering_demo_idx]][0][
                                1].mdp.measure_visual_dissimilarity(
                                traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])
                        else:
                            # get dissimilar demos
                            visual_dissimilarities[j] = -wt_vi_traj_candidates[env_record[covering_demo_idx]][0][
                                1].mdp.measure_visual_dissimilarity(
                                traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])
                    else:
                        if summary_variant[1] == 'high':
                            # get dissimilar demos for diversity
                            visual_dissimilarities[j] = -wt_vi_traj_candidates[env_record[covering_demo_idx]][0][
                                1].mdp.measure_visual_dissimilarity(
                                traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])
                        else:
                            # get similar demos for redundancy
                            visual_dissimilarities[j] = wt_vi_traj_candidates[env_record[covering_demo_idx]][0][
                                1].mdp.measure_visual_dissimilarity(
                                traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])

                if summary_variant[1] == 'high':
                    # get demos of low visual complexity
                    complexities[j] = wt_vi_traj_candidates[env_record[covering_demo_idx]][0][1].mdp.measure_env_complexity()
                else:
                    # get demos of high visual complexity
                    complexities[j] = -wt_vi_traj_candidates[env_record[covering_demo_idx]][0][1].mdp.measure_env_complexity()

            tie_breaker = np.arange(len(covering_demo_idxs))
            # sorts from small to large values
            sorted_zipped = sorted(zip(visual_dissimilarities, complexities, tie_breaker, covering_demo_idxs))
            visual_dissimilarities_sorted, complexities_sorted, _, covering_demo_idxs_sorted = list(
                zip(*sorted_zipped))

            best_idxs = [covering_demo_idxs_sorted[0]]
            included_demo_idxs.extend(best_idxs)

            for best_idx in best_idxs:
                best_env_idx = env_record[best_idx]

                # record information associated with the best selected summary demo
                best_traj = traj_record[best_idx]
                best_mdp = wt_vi_traj_candidates[best_env_idx][0][1].mdp
                constraints_added = min_subset_constraints_record[best_idx]
                summary.append([best_mdp, best_traj, constraints_added])

                if len(min_BEC_summary) + len(summary) >= n_train_demos:
                    summary.extend(min_BEC_summary)
                    # flip the order of the summary from highest information / hardest to lowest information / easiest
                    if summary_variant[0] == 'backward':
                        summary.reverse()
                    return summary

    summary.extend(min_BEC_summary)
    # flip the order of the summary from highest information / hardest to lowest information / easiest
    if summary_variant[0] == 'backward':
        summary.reverse()
    return summary

def visualize_constraints(constraints, weights, step_cost_flag, plot_lim=[(-1, 1), (-1, 1)], scale=1.0, fig_name=None, just_save=False):
    '''
    Summary: Visualize the constraints. Use scale to determine whether to show L1 normalized weights and constraints or
    weights and constraints where the step cost is 1.
    '''
    # This visualization function is currently specialized to handle plotting problems with two unknown weights or two
    # unknown and one known weight. For higher dimensional constraints, this visualization function must be updated.

    plt.xlim(plot_lim[0][0] * scale, plot_lim[0][1] * scale)
    plt.ylim(plot_lim[1][0] * scale, plot_lim[1][1] * scale)

    wt_shading = 1. / len(constraints)

    if step_cost_flag:
        # if the final weight is the step cost, it is assumed that there are three weights, which must be accounted for
        # differently to plot the BEC region for the first two weights
        for constraint in constraints:
            if constraint[0, 1] == 0.:
                # completely vertical line going through zero
                pt = (-weights[0, -1] * constraint[0, 2]) / constraint[0, 0]
                plt.plot(np.array([pt, pt]) * scale, np.array([-1, 1]) * scale)

                # use (1, 0) as a test point to decide which half space to color
                if constraint[0, 0] + (weights[0, -1] * constraint[0, 2]) >= 0:
                    # color the right side of the line
                    plt.axvspan(pt * scale, 1 * scale, alpha=wt_shading, color='blue')
                else:
                    # color the left side of the line
                    plt.axvspan(-1 * scale, pt * scale, alpha=wt_shading, color='blue')
            else:
                pt_1 = (constraint[0, 0] - (weights[0, -1] * constraint[0, 2])) / constraint[0, 1]
                pt_2 = (-constraint[0, 0] - (weights[0, -1] * constraint[0, 2])) / constraint[0, 1]
                plt.plot(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale)

                # use (0, 1) as a test point to decide which half space to color
                if constraint[0, 1] + (weights[0, -1] * constraint[0, 2]) >= 0:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([1, 1]) * scale, alpha=wt_shading, color='blue')
                else:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([-1, -1]) * scale, alpha=wt_shading, color='blue')

        # visualize the L1 norm == 1 constraints
        plt.plot(np.array([-1 + abs(weights[0, -1]), 0]) * scale, np.array([0, 1 - abs(weights[0, -1])]) * scale, color='grey')
        plt.plot(np.array([0, 1 - abs(weights[0, -1])]) * scale, np.array([1 - abs(weights[0, -1]), 0]) * scale, color='grey')
        plt.plot(np.array([1 - abs(weights[0, -1]), 0]) * scale, np.array([0, -1 + abs(weights[0, -1])]) * scale, color='grey')
        plt.plot(np.array([0, -1 + abs(weights[0, -1])]) * scale, np.array([-1 + abs(weights[0, -1]), 0] )* scale, color='grey')
    else:
        for constraint in constraints:
            if constraint[0, 0] == 1.:
                # completely vertical line going through zero
                plt.plot(np.array([constraint[0, 1] / constraint[0, 0], -constraint[0, 1] / constraint[0, 0]]) * scale, np.array([-1, 1]) * scale)

                # use (1, 0) as a test point to decide which half space to color
                if constraint[0, 0] >= 0:
                    # color the right side of the line
                    plt.axvspan(0 * scale, 1 * scale, alpha=wt_shading, color='blue')
                else:
                    # color the left side of the line
                    plt.axvspan(-1 * scale, 0 * scale, alpha=wt_shading, color='blue')
            else:
                pt_1 = constraint[0, 0] / constraint[0, 1]
                pt_2 = -constraint[0, 0] / constraint[0, 1]
                plt.plot(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale)

                # use (0, 1) as a test point to decide which half space to color
                if constraint[0, 1] >= 0:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([1, 1]) * scale, alpha=wt_shading, color='blue')
                else:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([-1, -1]) * scale, alpha=wt_shading, color='blue')

    wt_marker_size = 200
    # plot ground truth weight
    plt.scatter(weights[0, 0] * scale, weights[0, 1] * scale, s=wt_marker_size, color='red', zorder=2)
    # plt.title('Area of Viable Reward Weights')
    plt.xlabel(r'$w_0$')
    plt.ylabel(r'$w_1$')
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=200, transparent=True)
    if not just_save:
        plt.show()
    plt.clf()


def visualize_summary(BEC_summaries_collection, weights, step_cost_flag):
    '''
    Summary: visualize the BEC demonstrations
    '''
    min_BEC_constraints_running = []
    for summary_idx, BEC_summary in enumerate(BEC_summaries_collection):
        print("Showing demo {} out of {}".format(summary_idx + 1, len(BEC_summaries_collection)))
        # print('BEC_length: {}'.format(BEC_helpers.calculate_BEC_length(BEC_summary[2], weights, step_cost_flag)[0]))

        # visualize demonstration
        BEC_summary[0].visualize_trajectory(BEC_summary[1])

        # visualize the min BEC constraints of this particular demonstration
        # visualize_constraints(BEC_summary[2], weights, step_cost_flag)
        # visualize_constraints(BEC_summary[2], weights, step_cost_flag, fig_name=str(summary_idx) + '.png', scale=abs(1 / weights[0, -1]))

        # visualize the min BEC constraints extracted from all demonstrations shown thus far
        # min_BEC_constraints_running.extend(BEC_summary[2])
        # min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)[0]
        # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag)
        # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, fig_name=str(summary_idx) + '.png', scale=abs(1 / weights[0, -1]))

        # visualize what the human would've done in this environment (giving the human the benefit of the doubt)
        print(colored('Visualizing human counterfactuals', 'blue'))
        # visualize the counterfactual trajectory at every (s,a) pair along the agent's optimal trajectory
        # for human_opt_traj in BEC_summary[3]:
        #     BEC_summary[0].visualize_trajectory(human_opt_traj) # the environment shoudld be the same for agent and human
        # only visualize the first counterfactual trajectory (often the more informative)
        BEC_summary[0].visualize_trajectory(BEC_summary[3][0])


def visualize_test_envs(test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, weights, step_cost_flag):
    for j, test_wt_vi_traj_tuple in enumerate(test_wt_vi_traj_tuples):
        print('Visualizing test environment {} with BEC length of {}'.format(j, test_BEC_lengths[j]))

        vi_candidate = test_wt_vi_traj_tuple[1]
        trajectory_candidate = test_wt_vi_traj_tuple[2]
        vi_candidate.mdp.visualize_trajectory(trajectory_candidate)
        # visualize_constraints(test_BEC_constraints[j], weights, step_cost_flag)
