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

def extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, BEC_depth=1, trajectories=None, print_flag=False):
    '''
    :param wt_vi_traj_candidates: Nested list of [weight, value iteration object, trajectory]
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :param BEC_depth (int): number of suboptimal actions to take before following the optimal policy to obtain the
                            suboptimal trajectory (and the corresponding suboptimal expected feature counts)
    :return: min_subset_constraints: List of constraints

    Summary: Obtain the minimum BEC constraints for each environment
    '''
    min_subset_constraints_record = []    # minimum BEC constraints conveyed by a trajectory
    env_record = []
    policy_constraints = []               # BEC constraints that define a policy (i.e. constraints arising from one action
                                          # deviations from every possible starting state and the corresponding optimal trajectories)
    traj_record = []
    processed_envs = []

    # go through each environment and corresponding optimal trajectory, and extract the behavior equivalence class (BEC) constraints
    for env_idx, wt_vi_traj_candidate in enumerate(wt_vi_traj_candidates):
        if print_flag:
            print("Extracting constraints from environment {}".format(env_idx))
        mdp = wt_vi_traj_candidate[0][1].mdp
        agent = FixedPolicyAgent(wt_vi_traj_candidate[0][1].policy)

        if trajectories is not None:
            constraints = []
            # a) demonstration-driven BEC
            # BEC constraints are obtained by ensuring that the optimal actions accumulate at least as much reward as
            # all other possible actions along a trajectory
            action_seq_list = list(itertools.product(mdp.actions, repeat=BEC_depth))

            traj_opt = trajectories[env_idx]
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
        else:
            # b) policy-driven BEC
            # wt_vi_traj_candidates can contain MDPs with the same environment but different initial states (to
            # accommodate demo BEC). by considering all reachable states of two identical MDPs with different initial
            # states, you will obtain duplicate test environments so only go through each MDP once for policy BEC.
            if mdp.env_code not in processed_envs:
                agent = FixedPolicyAgent(wt_vi_traj_candidate[0][1].policy)

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
                            policy_constraints.append(BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag))

                    # also store the BEC constraints for optimal trajectory in each state, along with the associated
                    # demo and environment number
                    min_subset_constraints_record.append(BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag))
                    traj_record.append(traj_opt)
                    env_record.append(env_idx)

                processed_envs.append(mdp.env_code)

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

def obtain_summary_counterfactual(summary_variant, wt_vi_traj_candidates, min_BEC_constraints, BEC_lengths_record,
                       min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=3,
                       downsample_threshold=float("inf"), pad_factor=0.02):

    summary = []
    feature_flag = 0

    # # bootstrap model of human's current understanding with a simple demo
    # a) select this simple demo on the fly
    # n_clusters = 6
    # cluster_idxs = [4, 2, 0]
    #
    # kmeans = KMeans(n_clusters=n_clusters).fit(np.array(BEC_lengths_record).reshape(-1, 1))
    # cluster_centers = kmeans.cluster_centers_
    # labels = kmeans.labels_
    #
    # ordering = np.arange(0, n_clusters)
    # sorted_zipped = sorted(zip(cluster_centers, ordering))
    # cluster_centers_sorted, ordering_sorted = list(zip(*sorted_zipped))
    #
    # partition_idx = ordering_sorted[cluster_idxs[len(summary)]]
    # covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
    # best_idx = random.sample(covering_demo_idxs, 1)[0]

    # # b) load up a previously selected simple demo
    # best_idx = 16460
    #
    # best_env_idx = env_record[best_idx]
    # # record information associated with the best selected summary demo
    # best_traj = traj_record[best_idx]
    # best_mdp = wt_vi_traj_candidates[best_env_idx][0][1].mdp
    # constraints_added = min_subset_constraints_record[best_idx]
    # min_BEC_constraints_running = constraints_added               # this is the running model of the human
    # summary.append([best_mdp, best_traj, constraints_added])
    #
    # del traj_record[best_idx]
    # del env_record[best_idx]
    #
    # print('Best idx: {}'.format(best_idx))

    # c) assuming a starting human model without any inducing demonstrations. include a dummy constraint that doesn't
    # reduce any BEC area so that the BEC area calculation doesn't break later
    min_BEC_constraints_running = [np.array([[.01, 0, -1]])]
    # also hardcoding the human's model for now for development
    with open('models/augmented_taxi/wt_vi_traj_candidates_human.pickle', 'rb') as f:
        wt_vi_traj_candidates_human = pickle.load(f)

    while len(summary) < n_train_demos:
        # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, scale=abs(1 / weights[0, -1]), fig_name=str(len(summary)) + '.png')
        # c) hardcoding the human's model for now
        w_human = np.array([[26, 0, -1]])
        w_human_normalized = w_human / np.linalg.norm(w_human[0, :], ord=1)
        # _, _, w_human_normalized = BEC_helpers.calculate_BEC_length(min_BEC_constraints_running, weights, step_cost_flag, return_midpt=True)
        print(w_human_normalized * abs(1 / weights[0, -1]))

        constraints_running = []
        info_gains = np.zeros(len(traj_record))
        human_counterfactual_trajs_record = [] # save the best trajectories of the human's model (given the benefit of the doubt)

        print("Length of summary: {}".format(len(summary)))

        # code to run through and recalculate the human's optimal policy (given new weights)
        # wt_vi_traj_candidates_human = copy.deepcopy(wt_vi_traj_candidates)
        # for idx, candidate in enumerate(wt_vi_traj_candidates_human):
        #     print(idx)
        #     mdp = candidate[0][1].mdp
        #     mdp.weights = w_human_normalized
        #     vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=50)
        #     vi_human.run_vi()
        #     trajectory = mdp_helpers.rollout_policy(mdp, vi_human)
        #     candidate[0][0] = w_human_normalized
        #     candidate[0][1] = vi_human
        #     candidate[0][2] = trajectory
        #     candidate[0][3]['weights'] = w_human_normalized
        # can save the human's policy and reuse if it you're fixing it (as is done currently), or if you've already
        # decided on a starting demo (as in b), and wish to skip one iteration of solving for the human's policy)
        # with open('models/augmented_taxi/wt_vi_traj_candidates_human.pickle', 'wb') as f:
        #     pickle.dump(wt_vi_traj_candidates_human, f)

        for idx in range(len(traj_record)):
            if idx % 1000 == 0:
                print("{}/{}".format(idx, len(traj_record)))

            traj_opt = traj_record[idx]
            agent = wt_vi_traj_candidates[env_record[idx]][0][1]
            mdp = agent.mdp

            constraints = []
            vi_human = wt_vi_traj_candidates_human[env_record[idx]][0][1]

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

                # obtain all optimal trajectory rollouts according to the human's model
                human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])

                cur_best_reward = np.float('-inf')
                best_reward_features = []
                best_human_traj = []
                # select the human's possible trajectory that has the highest true reward (i.e. give the human's policy the benefit of the doubt)
                for traj in human_opt_trajs:
                    mu_sb = mdp.accumulate_reward_features(traj, discount=True) # the human and agent should be working with identical mdps
                    reward_hyp = mdp.weights.dot(mu_sb.T)
                    if reward_hyp > cur_best_reward:
                        cur_best_reward = reward_hyp
                        best_reward_features = mu_sb
                        best_human_traj = traj

                constraints.append(mu_sa - best_reward_features)
                best_human_trajs_record.append(best_human_traj)

            human_counterfactual_trajs_record.append(best_human_trajs_record)

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
                info_gains[idx] = BEC_helpers.calculate_BEC_length(min_BEC_constraints_running, weights, step_cost_flag)[0] - BEC_helpers.calculate_BEC_length(hypothetical_constraints, weights, step_cost_flag)[0]

                constraints_running.append(constraints)
            except:
                # print("No valid constraints")
                pass
            # mdp.visualize_trajectory(traj_opt)
            # vi_human.mdp.visualize_trajectory(traj_hyp)

        # no need to continue search for demonstrations if none of them will improve the human's understanding
        if np.sum(info_gains) == 0:
            if feature_flag == 2:
                break
            else:
                # move onto conveying the next feature
                feature_flag += 1
                print(colored('Incrementing feature flag, which is now at {}'.format(feature_flag), 'red'))
                continue

        # todo: let's just try returning the smallest information gain for now (so that we can have incremental demonstrations)
        # there often isn't enough demonstrations to generate 6 clusters, as I initially tried to do below
        best_idx = np.argwhere(info_gains == min(i for i in info_gains if i > 0))[0][0] # info gain should be greater than 0
        print(colored('Max infogain: {}'.format(np.max(info_gains)), 'blue'))
        print(colored('Max Min infogain: {}'.format(min(i for i in info_gains if i > 0)), 'blue')) # smallest infogain above zero
        print(colored('Min infogain: {}'.format(np.min(info_gains)), 'blue'))

        # kmeans = KMeans(n_clusters=n_clusters).fit(np.array(info_gains).reshape(-1, 1))
        # cluster_centers = kmeans.cluster_centers_
        # labels = kmeans.labels_
        #
        # ordering = np.arange(0, n_clusters)
        # sorted_zipped = sorted(zip(cluster_centers, ordering))
        # cluster_centers_sorted, ordering_sorted = list(zip(*sorted_zipped))
        #
        # partition_idx = ordering_sorted[cluster_idxs[len(summary)]]
        # covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
        # best_idx = random.sample(covering_demo_idxs, 1)[0]

        best_env_idx = env_record[best_idx]
        # record information associated with the best selected summary demo
        best_traj = traj_record[best_idx]
        best_human_trajs = human_counterfactual_trajs_record[best_idx]
        best_mdp = wt_vi_traj_candidates[best_env_idx][0][1].mdp
        min_BEC_constraints_running.extend(constraints_running[best_idx])
        summary.append([best_mdp, best_traj, constraints_running[best_idx], best_human_trajs])

        del traj_record[best_idx]
        del env_record[best_idx]

        # this method doesn't always finish, so save the summary along the way
        with open('models/augmented_taxi/BEC_summary.pickle', 'wb') as f:
            pickle.dump(summary, f)

    # # manual exploration of two sets of demonstrations that have equal BEC lengths but appear to differ in informativeness
    # # for idx in [1367, 1368, 8824, 8835]:
    # for idx in [110, 1210]:
    #     print(idx)
    #     traj_opt = traj_record[idx]
    #     agent = wt_vi_traj_candidates[env_record[idx]][0][1]
    #     mdp = agent.mdp
    #
    #     # if desired, also visualize the BEC constraints of the agent's optimal demonstration
    #     visualize_constraints(min_subset_constraints_record[idx], weights, step_cost_flag, scale=abs(1 / weights[0, -1]),
    #                           fig_name=str(idx) + '_BEC_raw.png')
    #
    #     # solve for the human's optimal trajectory
    #     mdp_human = copy.deepcopy(mdp)
    #     mdp_human.set_init_state(traj_opt[0][0])
    #     w_human = np.array([[26, 0, -1]])
    #     mdp_human.weights = w_human / np.linalg.norm(w_human[0, :], ord=1)
    #     vi_human = ValueIteration(mdp_human, sample_rate=1, max_iterations=50)
    #     vi_human.run_vi()
    #
    #     constraints = []
    #
    #     # # a) accumulate the reward features and generate a single constraint
    #     # # reward features of optimal action
    #     # mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
    #     # traj_hyp = mdp_helpers.rollout_policy(mdp_human, vi_human)
    #     # mu_sb = mdp_human.accumulate_reward_features(traj_hyp, discount=True)
    #     # constraints.append(mu_sa - mu_sb)
    #
    #     # b) contrast differing expected feature counts for each state-action pair along the agent's optimal trajectory
    #     for sas_idx in range(len(traj_opt)):
    #         # reward features of optimal action
    #         mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)
    #
    #         sas = traj_opt[sas_idx]
    #         cur_state = sas[0]
    #
    #         # currently assumes that all actions are executable from all states
    #         traj_hyp = mdp_helpers.rollout_policy(mdp_human, vi_human, cur_state)
    #         mu_sb = mdp_human.accumulate_reward_features(traj_hyp, discount=True)
    #
    #         constraints.append(mu_sa - mu_sb)
    #
    #         # mdp_human.visualize_trajectory(traj_hyp)
    #
    #     # # c) contrast differing expected feature counts for each state-action pair along the human's optimal trajectory
    #     # cur_state = traj_opt[0][0]
    #     # traj_human = mdp_helpers.rollout_policy(mdp_human, vi_human, cur_state)
    #     # for sas_idx in range(len(traj_human)):
    #     #     # reward features of optimal action
    #     #     mu_sb = mdp.accumulate_reward_features(traj_human[sas_idx:], discount=True)
    #     #
    #     #     sas = traj_human[sas_idx]
    #     #     cur_state = sas[0]
    #     #
    #     #     # currently assumes that all actions are executable from all states
    #     #     traj_opt = mdp_helpers.rollout_policy(mdp, agent, cur_state=cur_state)
    #     #     mu_sa = mdp_human.accumulate_reward_features(traj_opt, discount=True)
    #     #
    #     #     constraints.append(mu_sa - mu_sb)
    #
    #     try:
    #         constraints = BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag)
    #         visualize_constraints(constraints, weights, step_cost_flag, scale=abs(1 / weights[0, -1]), fig_name=str(idx) + '.png')
    #         print(BEC_helpers.calculate_BEC_length(constraints, weights, step_cost_flag)[0])
    #     except:
    #         print("No valid constraints")
    #
    #     print(constraints)
    #     # if desired, visualize the agent's and human's optimal trajectories
    #     mdp.visualize_trajectory(traj_opt)
    #     # mdp_human.visualize_trajectory(traj_hyp)

    return summary


def obtain_summary(summary_variant, wt_vi_traj_candidates, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=3, downsample_threshold=float("inf"), pad_factor=0.02):
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

def visualize_constraints(constraints, weights, step_cost_flag, plot_lim=[(-1, 1), (-1, 1)], scale=1.0, fig_name=None):
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
    plt.show()


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
