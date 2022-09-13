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
from policy_summarization import policy_summarization_helpers as ps_helpers
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
from spherical_geometry import polygon as sph_polygon
import policy_summarization.BEC_visualization as BEC_viz
from policy_summarization import computational_geometry as cg

def extract_constraints_policy(args):
    env_idx, data_loc, step_cost_flag = args
    with open(mp_helpers.lookup_env_filename(data_loc, env_idx), 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    mdp = wt_vi_traj_env[0][1].mdp
    agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
    weights = mdp.weights

    min_subset_constraints_record = []    # minimum BEC constraints conveyed by a trajectory
    policy_constraints = []               # BEC constraints that define a policy (i.e. constraints arising from one action
                                          # deviations from every possible starting state and the corresponding optimal trajectories)
    traj_record = []
    reward_record = []                    # the rewards associated with the optimal trajectories

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
                    BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag))
                reward_record.append(weights.dot(mu_sa.T))

        # also store the BEC constraints for optimal trajectory in each state, along with the associated
        # demo and environment number
        min_subset_constraints_record.append(
            BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag))
        traj_record.append(traj_opt)

    return env_idx, traj_record, policy_constraints, min_subset_constraints_record, reward_record

def extract_constraints_demonstration(args):
    env_idx, vi, traj_opt, step_cost_flag = args

    min_subset_constraints_record = []    # minimum BEC constraints conveyed by a trajectory
    policy_constraints = []               # BEC constraints that define a policy (i.e. constraints arising from one action
                                          # deviations from every possible starting state and the corresponding optimal trajectories)
    traj_record = []
    reward_record = []                    # the rewards associated with the optimal trajectories

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

        if sas_idx == 0:
            reward_record.append(weights.dot(mu_sa.T))

    # store the BEC constraints for each environment, along with the associated demo and environment number
    min_subset_constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)
    min_subset_constraints_record.append(min_subset_constraints)
    traj_record.append(traj_opt)
    # slightly abusing the term 'policy' here since I'm only considering a subset of possible trajectories (i.e.
    # demos) that the policy can generate in these environments
    policy_constraints.append(min_subset_constraints)

    return env_idx, traj_record, policy_constraints, min_subset_constraints_record, reward_record


def extract_constraints(data_loc, step_cost_flag, pool, vi_traj_triplets=None, print_flag=False):
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
    reward_record = []

    n_envs = len(os.listdir('models/' + data_loc + '/gt_policies/'))

    print("Extracting the BEC constraints in each environment:")
    pool.restart()
    if vi_traj_triplets is None:
        # a) policy-driven BEC: generate constraints by considering the expected feature counts after taking one
        # suboptimal action in every possible state in the state space, then acting optimally afterward. see eq 13, 14
        # of Brown et al. 'Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications' 2019
        args = [(i, data_loc, step_cost_flag) for i in range(n_envs)]
        results = list(tqdm(pool.imap(extract_constraints_policy, args), total=len(args)))
        pool.close()
        pool.join()
        pool.terminate()

        # determine whether this domain has a consistent set of states between constituent MDPs
        consistent_state_count = True
        previous_state_count = None
        for result in results:
            if previous_state_count == None:
                previous_state_count = len(result[1])
            else:
                if previous_state_count != len(result[1]):
                    consistent_state_count = False
                else:
                    previous_state_count = len(result[1])

            env_record.append(result[0])
            traj_record.append(result[1])
            policy_constraints.extend(result[2])
            min_subset_constraints_record.append(result[3])
            reward_record.append(result[4])
    else:
        # b) demonstration-driven BEC: generate constraints by considering the expected feature counts after taking one
        # suboptimal action in every state along a trajectory (demonstration), then acting optimally afterward.
        # see eq 16 of Brown et al. 'Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications' 2019
        # need to specify the environment idx, environment, and corresponding optimal trajectories (first, second, and
        # third elements of vi_traj_triplet, respectively) that you want to extract constraints from
        args = [(vi_traj_triplet[0], vi_traj_triplet[1], vi_traj_triplet[2], step_cost_flag) for vi_traj_triplet in vi_traj_triplets]
        results = list(tqdm(pool.imap(extract_constraints_demonstration, args), total=len(args)))
        pool.close()
        pool.join()
        pool.terminate()

        for result in results:
            env_record.append(result[0])
            traj_record.append(result[1])
            policy_constraints.extend(result[2])
            min_subset_constraints_record.append(result[3])
            reward_record.append(result[4])

        # this isn't really relevant for demonstration BEC
        consistent_state_count = False

    return policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count

def extract_BEC_constraints(policy_constraints, min_subset_constraints_record, env_record, weights, step_cost_flag, pool):
    '''
    Summary: Obtain the minimum BEC constraints across all environments / demonstrations
    '''
    constraints_record = [item for sublist in policy_constraints for item in sublist]

    # first obtain the absolute min BEC constraint
    min_BEC_constraints = BEC_helpers.remove_redundant_constraints(constraints_record, weights, step_cost_flag)

    # then determine the BEC lengths of all other potential demos that could be shown
    BEC_lengths_record = []

    if step_cost_flag:
        # calculate the 2D intersection between minimum constraints and L1 norm constraint for each demonstration
        min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
        for j, min_subset_constraints in enumerate(min_subset_constraints_record_flattened):
            BEC_lengths_record.append(BEC_helpers.calculate_BEC_length(min_subset_constraints, weights, step_cost_flag)[0])
    else:
        # calculate the solid angle between the minimum constraints for each demonstration
        pool.restart()
        BEC_lengths_record = list(tqdm(pool.imap(BEC_helpers.calc_solid_angles, min_subset_constraints_record), total=len(min_subset_constraints_record)))
        pool.close()
        pool.join()
        pool.terminate()

    return min_BEC_constraints, BEC_lengths_record

def obtain_potential_summary_demos(BEC_lengths_record, n_demos, n_clusters, type='scaffolding', sample_count=None):
    if sample_count is None:
        # to allow for some variability in the demonstrations that are yielded
        sample_count = int(len(BEC_lengths_record) / 4)

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
            # print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)
    elif type[0] == 'medium':
        cluster_idx = 2
        partition_idx = ordering_sorted[cluster_idx]

        for j in range(n_demos):
            covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
            # print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)
    elif type[0] == 'high':
        cluster_idx = 0
        partition_idx = ordering_sorted[cluster_idx]

        for j in range(n_demos):
            covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
            # print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)
    else:
        # employ scaffolding

        # 0 is the cluster with the smallest BEC lengths (create 2n clusters to be able to skip every other)
        cluster_idxs = list(range(0, n_clusters, 2))

        for j in range(n_demos):
            # checking out partitions one at a time
            partition_idx = ordering_sorted[cluster_idxs[j]]

            covering_demo_idxs = [i for i, x in enumerate(labels) if x == partition_idx]
            # print('# covering demos: {}'.format(len(covering_demo_idxs)))

            covering_demo_idxs = random.sample(covering_demo_idxs, min(sample_count, len(covering_demo_idxs)))
            covering_demos_idxs.append(covering_demo_idxs)

        # filled this from hardest to easiest demos, so flip
        covering_demos_idxs.reverse()

    return covering_demos_idxs

def compute_counterfactuals(args):
    data_loc, model_idx, env_idx, w_human_normalized, env_filename, trajs_opt, min_BEC_constraints_running, step_cost_flag, summary_len, variable_filter, consider_human_models_jointly = args

    with open(env_filename, 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    agent = wt_vi_traj_env[0][1]
    weights = agent.mdp.weights

    human = copy.deepcopy(agent)
    mdp = human.mdp
    mdp.weights = w_human_normalized
    vi_human = ValueIteration(mdp, sample_rate=1)
    vi_human.run_vi()

    # only consider counterfactual trajectories from human models whose value iteration have converged
    if vi_human.stabilized:
        best_human_trajs_record_env = []
        constraints_env = []
        info_gain_env = []
        human_rewards_env = []
        overlap_in_opt_and_counterfactual_traj_env = []

        for traj_idx, traj_opt in enumerate(trajs_opt):
            constraints = []

            # # a) accumulate the reward features and generate a single constraint
            # mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
            # traj_hyp = mdp_helpers.rollout_policy(vi_human.mdp, vi_human)
            # mu_sb = vi_human.mdp.accumulate_reward_features(traj_hyp, discount=True)
            # constraints.append(mu_sa - mu_sb)

            # b) contrast differing expected feature counts for each state-action pair along the agent's optimal trajectory
            best_human_trajs_record = []
            best_human_reward = 0
            for sas_idx in range(len(traj_opt)):
                # reward features of optimal action
                mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

                sas = traj_opt[sas_idx]
                cur_state = sas[0]

                # obtain all optimal trajectory rollouts according to the human's model (assuming that it's a reasonable policy that has converged)
                human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])

                cur_best_reward = float('-inf')
                best_reward_features = []
                best_human_traj = []
                # select the human's possible trajectory that has the highest true reward (i.e. give the human's policy the benefit of the doubt)
                for traj in human_opt_trajs:
                    mu_sb = mdp.accumulate_reward_features(traj, discount=True)  # the human and agent should be working with identical mdps
                    reward_hyp = weights.dot(mu_sb.T)
                    if reward_hyp > cur_best_reward:
                        cur_best_reward = reward_hyp
                        best_reward_features = mu_sb
                        best_human_traj = traj

                # only store the reward of the full trajectory
                if sas_idx == 0:
                    best_human_reward = cur_best_reward
                constraints.append(mu_sa - best_reward_features)
                best_human_trajs_record.append(best_human_traj)

            if len(constraints) > 0:
                constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)

            # don't consider environments that convey information about a variable you don't currently wish to convey
            skip_demo = False
            if variable_filter is not None:
                for constraint in constraints:
                    if abs(variable_filter.dot(constraint.T)[0, 0]) > 0:
                        # conveys information about variable designated to be filtered out, so block this demonstration
                        skip_demo = True

            if not skip_demo:
                info_gain = BEC_helpers.calculate_information_gain(min_BEC_constraints_running, constraints, weights, step_cost_flag)
            else:
                info_gain = 0

            human_rewards_env.append(best_human_reward)
            best_human_trajs_record_env.append(best_human_trajs_record)
            constraints_env.append(constraints)
            info_gain_env.append(info_gain)

            if not consider_human_models_jointly:
                # you should only consider the overlap for the first counterfactual human trajectory (as opposed to
                # counterfactual trajectories that could've arisen from states after the first state)
                overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(best_human_trajs_record[0], traj_opt)

                overlap_in_opt_and_counterfactual_traj_env.append(overlap_pct)

    # else just populate with dummy variables
    else:
        best_human_trajs_record_env = [[[]] for i in range(len(trajs_opt))]
        constraints_env = [[] for i in range(len(trajs_opt))]
        info_gain_env = [0 for i in range(len(trajs_opt))]
        if not consider_human_models_jointly:
            overlap_in_opt_and_counterfactual_traj_env = [float('inf') for i in range(len(trajs_opt))]
        human_rewards_env = [np.array([[0]]) for i in range(len(trajs_opt))]

    with open('models/' + data_loc + '/counterfactual_data_' + str(summary_len) + '/model' + str(model_idx) +
              '/cf_data_env' + str(env_idx).zfill(5) + '.pickle', 'wb') as f:
        pickle.dump((best_human_trajs_record_env, constraints_env, human_rewards_env), f)

    if consider_human_models_jointly:
        return info_gain_env
    else:
        return info_gain_env, overlap_in_opt_and_counterfactual_traj_env

def combine_limiting_constraints_IG(args):
    '''
    Summary: combine the most limiting constraints across all potential human models for each potential demonstration
    '''
    env_idx, n_sample_human_models, data_loc, curr_summary_len, weights, step_cost_flag, variable_filter,\
    trajs_opt, min_BEC_constraints_running = args

    info_gains_record = []
    min_env_constraints_record = []
    all_env_constraints = []

    # jointly consider the constraints generated by suboptimal trajectories by each human model
    for model_idx in range(n_sample_human_models):
        with open('models/' + data_loc + '/counterfactual_data_' + str(curr_summary_len) + '/model' + str(
                model_idx) + '/cf_data_env' + str(
            env_idx).zfill(5) + '.pickle', 'rb') as f:
            best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)
        all_env_constraints.append(constraints_env)

    all_env_constraints_joint = [list(itertools.chain.from_iterable(i)) for i in zip(*all_env_constraints)]
    # for each possible demonstration in each environment, find the non-redundant constraints across all human models
    # and use that to calculate the information gain for that demonstration
    for traj_idx in range(len(all_env_constraints_joint)):
        if len(all_env_constraints_joint[traj_idx]) > 1:
            min_env_constraints = BEC_helpers.remove_redundant_constraints(all_env_constraints_joint[traj_idx],
                                                                           weights, step_cost_flag)
        else:
            min_env_constraints = all_env_constraints_joint[traj_idx]

        min_env_constraints_record.append(min_env_constraints)

        # don't consider environments that convey information about a variable you don't currently wish to convey
        skip_demo = False
        if variable_filter is not None:
            for constraint in min_env_constraints:
                if abs(variable_filter.dot(constraint.T)[0, 0]) > 0:
                    # conveys information about variable designated to be filtered out, so block this demonstration
                    skip_demo = True

        if not skip_demo:
            ig = BEC_helpers.calculate_information_gain(min_BEC_constraints_running, min_env_constraints, weights, step_cost_flag)
            info_gains_record.append(ig)
        else:
            info_gains_record.append(0)

    # obtain the counterfactual human trajectories that could've given rise to the most limiting constraints and
    # how much it overlaps the agent's optimal trajectory
    human_counterfactual_trajs = [[] for i in range(len(min_env_constraints_record))]
    overlap_in_opt_and_counterfactual_traj = [[] for i in range(len(min_env_constraints_record))]
    overlap_in_opt_and_counterfactual_traj_avg = []

    for model_idx in range(n_sample_human_models):
        with open('models/' + data_loc + '/counterfactual_data_' + str(curr_summary_len) + '/model' + str(
                model_idx) + '/cf_data_env' + str(
            env_idx).zfill(5) + '.pickle', 'rb') as f:
            best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)

        # for each of the minimum constraint sets in each environment (with a unique starting state)
        for traj_idx, min_env_constraints in enumerate(min_env_constraints_record):
            in_minimum_set = False

            # see if any of the constraints generated using this human trajectory match any of those in the minimum constraint set
            for constraint in constraints_env[traj_idx]:
                for min_env_constraint in min_env_constraints:
                    # if there is a match, consider the overlap with the agent's optimal trajectory and store that trajectory
                    if BEC_helpers.equal_constraints(constraint, min_env_constraint):
                        in_minimum_set = True

                        # you should only consider the overlap for the first counterfactual human trajectory (as opposed to
                        # counterfactual trajectories that could've arisen from states after the first state)
                        overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(best_human_trajs_record_env[traj_idx][0], trajs_opt[traj_idx])

                        overlap_in_opt_and_counterfactual_traj[traj_idx].append(overlap_pct)
                        # store the required information for replaying the closest human counterfactual trajectory
                        human_counterfactual_trajs[traj_idx].append((curr_summary_len, model_idx, env_idx, traj_idx))
                        break

                # a counterfactual trajectory only needs to contribute one constraint in the minimal set for the
                # corresponding optimal trajectory to be considered for the summary
                if in_minimum_set:
                    break

    # take the average overlap across all counterfactual trajectories that contributed to the most limiting constraints
    for traj_idx, overlap_pcts in enumerate(overlap_in_opt_and_counterfactual_traj):
        if len(overlap_pcts) > 0:
            overlap_in_opt_and_counterfactual_traj_avg.append(np.mean(overlap_pcts))
        else:
            overlap_in_opt_and_counterfactual_traj_avg.append(0)

    return info_gains_record, min_env_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs

def overlap_demo_BEC_and_human_posterior(args):
    '''
    Summary: combine the most limiting constraints across all potential human models for each potential demonstration
    '''
    env_idx, n_sample_human_models, min_subset_constraints, prior, posterior, data_loc, counterfactual_folder_idx, weights, trajs_opt, step_cost_flag, pool = args

    human_model = posterior
    human_model = BEC_helpers.remove_redundant_constraints(human_model, weights, step_cost_flag)

    BEC_areas = []

    # obtain spherical polygon comprising the posterior (human's model)
    posterior_ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(human_model)
    posterior_poly = Polyhedron.Polyhedron(ieqs=posterior_ieqs)

    posterior_spherical_polygon_vertices = np.array(BEC_helpers.obtain_sph_polygon_vertices(posterior_poly, add_noise=True))  # add noise to help smooth calculation
    if len(posterior) > 1:
        # taking the mean of vertices only works if they're not all lying on a shared plane (else there is
        # ambiguity)
        inside_posterior_sph_polygon = cg.compute_average_point(posterior_spherical_polygon_vertices)
    else:
        # if there is only a single constraint/plane (i.e. a half sphere), you can simply take the plane normal
        # as the inside point (with some noise to prevent out of domain issues with trig function)
        inside_posterior_sph_polygon = human_model[0][0] + np.random.sample(3) * 0.001

    posterior_spherical_polygon_vertices = cg.sort_points_by_angle(posterior_spherical_polygon_vertices, inside_posterior_sph_polygon)
    posterior_sph_polygon = sph_polygon.SphericalPolygon(posterior_spherical_polygon_vertices, inside=tuple(inside_posterior_sph_polygon))

    # if abs(BEC_helpers.calc_solid_angles([human_model])[0] - posterior_sph_polygon.area()) > 1 or np.isnan(posterior_sph_polygon.area()):
    #     raise AssertionError("Too much deviation in prior BEC areas")

    # for each possible demonstration in each environment, find the overlap in area between the spherical polygon
    # comprising the posterior and the counterfactual constraints created by the demonstration
    for constraint_idx, constraints in enumerate(min_subset_constraints):
        constraints_copy = constraints.copy()

        if len(constraints_copy) == 0:
            print(colored("NO CONSTRAINTS", 'red'))
            BEC_areas.append(posterior_sph_polygon.area())
        else:
            if len(constraints_copy) > 1:
                constraints_copy = BEC_helpers.remove_redundant_constraints(constraints_copy,
                                                                            weights, step_cost_flag)

            traj_ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_copy)
            traj_poly = Polyhedron.Polyhedron(ieqs=traj_ieqs)  # automatically finds the minimal H-representation
            traj_spherical_polygon_vertices = np.array(BEC_helpers.obtain_sph_polygon_vertices(traj_poly,
                                                                                               add_noise=True))  # add noise to help smooth calculation
            if len(constraints_copy) > 1:
                # taking the mean of vertices only works if they're not all lying on a shared plane (else there is
                # ambiguity)
                inside_traj_sph_polygon = cg.compute_average_point(traj_spherical_polygon_vertices)
            else:
                # if there is only a single constraint/plane (i.e. a half sphere), you can simply take the plane normal
                # as the inside point (with some noise to prevent out of domain issues with trig function)
                inside_traj_sph_polygon = constraints_copy[0][0] + np.random.sample(3) * 0.001

            traj_spherical_polygon_vertices = cg.sort_points_by_angle(traj_spherical_polygon_vertices,
                                                                      inside_traj_sph_polygon)
            traj_sph_polygon = sph_polygon.SphericalPolygon(traj_spherical_polygon_vertices,
                                                        inside=tuple(inside_traj_sph_polygon))

            # if abs(BEC_helpers.calc_solid_angles([constraints_copy])[0] - traj_sph_polygon.area()) > 1  or np.isnan(traj_sph_polygon.area()):
            #     print('env: {}, traj: {}'.format(env_idx, traj_idx))
            #     raise AssertionError("Too much deviation in trajectory BEC areas")

            if traj_sph_polygon.intersects_poly(posterior_sph_polygon):
                intersected_polygon = traj_sph_polygon.intersection(posterior_sph_polygon)
                intersected_area = intersected_polygon.area()
                if intersected_area > 1.05 * min(traj_sph_polygon.area(), posterior_sph_polygon.area()):
                    # the intersection shouldn't be larger than the smaller of the two constituent areas
                    print('Env: {}, constraint idx: {}'.format(env_idx, constraint_idx))
                    raise AssertionError("bad intersection")
                BEC_areas.append(traj_sph_polygon.intersection(posterior_sph_polygon).area())
            else:
                BEC_areas.append(0)

    return BEC_areas

def obtain_summary_counterfactual(data_loc, summary_variant, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                       n_train_demos=3, prior=[], downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1):
    summary = []

    # impose prior
    min_BEC_constraints_running = prior.copy()

    # count how many nonzero constraints are present for each reward weight (i.e. variable) in the minimum BEC constraints
    # (which are obtained using one-step deviations). mask variables in order of fewest nonzero constraints for variable scaffolding
    # rationale: the variable with the most nonzero constraints, often the step cost, serves as a good reference/ratio variable
    min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
    min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_flattened for item in sublist]
    min_subset_constraints_record_array = np.array(min_subset_constraints_record_flattened)

    if summary_variant == 'counterfactual_only':
        # no variable scaffolding
        nonzero_counter = np.array([[float('inf'), float('inf'), float('inf')]])
    else:
        # for variable scaffolding
        nonzero_counter = (min_subset_constraints_record_array != 0).astype(float)
        nonzero_counter = np.sum(nonzero_counter, axis=0)
        nonzero_counter = nonzero_counter.flatten()

    variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
    print('variable filter: {}'.format(variable_filter))

    # clear the demonstration generation log
    open('models/' + data_loc + '/demo_gen_log.txt', 'w').close()

    while len(summary) < n_train_demos:
        # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, fig_name=str(len(summary)) + '.png', just_save=True)

        # (approximately) uniformly divide up the valid BEC area along 2-sphere
        sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, n_human_models)

        if len(sample_human_models) == 0:
            print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
            return summary

        info_gains_record = []
        overlap_in_opt_and_counterfactual_traj_record = []

        print("Length of summary: {}".format(len(summary)))
        with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
            myfile.write('Length of summary: {}\n'.format(len(summary)))

        for model_idx, human_model in enumerate(sample_human_models):
            print(colored('Model #: {}'.format(model_idx), 'red'))
            print(colored('Model val: {}'.format(human_model), 'red'))

            with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
                myfile.write('Model #: {}\n'.format(model_idx))
                myfile.write('Model val: {}\n'.format(human_model))

            # based on the human's current model, obtain the information gain generated when comparing to the agent's
            # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
            # are saved for reference later)
            print("Obtaining counterfactual information gains:")

            cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + str(len(summary)) + '/model' + str(model_idx)
            os.makedirs(cf_data_dir, exist_ok=True)
            if consider_human_models_jointly:
                pool.restart()
                args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], min_BEC_constraints_running, step_cost_flag, len(summary), variable_filter, consider_human_models_jointly) for i in range(len(traj_record))]
                info_gain_envs = list(tqdm(pool.imap(compute_counterfactuals, args), total=len(args)))
                pool.close()
                pool.join()
                pool.terminate()

                info_gains_record.append(info_gain_envs)
            else:
                pool.restart()
                args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], min_BEC_constraints_running, step_cost_flag, len(summary), variable_filter, consider_human_models_jointly) for i in range(len(traj_record))]
                info_gain_envs, overlap_in_opt_and_counterfactual_traj_env = zip(*pool.imap(compute_counterfactuals, tqdm(args), total=len(args)))
                pool.close()
                pool.join()
                pool.terminate()

                info_gains_record.append(info_gain_envs)
                overlap_in_opt_and_counterfactual_traj_record.append(overlap_in_opt_and_counterfactual_traj_env)

        with open('models/' + data_loc + '/info_gains_' + str(len(summary)) + '.pickle', 'wb') as f:
            pickle.dump(info_gains_record, f)


        # do a quick check of whether there's any information to be gained from any of the trajectories
        no_info_flag = False
        max_info_gain = 1
        if consistent_state_count:
            info_gains = np.array(info_gains_record)
            if np.sum(info_gains > 1) == 0:
                no_info_flag = True
        else:
            info_gains_flattened_across_models = list(itertools.chain.from_iterable(info_gains_record))
            info_gains_flattened_across_envs = list(itertools.chain.from_iterable(info_gains_flattened_across_models))
            if sum(np.array(info_gains_flattened_across_envs) > 1) == 0:
                no_info_flag = True

        # no need to continue search for demonstrations if none of them will improve the human's understanding
        if no_info_flag:
            # sample up two more set of human models to try and find a demonstration that will improve the human's
            # understanding before concluding that there is no more information to be conveyed
            if variable_filter is None:
                break
            else:
                # no more informative demonstrations with this variable filter, so update it
                variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
                print(colored('Did not find any informative demonstrations.', 'red'))
                print('variable filter: {}'.format(variable_filter))

                continue

        if consider_human_models_jointly:
            print("Combining the most limiting constraints across human models:")
            pool.restart()
            args = [(i, len(sample_human_models), data_loc, len(summary), weights, step_cost_flag, variable_filter,
                     traj_record[i], min_BEC_constraints_running) for
                    i in range(len(traj_record))]
            info_gains_record, min_env_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
                *pool.imap(combine_limiting_constraints_IG, tqdm(args)))
            pool.close()
            pool.join()
            pool.terminate()

            # the possibility that no demonstration provides information gain must be checked for again,
            # in case all limiting constraints involve a masked variable and shouldn't be considered for demonstration yet
            if consistent_state_count:
                info_gains = np.array(info_gains_record)
                traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_avg)

                # obj_function = info_gains * (traj_overlap_pcts + c)  # objective 2: scaled
                obj_function = info_gains

                # not considering demos where there is no info gain helps ensure that the final demonstration
                # provides the maximum info gain (in conjuction with previously shown demonstrations)
                obj_function[info_gains == 1] = 0

                max_info_gain = np.max(info_gains)
                if max_info_gain == 1:
                    no_info_flag = True
                else:
                    # if visuals aren't considered, then you can simply return one of the demos that maximizes the obj function
                    # best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)

                    if obj_func_proportion == 1:
                        # a) select the trajectory with the maximal information gain
                        best_env_idxs, best_traj_idxs = np.where(obj_function == max(obj_function.flatten()))
                    else:
                        # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations0
                        obj_function_flat = obj_function.flatten()
                        obj_function_flat.sort()

                        best_obj = obj_function_flat[-1]
                        target_obj = obj_func_proportion * best_obj
                        target_idx = np.argmin(abs(obj_function_flat - target_obj))
                        closest_obj = obj_function_flat[target_idx]
                        best_env_idxs, best_traj_idxs = np.where(obj_function == obj_function_flat[closest_obj])

                    best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)
            else:
                best_obj = float('-inf')
                best_env_idxs = []
                best_traj_idxs = []

                if obj_func_proportion == 1:
                    # a) select the trajectory with the maximal information gain
                    for env_idx, info_gains_per_env in enumerate(info_gains_record):
                        for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
                            if info_gain_per_traj > 1:

                                # obj = info_gain_per_traj * (
                                #             overlap_in_opt_and_counterfactual_traj_avg[env_idx][traj_idx] + c)  # objective 2: scaled
                                obj = info_gain_per_traj

                                if np.isclose(obj, best_obj):
                                    best_env_idxs.append(env_idx)
                                    best_traj_idxs.append(traj_idx)
                                elif obj > best_obj:
                                    best_obj = obj

                                    best_env_idxs = [env_idx]
                                    best_traj_idxs = [traj_idx]
                                if info_gain_per_traj > max_info_gain:
                                    max_info_gain = info_gain_per_traj
                                    print("new max info: {}".format(max_info_gain))
                else:
                    # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations)
                    # first find the max information gain
                    for env_idx, info_gains_per_env in enumerate(info_gains_record):
                        for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
                            if info_gain_per_traj > 1:
                                obj = info_gain_per_traj

                                if np.isclose(obj, best_obj):
                                    pass
                                elif obj > best_obj:
                                    best_obj = obj

                                if info_gain_per_traj > max_info_gain:
                                    max_info_gain = info_gain_per_traj
                                    print("new max info: {}".format(max_info_gain))

                    target_obj = obj_func_proportion * best_obj
                    closest_obj_dist = float('inf')

                    for env_idx, info_gains_per_env in enumerate(info_gains_record):
                        for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
                            if info_gain_per_traj > 1:

                                obj = info_gain_per_traj

                                if np.isclose(abs(target_obj - obj), closest_obj_dist):
                                    best_env_idxs.append(env_idx)
                                    best_traj_idxs.append(traj_idx)
                                elif abs(target_obj - obj) < closest_obj_dist:
                                    closest_obj_dist = abs(obj - target_obj)

                                    best_env_idxs = [env_idx]
                                    best_traj_idxs = [traj_idx]

                if max_info_gain == 1:
                    no_info_flag = True
                else:
                    best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)

            print("current max info: {}".format(max_info_gain))
            if no_info_flag:
                if variable_filter is None:
                    break
                else:
                    # no more informative demonstrations with this variable filter, so update it
                    variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
                    print(colored('Did not find any informative demonstrations.', 'red'))
                    print('variable filter: {}'.format(variable_filter))
                    continue

            best_traj = traj_record[best_env_idx][best_traj_idx]

            filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
            with open(filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)
            best_mdp = wt_vi_traj_env[0][1].mdp
            best_mdp.set_init_state(best_traj[0][0]) # for completeness
            min_BEC_constraints_running.extend(min_env_constraints_record[best_env_idx][best_traj_idx])
            min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
            summary.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx],
                            sample_human_models])
        else:
            # b) consider each human model separately
            if consistent_state_count:
                info_gains = np.array(info_gains_record)
                traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_record)

                # obj_function = info_gains * (traj_overlap_pcts + c)                  # objective 2: scaled
                obj_function = info_gains

                obj_function[info_gains == 1] = 0

                select_model, best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)

                if max(info_gains.flatten()) == 1:
                    no_info_flag = True
                else:
                    best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)
                    best_env_idxs, best_traj_idxs = np.where(obj_function == max(obj_function.flatten()))
                    max_info_gain = np.max(info_gains)

                    best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)
            else:
                best_obj = float('-inf')
                select_model = best_env_idx = best_traj_idx = 0
                for model_idx, info_gains_per_model in enumerate(info_gains_record):
                    for env_idx, info_gains_per_env in enumerate(info_gains_per_model):
                        for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
                            if info_gain_per_traj > 1:
                                # obj = info_gain_per_traj * (overlap_in_opt_and_counterfactual_traj_record[model_idx][env_idx][traj_idx] + c)
                                obj = info_gain_per_traj

                                if np.isclose(obj, best_obj):
                                    best_env_idxs.append(env_idx)
                                    best_traj_idxs.append(traj_idx)
                                elif obj > best_obj:
                                    best_obj = obj
                                    max_info_gain = info_gain_per_traj

                                    best_env_idxs = [env_idx]
                                    best_traj_idxs = [traj_idx]

                                if info_gain_per_traj > max_info_gain:
                                    max_info_gain = info_gain_per_traj

                if max_info_gain == 1:
                    no_info_flag = True
                else:
                    best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)

            if no_info_flag:
                if variable_filter is None:
                    break
                else:
                    # no more informative demonstrations with this variable filter, so update it
                    variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
                    print(colored('Did not find any informative demonstrations.', 'red'))
                    print('variable filter: {}'.format(variable_filter))
                    continue

            with open('models/' + data_loc + '/counterfactual_data_' + str(len(summary)) + '/model' + str(
                    select_model) + '/cf_data_env' + str(
                    best_env_idx).zfill(5) + '.pickle', 'rb') as f:
                best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)

            # best_human_trajs = best_human_trajs_record_env[best_traj_idx]
            best_traj = traj_record[best_env_idx][best_traj_idx]

            filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
            with open(filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)
            best_mdp = wt_vi_traj_env[0][1].mdp
            best_mdp.set_init_state(best_traj[0][0]) # for completeness
            min_BEC_constraints_running.extend(constraints_env[best_traj_idx])
            min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
            summary.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), constraints_env[best_traj_idx],
                            sample_human_models, select_model])

        print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
        with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
            myfile.write('Max infogain: {}\n'.format(max_info_gain))
            myfile.write('\n')

        # this method doesn't always finish, so save the summary along the way
        with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
            pickle.dump(summary, f)

    return summary

def obtain_preliminary_tests(data_loc, min_BEC_constraints, min_subset_constraints_record, traj_record, downsample_threshold=float("inf")):
    preliminary_test_info = []

    # if you're looking for demonstrations that will convey the most constraining BEC region or will be employing scaffolding,
    # obtain the demos needed to convey the most constraining BEC region
    BEC_constraints = min_BEC_constraints
    BEC_constraint_bookkeeping = BEC_helpers.perform_BEC_constraint_bookkeeping(BEC_constraints,
                                                                                min_subset_constraints_record)

    # todo: remove any environments or trajectories conveyed by an earlier summary

    # downsampling strategy 1: randomly cull sets with too many members for computational feasibility
    # for j, set in enumerate(sets):
    #     if len(set) > downsample_threshold:
    #         sets[j] = random.sample(set, downsample_threshold)

    # downsampling strategy 2: if there are any env_traj pairs that cover more than one constraint, use it and remove all
    # env_traj pairs that would've conveyed the same constraints
    # initialize all env_traj tuples with covering the first min BEC constraint
    env_constraint_mapping = {}
    for key in BEC_constraint_bookkeeping[0]:
        env_constraint_mapping[key] = [0]
    max_constraint_count = 1
    max_env_traj_tuples = [key]

    # for all other env_traj tuples,
    for constraint_idx, env_traj_tuples in enumerate(BEC_constraint_bookkeeping[1:]):
        for env_traj_tuple in env_traj_tuples:
            # if this env_traj tuple has already been seen previously
            if env_traj_tuple in env_constraint_mapping.keys():
                env_constraint_mapping[env_traj_tuple].append(constraint_idx)
                # and adding another constraint to this tuple increases the highest constraint coverage by a single tuple
                if len(env_constraint_mapping[env_traj_tuple]) > max_constraint_count:
                    # update the max values, replacing the max tuple
                    max_constraint_count = len(env_constraint_mapping[env_traj_tuple])
                    max_env_traj_tuples = [env_traj_tuple]
                # otherwise, if it simply equals the highest constraint coverage, add this tuple to the contending list
                elif len(env_constraint_mapping[env_traj_tuple]) == max_constraint_count:
                    max_env_traj_tuples.append(env_traj_tuple)
            else:
                env_constraint_mapping[env_traj_tuple] = [constraint_idx]

    env_complexity_mapping = {} # helps prevent reopening the pickle file of the same environment multiple times. could parallelize this
    if max_constraint_count == 1:
        # no one demo covers multiple constraints. so greedily select demos from base list that is mot visually complex
        # filter for the most visually complex environment
        for constraint_idx, env_traj_tuples in enumerate(BEC_constraint_bookkeeping):
            old_env = -1
            max_complexity = -1
            max_complexity_env_traj_tuple = -1

            for env_traj_tuple in env_traj_tuples:
                env, traj = env_traj_tuple
                # assuming that the tuples are stored in order of environment indices
                if env != old_env:
                    if env not in env_complexity_mapping.keys():
                        filename = mp_helpers.lookup_env_filename(data_loc, env)
                        with open(filename, 'rb') as f:
                            wt_vi_traj_env = pickle.load(f)

                        complexity = wt_vi_traj_env[0][1].mdp.measure_env_complexity()
                        env_complexity_mapping[env] = complexity
                    else:
                        complexity = env_complexity_mapping[env]
                    if complexity > max_complexity:
                        max_complexity = complexity
                        max_complexity_env_traj_tuple = env_traj_tuple
                else:
                    continue
            preliminary_test_info.append((max_complexity_env_traj_tuple, BEC_constraints[constraint_idx]))
    else:
        # # filter for the most visually complex environment
        # old_env = -1
        # max_complexity = -1
        # max_complexity_env_traj_tuple = -1
        # for env_traj_tuple in max_env_traj_tuples:
        #     env, traj = env_traj_tuple
        #     if env != old_env:
        #         filename = mp_helpers.lookup_env_filename(data_loc, env)
        #         with open(filename, 'rb') as f:
        #             wt_vi_traj_env = pickle.load(f)
        #
        #         complexity = wt_vi_traj_env[0][1].mdp.measure_env_complexity()
        #         if complexity > max_complexity:
        #             max_complexity = complexity
        #             max_complexity_env_traj_tuple = env_traj_tuple
        #     else:
        #         continue

        raise Exception('Down-select from env_traj that covers the most constraints (not yet implemented).')

    preliminary_tests = []
    for info in preliminary_test_info:
        env_idx, traj_idx = info[0]
        traj = traj_record[env_idx][traj_idx]
        constraints = info[1]
        filename = mp_helpers.lookup_env_filename(data_loc, env_idx)
        with open(filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)
        best_mdp = wt_vi_traj_env[0][1].mdp
        best_mdp.set_init_state(traj[0][0])  # for completeness
        preliminary_tests.append([best_mdp, traj, (env_idx, traj_idx), constraints])

    # todo: keep track of env and traj shown during training and testing and standardize how training and testing demos are stored

    return preliminary_tests

def obtain_SCOT_summaries(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag,downsample_threshold=float("inf")):
    min_BEC_summary = []

    # if you're looking for demonstrations that will convey the most constraining BEC region or will be employing scaffolding,
    # obtain the demos needed to convey the most constraining BEC region
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

    # obtain one combination of demos+environments pairs that cover all BEC constraints
    filtered_combo = []
    for combination in itertools.product(*sets):
        filtered_combo.append(np.unique(combination))
        break

    best_idxs = filtered_combo[0]

    for best_idx in best_idxs:
        best_env_idx = env_record[best_idx]

        # record information associated with the best selected summary demo
        best_traj = traj_record[best_idx]
        filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
        with open(filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)
        best_mdp = wt_vi_traj_env[0][1].mdp
        best_mdp.set_init_state(best_traj[0][0])  # for completeness
        constraints_added = min_subset_constraints_record[best_idx]
        min_BEC_summary.append([best_mdp, best_traj, constraints_added])

    for best_idx in sorted(best_idxs, reverse=True):
        del min_subset_constraints_record[best_idx]
        del traj_record[best_idx]
        del env_record[best_idx]
        del BEC_lengths_record[best_idx]

    return min_BEC_summary

def obtain_summary(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=3, downsample_threshold=float("inf")):
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

    env_traj_tracer = []
    for j, env_idx in enumerate(env_record):
        env_traj_tracer.extend(list(zip([env_idx] * len(traj_record[j]), list(np.arange(len(traj_record[env_idx]))))))

    # the code below was made with flattened lists in mind, so flatten accordingly (copy first to only edit the originals)
    BEC_lengths_record = copy.deepcopy(BEC_lengths_record)
    min_subset_constraints_record = copy.deepcopy(min_subset_constraints_record)
    traj_record = copy.deepcopy(traj_record)
    BEC_lengths_record = [item for sublist in BEC_lengths_record for item in sublist]
    min_subset_constraints_record = [item for sublist in min_subset_constraints_record for item in sublist]
    traj_record = [item for sublist in traj_record for item in sublist]

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

    # obtain SCOT demos
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
    # visualize_constraints(min_BEC_constraints, weights, step_cost_flag)

    wt_vi_traj_dict = {}
    for j, combo in enumerate(filtered_combo):
        if j % 100000 == 0:
            print("{}/{}".format(j, len(filtered_combo)))

        complexity = 0
        for key in combo:
            env_idx = env_traj_tracer[key][0]
            if env_idx in wt_vi_traj_dict.keys():
                wt_vi_traj_env = wt_vi_traj_dict[env_idx]
            else:
                filename = mp_helpers.lookup_env_filename(data_loc, env_idx)
                with open(filename, 'rb') as f:
                    wt_vi_traj_env = pickle.load(f)
                wt_vi_traj_dict[env_idx] = wt_vi_traj_env

            # if summary_variant[1] == 'high':
                # get demos of low visual complexity
            complexity += wt_vi_traj_env[0][1].mdp.measure_env_complexity()
                # get simple demos
                # BEC_length += \
                # BEC_helpers.calculate_BEC_length(min_subset_constraints_record[env], weights, step_cost_flag)[0]
            # else:
            #     # get demos of high visual complexity
            #     complexity += -wt_vi_traj_env[0][1].mdp.measure_env_complexity()
            #     # get complex demos
            #     # BEC_length += \
            #     # -BEC_helpers.calculate_BEC_length(min_subset_constraints_record[env], weights, step_cost_flag)[0]

        complexities[j] = complexity / len(combo)

        visual_dissimilarity = 0
        if len(combo) >= 2:
            # compare every possible pairing of this set of demos
            pairs = list(itertools.combinations(combo, 2))
            for pair in pairs:
                # get similar demos
                visual_dissimilarity += wt_vi_traj_dict[env_traj_tracer[pair[0]][0]][0][1].mdp.measure_visual_dissimilarity(traj_record[pair[0]][0][0],
                                                                                wt_vi_traj_dict[env_traj_tracer[pair[1]][0]][0][1].mdp,
                                                                                traj_record[pair[1]][0][0])
                # else:
                #     # get dissimilar demos
                #     visual_dissimilarity += -wt_vi_traj_candidates[
                #         env_record[pair[0]]][0][1].mdp.measure_visual_dissimilarity(traj_record[pair[0]][0][0],
                #                                                                     wt_vi_traj_candidates[
                #                                                                         env_record[pair[1]]][0][1].mdp,
                #                                                                     traj_record[pair[1]][0][0])
            visual_dissimilarities[j] = visual_dissimilarity / len(pairs)


    tie_breaker = np.arange(len(filtered_combo))
    sorted_zipped = sorted(zip(visual_dissimilarities, complexities, tie_breaker, filtered_combo))
    visual_dissimilarities_sorted, complexities_sorted, _, filtered_combo_sorted = list(
        zip(*sorted_zipped))

    best_idxs = filtered_combo_sorted[0]

    for best_idx in best_idxs:
        best_env_idx = env_traj_tracer[best_idx][0]

        # record information associated with the best selected summary demo
        best_traj = traj_record[best_idx]
        best_mdp = wt_vi_traj_dict[best_env_idx][0][1].mdp
        best_mdp.set_init_state(best_traj[0][0])  # for completeness
        constraints_added = min_subset_constraints_record[best_idx]
        min_BEC_summary.append([best_mdp, best_traj, env_traj_tracer[best_idx], constraints_added])

    for best_idx in sorted(best_idxs, reverse=True):
        del min_subset_constraints_record[best_idx]
        del traj_record[best_idx]
        del BEC_lengths_record[best_idx]
        del env_traj_tracer[best_idx]

    print('# SCOT demos: {}'.format(len(min_BEC_summary)))

    # fill out the rest of the vacant demo slots
    included_demo_idxs = []
    ongoing_summary_constraints = []

    if summary_variant == 'feature_only':
        # determine how many demonstrations should be allotted to each masked variable
        # this needs to be updated if more than one variable can be masked at a time (e.g. when there are more than 3 features)
        demo_chunk_per_variable_scaffolding = int(np.ceil((n_train_demos - len(min_BEC_summary)) / (weights.shape[1] + 1)))
        n_demos_per_variable_scaffolding = []
        # +1 is to account for no variable scaffolding
        for weight in range(weights.shape[1] + 1):
            n_demos_per_variable_scaffolding.append(min(demo_chunk_per_variable_scaffolding, n_train_demos - len(min_BEC_summary) - sum(n_demos_per_variable_scaffolding)))
        n_demos_per_variable_scaffolding.reverse() # so that you can easily pop off the next item

        print('n_demos for variable scaffolding: {}'.format(n_demos_per_variable_scaffolding))

        # count how many nonzero constraints are present for each reward weight (i.e. variable) in the minimum BEC constraints
        # (which are obtained using one-step deviations). mask variables in order of fewest nonzero constraints for variable scaffolding
        # rationale: the variable with the most nonzero constraints, often the step cost, serves as a good reference/ratio variable
        min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
        min_subset_constraints_record_array = np.array(min_subset_constraints_record_flattened)
        # for variable scaffolding
        nonzero_counter = (min_subset_constraints_record_array != 0).astype(float)
        nonzero_counter = np.sum(nonzero_counter, axis=0)
        nonzero_counter = nonzero_counter.flatten()

        variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
        print('variable filter: {}'.format(variable_filter))

    if len(min_BEC_summary) < n_train_demos:
        while len(summary) + len(min_BEC_summary) < n_train_demos:
            if summary_variant == 'feature_only':
                BEC_lengths_record_filtered = []
                min_subset_constraints_record_filtered = []
                traj_record_filtered = []
                env_traj_tracer_filtered = []

                # resulting BEC area of the summary demonstrations selected to date (excluding SCOT demos)
                if len(ongoing_summary_constraints) > 0:
                    curr_summary_BEC_length = BEC_helpers.calc_solid_angles([ongoing_summary_constraints])[0]
                else:
                    curr_summary_BEC_length = 4 * np.pi

                for traj_idx, min_env_constraints in enumerate(min_subset_constraints_record):
                    skip_demo = False
                    if variable_filter is not None:
                        for constraint in min_env_constraints:
                            if abs(variable_filter.dot(constraint.T)[0, 0]) > 0:
                                # conveys information about variable designated to be filtered out, so block this demonstration
                                skip_demo = True
                                break
                    # only add if it doesn't convey information about a variable designated to be filtered out and
                    # if the demonstration decreases the ongoing BEC summary area
                    if not skip_demo and BEC_lengths_record[traj_idx] < curr_summary_BEC_length:
                        BEC_lengths_record_filtered.append(BEC_lengths_record[traj_idx])
                        min_subset_constraints_record_filtered.append(min_env_constraints)
                        traj_record_filtered.append(traj_record[traj_idx])
                        env_traj_tracer_filtered.append(env_traj_tracer[traj_idx])

                n_demos_sought = n_demos_per_variable_scaffolding.pop()

                if len(BEC_lengths_record_filtered) > 0:
                    # take care not to ask for more demos / clusters than there are demos in BEC_lengths_record_filters
                    covering_demos_idxs = obtain_potential_summary_demos(BEC_lengths_record_filtered, min(n_demos_sought, len(BEC_lengths_record_filtered)), min((n_demos_sought * 2) - 1, (len(BEC_lengths_record_filtered) * 2) - 1))
                else:
                    covering_demos_idxs = None
            else:
                BEC_lengths_record_filtered = BEC_lengths_record
                min_subset_constraints_record_filtered = min_subset_constraints_record
                traj_record_filtered = traj_record
                env_traj_tracer_filtered = env_traj_tracer

                n_demos_sought = None
                covering_demos_idxs = obtain_potential_summary_demos(BEC_lengths_record_filtered, n_train_demos - len(min_BEC_summary), ((n_train_demos - len(min_BEC_summary)) * 2) - 1)

            if covering_demos_idxs is not None:
                for covering_demo_idxs in covering_demos_idxs:
                    #remove demos that have already been included
                    duplicate_idxs = []
                    for covering_idx, covering_demo_idx in enumerate(covering_demo_idxs):
                        for included_demo_idx in included_demo_idxs:
                            if included_demo_idx == env_traj_tracer_filtered[covering_demo_idx]:
                                duplicate_idxs.append(covering_idx)

                    for duplicate_idx in sorted(duplicate_idxs, reverse=True):
                        del covering_demo_idxs[duplicate_idx]

                    visual_dissimilarities = np.zeros(len(covering_demo_idxs))
                    complexities = np.zeros(len(covering_demo_idxs))

                    for j, covering_demo_idx in enumerate(covering_demo_idxs):

                        env_idx = env_traj_tracer_filtered[covering_demo_idx][0]
                        if env_idx in wt_vi_traj_dict.keys():
                            wt_vi_traj_env = wt_vi_traj_dict[env_idx]
                        else:
                            filename = mp_helpers.lookup_env_filename(data_loc, env_idx)
                            with open(filename, 'rb') as f:
                                wt_vi_traj_env = pickle.load(f)
                            wt_vi_traj_dict[env_idx] = wt_vi_traj_env

                        # only compare the visual dissimilarity to the most recent summary
                        if len(summary) > 0:
                            # if summary_variant[0] == 'forward' or summary_variant[0] == 'backward':
                            #     if summary_variant[1] == 'high':
                            # get similar demos
                            visual_dissimilarities[j] = wt_vi_traj_env[0][1].mdp.measure_visual_dissimilarity(
                                traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])
                            #     else:
                            #         # get dissimilar demos
                            #         visual_dissimilarities[j] = -wt_vi_traj_candidates[env_record[covering_demo_idx]][0][
                            #             1].mdp.measure_visual_dissimilarity(
                            #             traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])
                            # else:
                            #     if summary_variant[1] == 'high':
                            #         # get dissimilar demos for diversity
                            #         visual_dissimilarities[j] = -wt_vi_traj_candidates[env_record[covering_demo_idx]][0][
                            #             1].mdp.measure_visual_dissimilarity(
                            #             traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])
                            #     else:
                            #         # get similar demos for redundancy
                            #         visual_dissimilarities[j] = wt_vi_traj_candidates[env_record[covering_demo_idx]][0][
                            #             1].mdp.measure_visual_dissimilarity(
                            #             traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])

                        # if summary_variant[1] == 'high':
                        # get demos of low visual complexity
                        complexities[j] = wt_vi_traj_env[0][1].mdp.measure_env_complexity()
                        # else:
                        #     # get demos of high visual complexity
                        #     complexities[j] = -wt_vi_traj_candidates[env_record[covering_demo_idx]][0][1].mdp.measure_env_complexity()

                    tie_breaker = np.arange(len(covering_demo_idxs))
                    # sorts from small to large values
                    sorted_zipped = sorted(zip(visual_dissimilarities, complexities, tie_breaker, covering_demo_idxs))
                    visual_dissimilarities_sorted, complexities_sorted, _, covering_demo_idxs_sorted = list(
                        zip(*sorted_zipped))

                    best_idx = covering_demo_idxs_sorted[0]
                    best_env_idx = env_traj_tracer_filtered[best_idx][0]

                    # record information associated with the best selected summary demo
                    best_traj = traj_record_filtered[best_idx]
                    best_mdp = wt_vi_traj_dict[best_env_idx][0][1].mdp
                    constraints_added = min_subset_constraints_record_filtered[best_idx]
                    summary.append([best_mdp, best_traj, env_traj_tracer_filtered[best_idx], constraints_added])
                    print('Constrained added: {}'.format(constraints_added))
                    ongoing_summary_constraints.extend(constraints_added)

                    included_demo_idxs.append(env_traj_tracer_filtered[best_idx])

                    if n_demos_sought is not None:
                        n_demos_sought -= 1

                    if len(min_BEC_summary) + len(summary) >= n_train_demos:
                        summary.extend(min_BEC_summary)
                        # flip the order of the summary from highest information / hardest to lowest information / easiest
                        if summary_variant[0] == 'backward':
                            summary.reverse()
                        return summary

            if len(ongoing_summary_constraints) > 1:
                ongoing_summary_constraints = BEC_helpers.remove_redundant_constraints(ongoing_summary_constraints, weights, step_cost_flag)

            # rollover any unused demonstrations
            if n_demos_sought > 0:
                n_demos_per_variable_scaffolding[-1] += n_demos_sought

            print('updated n_demos for variable scaffolding: {}'.format(n_demos_per_variable_scaffolding))

            variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
            print('variable filter: {}'.format(variable_filter))

        summary.extend(min_BEC_summary)
        print(len(min_BEC_summary))
        # flip the order of the summary from highest information / hardest to lowest information / easiest
        if summary_variant[0] == 'backward':
            summary.reverse()
    else:
        # if you've obtained more demonstrations than were requested, return a subset of the most informative ones
        summary = min_BEC_summary[-n_train_demos:]
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
    for summary_idx, BEC_summary in enumerate(BEC_summaries_collection):
        print("Showing demo {} out of {}".format(summary_idx + 1, len(BEC_summaries_collection)))

        # visualize demonstration
        BEC_summary[0].visualize_trajectory(BEC_summary[1])


def visualize_test_envs(posterior, test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers, weights, step_cost_flag):
    for j, test_wt_vi_traj_tuple in enumerate(test_wt_vi_traj_tuples):
        print('Visualizing test environment {} with BEC length of {}'.format(j, test_BEC_lengths[j]))

        print(selected_env_traj_tracers[j])
        print(test_BEC_constraints[j])
        vi_candidate = test_wt_vi_traj_tuple[1]
        trajectory_candidate = test_wt_vi_traj_tuple[2]
        vi_candidate.mdp.visualize_trajectory(trajectory_candidate)

        # visualize_constraints(test_BEC_constraints[j], weights, step_cost_flag)
        #
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        #
        # ieqs_posterior = BEC_helpers.constraints_to_halfspace_matrix_sage(posterior)
        # poly_posterior = Polyhedron.Polyhedron(ieqs=ieqs_posterior)  # automatically finds the minimal H-representation
        # BEC_viz.visualize_spherical_polygon(poly_posterior, fig=fig, ax=ax, plot_ref_sphere=False, color='g')
        #
        # ieqs_test = BEC_helpers.constraints_to_halfspace_matrix_sage(test_BEC_constraints[j])
        # poly_test = Polyhedron.Polyhedron(ieqs=ieqs_test)
        # BEC_viz.visualize_spherical_polygon(poly_test, fig=fig, ax=ax, plot_ref_sphere=False)
        #
        # ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        #
        # plt.show()