import matplotlib.pyplot as plt
from simple_rl.utils import mdp_helpers
from simple_rl.agents import FixedPolicyAgent
import policy_summarization.BEC_helpers as BEC_helpers
import numpy as np
import itertools
import random

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
    min_subset_constraints_record = []
    env_record = []
    constraints_record = []
    traj_record = []
    processed_envs = []

    # go through each environment and corresponding optimal trajectory, and extract the behavior equivalence class (BEC) constraints
    for env_idx, wt_vi_traj_candidate in enumerate(wt_vi_traj_candidates):
        if print_flag:
            print("Extracting constraints from environment {}".format(env_idx))
        mdp = wt_vi_traj_candidate[0][1].mdp

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
                    traj_hyp = mdp_helpers.rollout_policy(mdp, FixedPolicyAgent(wt_vi_traj_candidate[0][1].policy), cur_state, action_seq)
                    mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)

                    constraints.append(mu_sa - mu_sb)

            # store the BEC constraints for each environment, along with the associated demo and environment number
            min_subset_constraints = BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag)
            min_subset_constraints_record.append(min_subset_constraints)
            constraints_record.extend(min_subset_constraints)
            traj_record.append(traj_opt)
            env_record.append(env_idx)
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
                    min_subset_constraints = BEC_helpers.clean_up_constraints(constraints, weights, step_cost_flag)
                    # store the BEC constraints for each environment, along with the associated demo and environment number
                    min_subset_constraints_record.append(min_subset_constraints)
                    constraints_record.extend(min_subset_constraints)
                    traj_record.append(traj_opt)
                    env_record.append(env_idx)

                processed_envs.append(mdp.env_code)

    return min_subset_constraints_record, env_record, traj_record

def extract_BEC_constraints(min_subset_constraints_record, weights, step_cost_flag):
    '''
    Summary: Obtain the minimum BEC constraints across all environments
    '''
    constraints_record = [item for sublist in min_subset_constraints_record for item in sublist]

    # first obtain the absolute min BEC constraint
    min_BEC_constraints = BEC_helpers.clean_up_constraints(constraints_record, weights, step_cost_flag)

    # then determine the BEC lengths of all other potential demos that could be shown
    BEC_lengths = np.zeros(len(min_subset_constraints_record))
    for j, min_subset_constraints in enumerate(min_subset_constraints_record):
        BEC_lengths[j] = BEC_helpers.calculate_BEC_length(min_subset_constraints, weights, step_cost_flag)[0]

    # sort and bin the BEC lengths of each possible environment + demonstration
    unique_BEC_lengths, unique_BEC_bins = np.unique(np.round(BEC_lengths, 5), return_inverse=True)
    unique_BEC_bins = list(unique_BEC_bins)

    # ordered from most constraining to least constraining
    return min_BEC_constraints, unique_BEC_lengths, unique_BEC_bins


def obtain_summary(wt_vi_traj_candidates, min_BEC_constraints, unique_BEC_lengths, unique_BEC_bins, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=4, downsample_threshold=float("inf")):
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
    summary_constraints = []
    counter = 0

    # first obtain the demos to convey the min BEC constraints, then a set of easy BEC constraints, then
    # fill up the remaining demo slots with increasingly difficult demos in a receding-horizon fashion
    while len(min_BEC_summary) + len(summary) < n_train_demos:
        # first obtain the demos needed to convey the minimum BEC constraints
        if counter == 0:
            BEC_constraints = min_BEC_constraints

            BEC_constraint_bookkeeping = BEC_helpers.perform_BEC_constraint_bookkeeping(BEC_constraints,
                                                                                        min_subset_constraints_record)

            # extract sets of demos+environments pairs that can cover each BEC constraint
            sets = []
            for constraint_idx in range(BEC_constraint_bookkeeping.shape[1]):
                sets.append(np.argwhere(BEC_constraint_bookkeeping[:, constraint_idx] == 1).flatten().tolist())

            # downsample some sets with too many members for computational feasibility
            for j, set in enumerate(sets):
                print(len(set))
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
                        visual_dissimilarity += wt_vi_traj_candidates[
                            env_record[pair[0]]][0][1].mdp.measure_visual_dissimilarity(traj_record[pair[0]][0][0],
                                                                                        wt_vi_traj_candidates[
                                                                                            env_record[pair[1]]][0][1].mdp,
                                                                                        traj_record[pair[1]][0][0])

                    visual_dissimilarities[j] = visual_dissimilarity / len(pairs)
                else:
                    visual_dissimilarities[j] = 0

                complexity = 0
                BEC_length = 0
                for env in combo:
                    complexity += wt_vi_traj_candidates[env_record[env]][0][1].mdp.measure_env_complexity()

                    BEC_length += BEC_helpers.calculate_BEC_length(min_subset_constraints_record[env], weights, step_cost_flag)[0]

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
                best_mdp = wt_vi_traj_candidates[best_env_idx][0][1].mdp
                constraints_added = min_subset_constraints_record[best_idx]
                if counter == 0:
                    min_BEC_summary.append([best_mdp, best_traj, constraints_added])
                else:
                    summary_constraints.extend(min_subset_constraints_record[best_idx])
                    summary.append([best_mdp, best_traj, constraints_added])

                if len(min_BEC_summary) + len(summary) == n_train_demos:
                    summary.extend(min_BEC_summary)
                    return summary

            for best_idx in sorted(best_idxs, reverse=True):
                del min_subset_constraints_record[best_idx]
                del traj_record[best_idx]
                del env_record[best_idx]
                del unique_BEC_bins[best_idx]

            current_BEC_length_idx = len(unique_BEC_lengths) - 1

        # for subsequent regular demos, greedily select the best demo for each new BEC constraint
        else:
            # divide up the space between the most recent constraint set and the min BEC constraint set (idx = 0) and take the furthest one out
            current_BEC_length_idx = np.linspace(0, current_BEC_length_idx, n_train_demos - len(min_BEC_summary) + 1, endpoint=False)[1:].astype(int)[-1]
            print(current_BEC_length_idx)

            covering_demo_idxs = np.argwhere(unique_BEC_bins == current_BEC_length_idx).flatten()

            visual_dissimilarities = np.zeros(len(covering_demo_idxs))
            complexities = np.zeros(len(covering_demo_idxs))
            BEC_lengths = np.zeros(len(covering_demo_idxs))
            for j, covering_demo_idx in enumerate(covering_demo_idxs):

                # only compare the visual dissimilarity to the most recent summary
                if len(summary) > 0:
                    visual_dissimilarities[j] = wt_vi_traj_candidates[env_record[covering_demo_idx]][0][1].mdp.measure_visual_dissimilarity(
                        traj_record[covering_demo_idx][0][0], summary[-1][0], summary[-1][1][0][0])
                else:
                    visual_dissimilarities[j] = 0

                complexities[j] = wt_vi_traj_candidates[env_record[covering_demo_idx]][0][1].mdp.measure_env_complexity()
                BEC_lengths[j] = BEC_helpers.calculate_BEC_length(min_subset_constraints_record[covering_demo_idx], weights, step_cost_flag)[0]

            tie_breaker = np.arange(len(covering_demo_idxs))
            sorted_zipped = sorted(zip(visual_dissimilarities, complexities, BEC_lengths, tie_breaker, covering_demo_idxs))
            visual_dissimilarities_sorted, complexities_sorted, BEC_lengths_sorted, _, covering_demo_idxs_sorted = list(
                zip(*sorted_zipped))

            best_idxs = [covering_demo_idxs_sorted[0]]

            for best_idx in best_idxs:
                best_env_idx = env_record[best_idx]

                # record information associated with the best selected summary demo
                best_traj = traj_record[best_idx]
                best_mdp = wt_vi_traj_candidates[best_env_idx][0][1].mdp
                constraints_added = min_subset_constraints_record[best_idx]
                summary_constraints.extend(min_subset_constraints_record[best_idx])
                summary.append([best_mdp, best_traj, constraints_added])

                if len(min_BEC_summary) + len(summary) == n_train_demos:
                    summary.extend(min_BEC_summary)
                    return summary

            for best_idx in sorted(best_idxs, reverse=True):
                del min_subset_constraints_record[best_idx]
                del traj_record[best_idx]
                del env_record[best_idx]
                del unique_BEC_bins[best_idx]

        counter += 1

    summary.extend(min_BEC_summary)
    return summary

def visualize_constraints(constraints, weights, step_cost_flag, plot_lim=[(-1, 1), (-1, 1)], scale=1.0, fig_name=None):
    '''
    Summary: Visualize the constraints
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
                if 1 >= pt:
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
        # plt.plot(np.array([-1 + abs(weights[0, -1]), 0]) * scale, np.array([0, 1 - abs(weights[0, -1])]) * scale, color='grey')
        # plt.plot(np.array([0, 1 - abs(weights[0, -1])]) * scale, np.array([1 - abs(weights[0, -1]), 0]) * scale, color='grey')
        # plt.plot(np.array([1 - abs(weights[0, -1]), 0]) * scale, np.array([0, -1 + abs(weights[0, -1])]) * scale, color='grey')
        # plt.plot(np.array([0, -1 + abs(weights[0, -1])]) * scale, np.array([-1 + abs(weights[0, -1]), 0] )* scale, color='grey')
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
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=200, transparent=True)
    plt.show()


def visualize_summary(BEC_summaries_collection, weights, step_cost_flag):
    '''
    :param BEC_summary: Nested list of [mdp, trajectory]

    Summary: visualize the BEC demonstrations
    '''
    min_BEC_constraints_running = []
    for summary_idx, BEC_summary in enumerate(BEC_summaries_collection):
        print("Showing demo {} out of {}".format(summary_idx + 1, len(BEC_summaries_collection)))
        # visualize demonstration
        BEC_summary[0].visualize_trajectory(BEC_summary[1])
        # visualize constraints enforced by demonstration above
        # print(mdp_traj_constraint[2])

        # visualize the min BEC constraints of this particular demonstration
        # visualize_constraints(BEC_summary[2], weights, step_cost_flag, fig_name=str(summary_idx) + '.png')

        # visualize the min BEC constraints extracted from all demonstrations shown thus far
        # min_BEC_constraints_running.extend(BEC_summary[2])
        # min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)[0]
        # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag)