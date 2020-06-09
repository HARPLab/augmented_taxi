import matplotlib.pyplot as plt
from simple_rl.utils import mdp_helpers
from simple_rl.agents import FixedPolicyAgent
import numpy as np
from scipy.optimize import linprog
import itertools
from pypoman import compute_polytope_vertices, intersect_line_polygon

def normalize_constraints(constraints):
    '''
    Summary: Normalize all constraints such that the L1 norm is equal to 1
    '''
    normalized_constraints = []
    zero_constraint = np.zeros(constraints[0].shape)
    for constraint in constraints:
        if not equal_constraints(constraint, zero_constraint):
            normalized_constraints.append(constraint / np.linalg.norm(constraint[0, :], ord=1))

    return normalized_constraints

def remove_duplicate_constraints(constraints):
    '''
    Summary: Remove any duplicate constraints
    '''
    nonredundant_constraints = []
    zero_constraint = np.zeros(constraints[0].shape)

    for query in constraints:
        add_it = True
        for comp in nonredundant_constraints:
            # don't keep any duplicate constraints or degenerate zero constraints
            if equal_constraints(query, comp) or equal_constraints(query, zero_constraint):
                add_it = False
                break
        if add_it:
            nonredundant_constraints.append(query)

    return nonredundant_constraints

def equal_constraints(c1, c2):
    '''
    Summary: Check for equality between two constraints c1 and c2
    '''
    if np.sum(abs(c1 - c2)) <= 1e-05:
        return True
    else:
        return False

def remove_redundant_constraints(constraints, weights, step_cost_flag):
    '''
    Summary: Remove redundant constraint that do not change the underlying BEC region
    '''
    # these lists are effectively one level deep so a shallow copy should suffice. copy over the original constraints
    # and remove redundant constraints one by one
    nonredundant_constraints = constraints.copy()

    for query_constraint in constraints:
        # create a set of constraints the excludes the current constraint in question (query_constraint)
        constraints_other = []
        for nonredundant_constraint in nonredundant_constraints:
            if not equal_constraints(query_constraint, nonredundant_constraint):
                constraints_other.append(list(-nonredundant_constraint[0]))

        # if there are other constraints left to compare to
        if len(constraints_other) > 0:
            # solve linear program
            # min_x a^Tx, st -Ax >= -b (note that scipy traditionally accepts bounds as Ax <= b, hence the negative multiplier to the constraints)
            a = np.ndarray.tolist(query_constraint[0])
            b = [0] * len(constraints_other)
            if step_cost_flag:
                # the last weight is the step cost, which is assumed to be known by the learner. adjust the bounds accordingly
                res = linprog(a, A_ub=constraints_other, b_ub=b, bounds=[(-1, 1), (-1, 1), (weights[0, -1], weights[0, -1])])
            else:
                res = linprog(a, A_ub=constraints_other, b_ub=b, bounds=[(-1, 1)] * constraints[0].shape[1])

            # if query_constraint * res.x^T >= 0, then this constraint is redundant. copy over everything except this constraint
            if query_constraint.dot(res.x.reshape(-1, 1))[0][0] >= -1e-05: # account for slight numerical instability
                copy_array = []
                for nonredundant_constraint in nonredundant_constraints:
                    if not equal_constraints(query_constraint, nonredundant_constraint):
                        copy_array.append(nonredundant_constraint)
                nonredundant_constraints = copy_array
        else:
            break

    return nonredundant_constraints

def _compute_lengths(lines):
    lengths = np.zeros(len(lines))
    n = 0
    for line in lines:
        lengths[n] = np.linalg.norm(line[1] - line[0])
        n += 1

    return lengths

def calculate_BEC_length(constraints, weights, step_cost_flag):
    '''
    :param constraints (list of constraints, corresponding to the A of the form Ax >= 0): constraints that comprise the
        BEC region
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :return: total_intersection_length: total length of the intersection between the BEC region and the L1 constraints
    '''
    if step_cost_flag:
        # convert the half space representation of a convex polygon (Ax < b) into the corresponding polytope vertices
        n_boundary_constraints = 4
        A = np.zeros((len(constraints) + n_boundary_constraints, len(constraints[0][0]) - 1))
        b = np.zeros(len(constraints) + n_boundary_constraints)

        for j in range(len(constraints)):
            A[j, :] = np.array([-constraints[j][0][0], -constraints[j][0][1]])
            b[j] = constraints[j][0][2] * weights[0, -1]

        # add the L1 constraints
        A[len(constraints), :] = np.array([1, 0])
        b[len(constraints)] = 1
        A[len(constraints) + 1, :] = np.array([-1, 0])
        b[len(constraints) + 1] = 1
        A[len(constraints) + 2, :] = np.array([0, 1])
        b[len(constraints) + 2] = 1
        A[len(constraints) + 3, :] = np.array([0, -1])
        b[len(constraints) + 3] = 1

        # compute the vertices of the convex polygon formed by the BEC constraints (BEC polygon)
        vertices = compute_polytope_vertices(A, b)

        # intersect the L1 constraints with the BEC polygon
        L1_intersections = []
        # L1 constraints in 2D
        L1_constraints = [[[-1 + abs(weights[0, -1]), 0], [0, 1 - abs(weights[0, -1])]], [[0, 1 - abs(weights[0, -1])], [1 - abs(weights[0, -1]), 0]],
                          [[1 - abs(weights[0, -1]), 0], [0, -1 + abs(weights[0, -1])]], [[0, -1 + abs(weights[0, -1])], [-1 + abs(weights[0, -1]), 0]]]

        for constraint in L1_constraints:
            intersection = intersect_line_polygon(constraint, vertices, False)
            if len(intersection) > 0:
                L1_intersections.append(intersection)

        intersection_lengths = _compute_lengths(L1_intersections)
        total_intersection_length = np.sum(intersection_lengths)
    else:
        raise Exception("Not yet implemented.")
    return total_intersection_length

def visualize_constraints(constraints, weights, step_cost_flag):
    '''
    Summary: Visualize the constraints
    '''
    # This visualization function is currently specialized to handle plotting problems with two unknown weights or two
    # unknown and one known weight. For higher dimensional constraints, this visualization function must be updated.

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    wt_shading = 1. / len(constraints)

    if step_cost_flag:
        # if the final weight is the step cost, it is assumed that there are three weights, which must be accounted for
        # differently to plot the BEC region for the first two weights
        for constraint in constraints:
            if constraint[0, 1] == 0.:
                # completely vertical line going through zero
                pt = (-weights[0, -1] * constraint[0, 2]) / constraint[0, 0]
                plt.plot([pt, pt], [-1, 1])

                # use (1, 0) as a test point to decide which half space to color
                if 1 >= pt:
                    # color the right side of the line
                    plt.axvspan(pt, 1, alpha=wt_shading, color='blue')
                else:
                    # color the left side of the line
                    plt.axvspan(-1, pt, alpha=wt_shading, color='blue')
            else:
                pt_1 = (constraint[0, 0] - (weights[0, -1] * constraint[0, 2])) / constraint[0, 1]
                pt_2 = (-constraint[0, 0] - (weights[0, -1] * constraint[0, 2])) / constraint[0, 1]
                plt.plot([-1, 1], [pt_1, pt_2])

                # use (0, 1) as a test point to decide which half space to color
                if constraint[0, 1] + (weights[0, -1] * constraint[0, 2]) >= 0:
                    plt.fill_between([-1, 1], [pt_1, pt_2], [1, 1], alpha=wt_shading, color='blue')
                else:
                    plt.fill_between([-1, 1], [pt_1, pt_2], [-1, -1], alpha=wt_shading, color='blue')

        # visualize the L1 norm == 1 constraints
        plt.plot([-1 + abs(weights[0, -1]), 0], [0, 1 - abs(weights[0, -1])], color='grey')
        plt.plot([0, 1 - abs(weights[0, -1])], [1 - abs(weights[0, -1]), 0], color='grey')
        plt.plot([1 - abs(weights[0, -1]), 0], [0, -1 + abs(weights[0, -1])], color='grey')
        plt.plot([0, -1 + abs(weights[0, -1])], [-1 + abs(weights[0, -1]), 0], color='grey')
    else:
        for constraint in constraints:
            if constraint[0, 0] == 1.:
                # completely vertical line going through zero
                plt.plot([constraint[0, 1] / constraint[0, 0], -constraint[0, 1] / constraint[0, 0]], [-1, 1])

                # use (1, 0) as a test point to decide which half space to color
                if constraint[0, 0] >= 0:
                    # color the right side of the line
                    plt.axvspan(0, 1, alpha=wt_shading, color='blue')
                else:
                    # color the left side of the line
                    plt.axvspan(-1, 0, alpha=wt_shading, color='blue')
            else:
                pt_1 = constraint[0, 0] / constraint[0, 1]
                pt_2 = -constraint[0, 0] / constraint[0, 1]
                plt.plot([-1, 1], [pt_1, pt_2])

                # use (0, 1) as a test point to decide which half space to color
                if constraint[0, 1] >= 0:
                    plt.fill_between([-1, 1], [pt_1, pt_2], [1, 1], alpha=wt_shading, color='blue')
                else:
                    plt.fill_between([-1, 1], [pt_1, pt_2], [-1, -1], alpha=wt_shading, color='blue')

    wt_marker_size = 200
    # plot ground truth weight
    plt.scatter(weights[0, 0], weights[0, 1], s=wt_marker_size, color='red')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.show()

def extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, BEC_depth=1, trajectories=None):
    '''
    :param wt_vi_traj_candidates: Nested list of [weight, value iteration object, trajectory]
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :param BEC_depth (int): number of suboptimal actions to take before following the optimal policy to obtain the
                            suboptimal trajectory (and the corresponding suboptimal expected feature counts)
    :return: min_subset_constraints: List of constraints

    Summary: Obtain the constraints that comprise the BEC region of a set of demonstrations
    '''
    # go through each environment and corresponding optimal trajectory, and extract the behavior equivalence class (BEC) constraints
    constraints = []
    counter = 0

    for wt_vi_traj_candidate in wt_vi_traj_candidates:
        # print("Extracting constraints from environment {}".format(counter))
        mdp = wt_vi_traj_candidate[0][1].mdp

        if trajectories is not None:
            # a) demonstration-driven BEC
            # BEC constraints are obtained by ensuring that the optimal actions accumulate at least as much reward as
            # all other possible actions along a trajectory
            action_seq_list = list(itertools.product(mdp.actions, repeat=BEC_depth))

            traj = trajectories[counter]
            for sas_idx in range(len(traj)):
                # reward features of optimal action
                mu_sa = mdp.accumulate_reward_features(traj[sas_idx:], discount=True)

                sas = traj[sas_idx]
                cur_state = sas[0]

                # currently assumes that all actions are executable from all states
                for action_seq in action_seq_list:
                    traj_hyp = mdp_helpers.rollout_policy(mdp, FixedPolicyAgent(wt_vi_traj_candidate[0][1].policy), cur_state, action_seq)
                    mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)

                    constraints.append(mu_sa - mu_sb)
        else:
            # b) policy-driven BEC
            agent = FixedPolicyAgent(wt_vi_traj_candidate[0][1].policy)

            for state in mdp.states:
                action_opt = agent.act(state, 0)
                traj_opt = mdp_helpers.rollout_policy(mdp, agent, cur_state=state, action_seq=[action_opt])
                mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)

                # currently assumes that all actions are executable from all states
                for action in mdp.actions:
                    if action_opt != action:
                        traj_hyp = mdp_helpers.rollout_policy(mdp, agent, cur_state=state, action_seq=[action])
                        mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)

                        constraints.append(mu_sa - mu_sb)
        counter += 1

    normalized_constraints = normalize_constraints(constraints)
    if len(normalized_constraints) > 0:
        nonduplicate_constraints = remove_duplicate_constraints(normalized_constraints)
        if len(nonduplicate_constraints) > 1:
            min_subset_constraints = remove_redundant_constraints(nonduplicate_constraints, weights, step_cost_flag)
        else:
            min_subset_constraints = nonduplicate_constraints
    else:
        min_subset_constraints = normalized_constraints

    return min_subset_constraints

def count_new_covers(new_constraints, BEC_constraints, covered_BEC_constraints):
    '''
    :param new_constraints: New constraints imposed by the recent demo and/or policy (list of constraints)
    :param BEC_constraints: Minimum set of constraints defining the BEC of a set of demos / policy (list of constraints)
    :param covered_BEC_constraints: Bookmarking of BEC_constraints that are already accounted for (boolean list)
    :return: count: number of new BEC constraints that can be accounted for
    '''
    count = 0

    for new_constraint in new_constraints:
        for BEC_constraint_idx in range(len(BEC_constraints)):
            if equal_constraints(new_constraint, BEC_constraints[BEC_constraint_idx]) and not covered_BEC_constraints[BEC_constraint_idx]:
                count += 1

    return count

def update_covered_constraints(constraints_added, BEC_constraints, covered_BEC_constraints):
    '''
    :param constraints_added: New constraints imposed by the recent demo and/or policy (list of constraints)
    :param BEC_constraints: Minimum set of constraints defining the BEC of a set of demos / policy (list of constraints)
    :param covered_BEC_constraints: Bookmarking of BEC_constraints that are already accounted for (boolean list)

    Summary: Update the bookmarking with the newly added constraints
    '''
    for constraint_added in constraints_added:
        for BEC_constraint_idx in range(len(BEC_constraints)):
            if equal_constraints(constraint_added, BEC_constraints[BEC_constraint_idx]) and not covered_BEC_constraints[BEC_constraint_idx]:
                covered_BEC_constraints[BEC_constraint_idx] = True


def obtain_summary(wt_vi_traj_candidates, BEC_constraints, weights, step_cost_flag, summary_type, BEC_depth):
    '''
    :param wt_vi_traj_candidates: Nested list of [weight, value iteration object, trajectory]
    :param BEC_constraints: Minimum set of constraints defining the BEC of a set of demos / policy (list of constraints)
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost

    :return: summary: Nested list of [mdp, trajectory]

    Summary: Obtain a minimal set of a demonstrations that recovers the behavioral equivalence class (BEC) of a set of demos / policy.
    An implementation of 'Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications' (Brown et al. AAAI 2019).
    '''
    n_BEC_constraints = len(BEC_constraints)
    covered_BEC_constraints = np.zeros(n_BEC_constraints, dtype=bool)
    summary = []

    total_covered = 0
    while total_covered < n_BEC_constraints:
        max_count = float("-inf")

        for env_idx in range(len(wt_vi_traj_candidates)):
            print("Extracting constraints from environment {}".format(env_idx))

            if summary_type == 'demo':
                # a) only consider the optimal trajectories from the start states
                new_constraints = extract_constraints([wt_vi_traj_candidates[env_idx]], weights, step_cost_flag, BEC_depth=BEC_depth, trajectories=[wt_vi_traj_candidates[env_idx][0][2]])
                count = count_new_covers(new_constraints, BEC_constraints, covered_BEC_constraints)
                if count > max_count:
                    max_count = count
                    constraints_added = new_constraints
                    # I don't think a shallow copy is required since this won't be getting changed
                    best_traj = wt_vi_traj_candidates[env_idx][0][2]
                    best_mdp = wt_vi_traj_candidates[env_idx][0][1].mdp
                    print('New max count: {}'.format(max_count))
            else:
                # b) consider all possible trajectories by the optimal policy
                mdp = wt_vi_traj_candidates[env_idx][0][1].mdp
                agent = FixedPolicyAgent(wt_vi_traj_candidates[env_idx][0][1].policy)
                for state in mdp.states:
                    action_opt = agent.act(state, 0)
                    traj_opt = mdp_helpers.rollout_policy(mdp, agent, cur_state=state, action_seq=[action_opt])
                    new_constraints = extract_constraints([wt_vi_traj_candidates[env_idx]], weights, step_cost_flag, trajectories=[traj_opt])
                    count = count_new_covers(new_constraints, BEC_constraints, covered_BEC_constraints)

                    if count > max_count:
                        max_count = count
                        constraints_added = new_constraints
                        # I don't think a shallow copy is required since this won't be getting changed
                        best_traj = traj_opt
                        best_mdp = mdp
                        print('New max count: {}'.format(max_count))

        summary.append((best_mdp, best_traj, constraints_added))
        update_covered_constraints(constraints_added, BEC_constraints, covered_BEC_constraints)
        total_covered += max_count
        print("{}/{} BEC constraints covered".format(total_covered, n_BEC_constraints))

    return summary

def visualize_summary(BEC_summary, weights, step_cost_flag):
    '''
    :param BEC_summary: Nested list of [mdp, trajectory]

    Summary: visualize the BEC demonstrations
    '''
    for mdp_traj_constraint in BEC_summary:
        # visualize demonstration
        mdp_traj_constraint[0].visualize_trajectory(mdp_traj_constraint[1])
        # visualize constraints enforced by demonstration above
        # print(mdp_traj_constraint[2])
        # visualize_constraints(mdp_traj_constraint[2], weights, step_cost_flag)