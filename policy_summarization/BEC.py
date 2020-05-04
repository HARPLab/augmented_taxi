import matplotlib.pyplot as plt
from simple_rl.utils import mdp_helpers
from simple_rl.agents import FixedPolicyAgent
import numpy as np
from scipy.optimize import linprog

def normalize_constraints(constraints):
    '''
    Summary: Normalize all constraints such that the L2 norm is equal to 1
    '''
    normalized_constraints = []
    zero_constraint = np.zeros(constraints[0].shape)
    for constraint in constraints:
        if not equal_constraints(constraint, zero_constraint):
            normalized_constraints.append(constraint / np.linalg.norm(constraint))

    return normalized_constraints

def remove_duplicate_constraints(constraints):
    '''
    Summary: Remove any duplicate constraints
    '''
    nonredundant_constraints = []
    zero_constraint = np.array([[0., 0.]])

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

def remove_redundant_constraints(constraints):
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

        # solve linear program
        # min_x a^Tx, st -Ax >= -b
        a = np.ndarray.tolist(query_constraint[0])
        b = [0] * len(constraints_other)
        x0_bounds = (-1, 1)
        x1_bounds = (-1, 1)
        res = linprog(a, A_ub=constraints_other, b_ub=b, bounds=[x0_bounds, x1_bounds])

        # if query_constraint * res.x^T >= 0, then this constraint is redundant. copy over everything except this constraint
        if query_constraint.dot(res.x.reshape(-1, 1))[0][0] >= -1e-05: # account for slight numerical instability
            copy = []
            for nonredundant_constraint in nonredundant_constraints:
                if not equal_constraints(query_constraint, nonredundant_constraint):
                    copy.append(nonredundant_constraint)
            nonredundant_constraints = copy

    return nonredundant_constraints


def visualize_constraints(constraints, gt_weight=None):
    '''
    Summary: Visualize the constraints
    '''
    # for higher dimensional constraints, update this visualization function must be updated
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    wt_shading = 1. / len(constraints)

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
            plt.plot([-1, 1], [constraint[0, 0] / constraint[0, 1], -constraint[0, 0] / constraint[0, 1]])

            # use (0, 1) as a test point to decide which half space to color
            if constraint[0, 1] >= 0:
                plt.fill_between([-1, 1], [constraint[0, 0] / constraint[0, 1], -constraint[0, 0] / constraint[0, 1]],
                                 [1, 1], alpha=wt_shading, color='blue')
            else:
                plt.fill_between([-1, 1], [constraint[0, 0] / constraint[0, 1], -constraint[0, 0] / constraint[0, 1]],
                                 [-1, -1], alpha=wt_shading, color='blue')

    # plot ground truth weight
    if gt_weight is not None:
        plt.scatter(gt_weight[0, 0], gt_weight[0, 1], s=200, color='red')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.show()

def extract_constraints(wt_vi_traj_candidates):
    '''
    :param wt_vi_traj_candidates: Nested list of [weight, value iteration object, trajectory]
    :return: min_subset_constraints: List of constraints

    Summary: Obtain the constraints that comprise the BEC region of a set of demonstrations
    '''
    # go through each environment and corresponding optimal trajectory, and extract the behavior equivalence class (BEC) constraints
    constraints = []
    counter = 0
    for wt_vi_traj_candidate in wt_vi_traj_candidates:
        # print("Extracting constraints from environment {}".format(counter))
        mdp_demo = wt_vi_traj_candidate[0][1].mdp
        traj_demo = wt_vi_traj_candidate[0][2]

        # BEC constraints are obtained by ensuring that the optimal actions accumulate at least as much reward as
        # all other possible actions along a trajectory
        for sas_idx in range(len(traj_demo)):
            # reward features of optimal action
            mu_sa = mdp_demo.accumulate_reward_features(traj_demo[sas_idx:], discount=True)

            sas = traj_demo[sas_idx]
            cur_state = sas[0]
            cur_action = sas[1]

            # currently assumes that all actions are executable from all states
            for action in mdp_demo.actions:
                if cur_action != action:
                    traj_hyp = mdp_helpers.rollout_policy(mdp_demo, FixedPolicyAgent(wt_vi_traj_candidate[0][1].policy), cur_state, action)
                    mu_sb = mdp_demo.accumulate_reward_features(traj_hyp, discount=True)

                    constraints.append(mu_sa - mu_sb)
        counter += 1

    normalized_constraints = normalize_constraints(constraints)
    nonduplicate_constraints = remove_duplicate_constraints(normalized_constraints)
    if len(nonduplicate_constraints) > 1:
        min_subset_constraints = remove_redundant_constraints(nonduplicate_constraints)
    else:
        min_subset_constraints = nonduplicate_constraints

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


def obtain_summary(wt_vi_traj_candidates, BEC_constraints):
    '''
    :param wt_vi_traj_candidates: Nested list of [weight, value iteration object, trajectory]
    :param BEC_constraints: Minimum set of constraints defining the BEC of a set of demos / policy (list of constraints)
    :return: summary: Nested list of [mdp, trajectory]

    Summary: Obtain a minimal set of a demonstrations that recovers the behavioral equivalence class (BEC) of a set of demos / policy.
    An implementation of 'Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications' (Brown et al. AAAI 2019).
    '''
    n_BEC_constraints = len(BEC_constraints)
    covered_BEC_constraints = np.zeros(n_BEC_constraints, dtype=bool)
    summary = []

    total_covered = 0
    while total_covered < n_BEC_constraints:
        max_count = 0

        for traj_idx in range(len(wt_vi_traj_candidates)):
            print("Extracting constraints from environment {}".format(traj_idx))
            new_constraints = extract_constraints([wt_vi_traj_candidates[traj_idx]])

            count = count_new_covers(new_constraints, BEC_constraints, covered_BEC_constraints)

            if count > max_count:
                max_count = count
                constraints_added = new_constraints
                # I don't think a shallow copy is required since this won't be getting changed
                best_traj = wt_vi_traj_candidates[traj_idx][0][2]
                best_mdp = wt_vi_traj_candidates[traj_idx][0][1].mdp

        summary.append((best_mdp, best_traj))
        update_covered_constraints(constraints_added, BEC_constraints, covered_BEC_constraints)
        total_covered += max_count
        print("{}/{} BEC constraints covered".format(total_covered, n_BEC_constraints))

    return summary

def visualize_summary(BEC_summary):
    '''
    :param BEC_summary: Nested list of [mdp, trajectory]

    Summary: visualize the BEC demonstrations
    '''
    for mdp_traj in BEC_summary:
        mdp_traj[0].visualize_trajectory(mdp_traj[1])