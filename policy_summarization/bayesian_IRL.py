# Python imports.
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
import itertools


def obtain_summary(n_demonstrations, weights, wt_uniform_sampling, wt_vi_traj_candidates, eval_fn):
    '''
    Args:
        n_demonstrations (int): number of demonstrations to return in summary
        weights (list of floats): ground truth reward weights used by agent to derive its optimal policy
        wt_uniform_sampling (list of candidate reward weights)
        wt_vi_traj_candidates (nested list of candidate reward weights, and corresponding policies and trajectories)
        eval_fn (string): desired likelihood function for computing the posterior probability of weight candidates

    Returns:
        IRL_summary (list of MDP/policy and corresponding trajectories of best demonstrations)

    Summary:
        An implementation of 'Enabling Robots to Communicate their Objectives' (Huang et al. AURO2019).
    '''
    priors = {}                # up-to-date priors on candidates
    history_priors = {}        # a history of updated priors for debugging

    # parse the desired evaluation function codes
    codes = eval_fn.split('_')
    inf_type = codes[0]
    eval_type = codes[1]

    # initialize the prior to be a uniform distribution
    for wt_candidate in wt_uniform_sampling:
        priors[wt_candidate.tostring()] = 1. / len(wt_uniform_sampling)
        history_priors[wt_candidate.tostring()] = [1. / len(wt_uniform_sampling)]

    IRL_summary = []
    update_coeff = 0.01 # 10^-5 to 10^5 used by Huang et al. for approximate inference
    idx_of_true_wt = np.ndarray.tolist(wt_uniform_sampling).index(np.ndarray.tolist(weights))
    demo_count = 0

    while demo_count < n_demonstrations and len(wt_vi_traj_candidates) > 0:
        if eval_type == 'MP':
            cond_posteriors = np.zeros(len(wt_vi_traj_candidates))
        else:
            cond_posteriors = np.zeros((len(wt_vi_traj_candidates), len(wt_uniform_sampling)))
        cond_trajectory_likelihoods_trajectories = []

        # for each environment
        for env_idx in range(len(wt_vi_traj_candidates)):
            Z = 0    # normalization factor
            wt_vi_traj_candidates_tuples = wt_vi_traj_candidates[env_idx]
            trajectory = wt_vi_traj_candidates_tuples[idx_of_true_wt][2]
            cond_trajectory_likelihoods = {}

            # compute the normalization factor
            for wt_vi_traj_candidates_tuple in wt_vi_traj_candidates_tuples:
                wt_candidate = wt_vi_traj_candidates_tuple[0]
                vi_candidate = wt_vi_traj_candidates_tuple[1]
                trajectory_candidate = wt_vi_traj_candidates_tuple[2]

                if inf_type == 'exact':
                    # a) exact inference IRL
                    reward_diff = wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory, discount=True).T) - wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate, discount=True).T)
                    if reward_diff >= 0:
                        cond_trajectory_likelihood = 1
                    else:
                        cond_trajectory_likelihood = 0
                elif inf_type == 'approx':
                    # b) approximate inference IRL
                    # take the abs value in case you're working with partial trajectories, in which the comparative rewards
                    # for short term behavior differs from comparative rewards for long term behavior
                    reward_diff = abs((wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate, discount=True).T) \
                                   - wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory, discount=True).T))[0][0])
                    cond_trajectory_likelihood = np.exp(-update_coeff * reward_diff)
                else:
                    raise ValueError("Error: The requested inference type is invalid.")

                cond_trajectory_likelihoods[wt_candidate.tostring()] = cond_trajectory_likelihood

                Z += cond_trajectory_likelihood * priors[wt_candidate.tostring()]

            cond_trajectory_likelihoods_trajectories.append(cond_trajectory_likelihoods)

            if eval_type == 'MP':
                # calculate what the new condition probability of the true weight vector would be given this demonstration
                cond_posteriors[env_idx] = 1. / Z * cond_trajectory_likelihoods[weights.tostring()] * priors[weights.tostring()]
            else:
                for wt_cand_idx in range(len(wt_uniform_sampling)):
                    wt_candidate = wt_uniform_sampling[wt_cand_idx]
                    cond_posteriors[env_idx, wt_cand_idx] = 1. / Z * cond_trajectory_likelihoods[wt_candidate.tostring()] * priors[wt_candidate.tostring()]

        if eval_type == 'MP':
            # a) select the demonstration that maximally increases the conditional posterior probability of the true weight vector (MP)
            best_env = np.argmax(cond_posteriors)
        elif eval_type == 'GP':
            # b) select the demonstration that maximally increases gap between the conditional posteriors of the true and the second best weight vector (GP)
            acq_vals = np.zeros(len(wt_vi_traj_candidates))
            cond_posteriors_sans = np.delete(cond_posteriors, idx_of_true_wt, 1)
            max_idx = np.argmax(cond_posteriors_sans, axis=1)
            max_vals = [cond_posteriors_sans[j][max_idx[j]] for j in range(max_idx.shape[0])]
            for env_idx in range(len(max_vals)):
                diff = cond_posteriors[env_idx][idx_of_true_wt] - max_vals[env_idx]
                acq_vals[env_idx] = diff
            best_env = np.argmax(acq_vals)
        elif eval_type == 'VOL':
            # c) select the demonstration that maximally increases the conditional posterior probability of the true weight vector
            # and minimizes the condition posterior probabilities of incorrect weight vectors (VOL)
            acq_vals = np.zeros(len(wt_vi_traj_candidates))
            for env_idx in range(len(wt_vi_traj_candidates)):
                acq_val = 0
                for wt_cand_idx in range(len(wt_uniform_sampling)):
                    wt_candidate = wt_uniform_sampling[wt_cand_idx]
                    if wt_cand_idx == idx_of_true_wt:
                        acq_val += cond_posteriors[env_idx, wt_cand_idx] - priors[wt_candidate.tostring()]
                    else:
                        acq_val += priors[wt_candidate.tostring()] - cond_posteriors[env_idx, wt_cand_idx]
                acq_vals[env_idx] = acq_val
            best_env = np.argmax(acq_vals)
            print(colored("Best acq_val: {}".format(np.max(acq_vals)), 'red'))
        else:
            raise ValueError("Error: The requested evaluation type is invalid.")

        print(colored('Best environment: {}'.format(best_env), 'red'))
        # store the MDP/policy and corresponding trajectory of the best next demonstration
        IRL_summary.append((wt_vi_traj_candidates[best_env][idx_of_true_wt][1], wt_vi_traj_candidates[best_env][idx_of_true_wt][2]))
        # remove this demonstration from further consideration
        wt_vi_traj_candidates.pop(best_env)

        # update the prior distribution
        prior_sum = 0.0
        for wt_candidate in wt_uniform_sampling:
            old_prior = priors[wt_candidate.tostring()]
            updated_prior = old_prior * cond_trajectory_likelihoods_trajectories[best_env][wt_candidate.tostring()]
            priors[wt_candidate.tostring()] = updated_prior
            prior_sum += updated_prior

        # normalize the prior distribution
        for wt_candidate in wt_uniform_sampling:
            normalized_prior = priors[wt_candidate.tostring()] / prior_sum
            priors[wt_candidate.tostring()] = normalized_prior
            history_priors[wt_candidate.tostring()].append(normalized_prior)

            if np.array_equal(wt_candidate, weights[0]):
                print(colored('wt: {}, updated_prior: {}'.format(np.round(wt_candidate, 3), normalized_prior), 'red'))
            else:
                print('wt: {}, updated_prior: {}'.format(np.round(wt_candidate, 3), normalized_prior))

        demo_count += 1

    return IRL_summary, wt_uniform_sampling, history_priors

def visualize_summary(summary, wt_uniform_sampling, history_priors, visualize_summary=True, visualize_history_priors=True):
    '''
    :param summary: Bayesian IRL summary (nested list of [MDP/policy, trajectory])
    :param wt_uniform_sampling: Candidate weights considered (numpy ndarray)
    :param history_priors: History of normalized probabilities of each candidate weight being to the true weight (nested list)
    :param visualize_demos: Boolean
    :param visualize_history_priors: Boolean

    Summary: Visualize the demonstrations comprising the Bayesian IRL summary and/or the update history of the
    probabilities of the candidate weights
    '''
    if visualize_summary:
        for policy_traj_tuple in summary:
            mdp_demo = policy_traj_tuple[0].mdp
            mdp_demo.visualize_trajectory(policy_traj_tuple[1])

    if visualize_history_priors:
        # visualize the evolution of the prior distribution with each new demonstration
        history_priors_per_demo = []
        x = range(len(wt_uniform_sampling))

        # group the priors by demo, and not by weight
        for j in range(len(summary) + 1):
            priors_per_wt_candidate = []
            for wt_candidate in wt_uniform_sampling:
                priors_per_wt_candidate.append(history_priors[wt_candidate.tostring()][j])
            history_priors_per_demo.append(priors_per_wt_candidate)

        # flatten the list of (x, history_priors_per_demo) tuples
        plt.plot(
            *list(itertools.chain.from_iterable([(x, history_priors_per_demo[j]) for j in range(len(history_priors_per_demo))])))
        plt.xlabel('Candidate reward weight vectors')
        plt.ylabel('Probability of candidates')
        plt.legend(['{}'.format(x) for x in range(len(summary) + 1)], bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.xticks(range(len(wt_uniform_sampling)), wt_uniform_sampling, rotation=90)
        plt.show()