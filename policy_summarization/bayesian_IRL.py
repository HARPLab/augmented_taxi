# Python imports.
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
from itertools import chain

# Other imports
from simple_rl.agents import FixedPolicyAgent

'''
Args:
    n_demonstrations (int): number of demonstrations to return in summary
    weights (list of floats): ground truth reward weights used by agent to derive its optimal policy
    wt_uniform_sampling (list of candidate reward weights)
    wt_vi_traj_candidates (nested list of candidate reward weights, and corresponding policies and trajectories)
    visualize (Boolean): visualize the progression of the prior probabilities of reward weights candidates

Returns:
    IRL_summary (list of MDP/policy and corresponding trajectories of best demonstrations)

Summary:
    An implementation of 'Enabling Robots to Communicate their Objectives' (Huang et al. AURO2019).
'''
def obtain_summary(n_demonstrations, weights, wt_uniform_sampling, wt_vi_traj_candidates, approximate=True, visualize=False):
    priors = {}                # up-to-date priors on candidates
    history_priors = {}        # a history of updated priors for debugging

    # initialize the prior to be a uniform distribution
    for wt_candidate in wt_uniform_sampling:
        priors[wt_candidate.tostring()] = 1. / len(wt_uniform_sampling)
        history_priors[wt_candidate.tostring()] = [1. / len(wt_uniform_sampling)]

    IRL_summary = []
    update_coeff = 0.01 # 10^-5 to 10^5 used by Huang et al. for approximate inference
    idx_of_true_wt = np.ndarray.tolist(wt_uniform_sampling).index(np.ndarray.tolist(weights))
    demo_count = 0

    while demo_count < n_demonstrations and len(wt_vi_traj_candidates) > 0:
        cond_posteriors = np.zeros(len(wt_vi_traj_candidates))
        cond_trajectory_likelihoods_trajectories = []

        # for each environment
        for k in range(len(wt_vi_traj_candidates)):
            Z = 0    # normalization factor
            wt_vi_traj_candidates_tuples = wt_vi_traj_candidates[k]
            trajectory = wt_vi_traj_candidates_tuples[idx_of_true_wt][2]
            cond_trajectory_likelihoods = {}

            # compute the normalization factor
            for wt_vi_traj_candidates_tuple in wt_vi_traj_candidates_tuples:
                wt_candidate = wt_vi_traj_candidates_tuple[0]
                vi_candidate = wt_vi_traj_candidates_tuple[1]
                trajectory_candidate = wt_vi_traj_candidates_tuple[2]

                if approximate is False:
                    # a) exact inference IRL
                    reward_diff = wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory).T) - wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate).T)
                    if reward_diff >= 0:
                        cond_trajectory_likelihood = 1
                    else:
                        cond_trajectory_likelihood = 0
                else:
                    # b) approximate inference IRL
                    # take the abs value in case you're working with partial trajectories, in which the comparative rewards
                    # for short term behavior differs from comparative rewards for long term behavior
                    reward_diff = abs((wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate).T) \
                                   - wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory).T))[0][0])
                    cond_trajectory_likelihood = np.exp(-update_coeff * reward_diff)

                cond_trajectory_likelihoods[wt_candidate.tostring()] = cond_trajectory_likelihood

                Z += cond_trajectory_likelihood * priors[wt_candidate.tostring()]


            cond_trajectory_likelihoods_trajectories.append(cond_trajectory_likelihoods)

            if approximate is False:
                # the reward diff should always be zero, assuming that the agent and the human are using the same reward
                # features and cond_trajectory_likelihood_star should always be 1. you can use the most recent vi_candidate's
                # MDP since the MDP should be shared by all within the same wt_vi_traj_candidate_tuples
                # a) exact inference IRL
                reward_diff_star = (weights.dot(vi_candidate.mdp.accumulate_reward_features(trajectory).T) - \
                                   weights.dot(vi_candidate.mdp.accumulate_reward_features(trajectory).T))[0][0]
                if reward_diff_star >= 0:
                    cond_trajectory_likelihood_star = 1
                else:
                    cond_trajectory_likelihood_star = 0
            else:
                # b) approximate inference IRL
                reward_diff_star = abs((weights.dot(vi_candidate.mdp.accumulate_reward_features(trajectory).T) - \
                                   weights.dot(vi_candidate.mdp.accumulate_reward_features(trajectory).T))[0][0])
                cond_trajectory_likelihood_star = np.exp(-update_coeff * reward_diff_star)

            # calculate what the new condition probability of the true weight vector would be given this demonstration
            cond_posteriors[k] = 1. / Z * cond_trajectory_likelihood_star * priors[
                weights.tostring()]

        # select the demonstration that maximally increases the conditional posterior probability of the true weight vector
        best_env = np.argmax(cond_posteriors)
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

    if visualize:
        # visualize the evolution of the prior distribution with each new demonstration
        history_priors_per_demo = []
        x = range(len(wt_uniform_sampling))

        # group the priors by demo, and not by weight
        for j in range(len(IRL_summary)):
            priors_per_wt_candidate = []
            for wt_candidate in wt_uniform_sampling:
                priors_per_wt_candidate.append(history_priors[wt_candidate.tostring()][j])
            history_priors_per_demo.append(priors_per_wt_candidate)

        # flatten the list of (x, history_priors_per_demo) tuples
        plt.plot(*list(chain.from_iterable([(x, history_priors_per_demo[j]) for j in range(len(history_priors_per_demo))])))
        plt.xlabel('Candidate reward weight vectors')
        plt.ylabel('Probability of candidates')
        plt.legend(['{}'.format(x) for x in range(len(IRL_summary))])
        plt.show()

    return IRL_summary, history_priors