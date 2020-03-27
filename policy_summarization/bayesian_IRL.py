# Python imports.
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np

# Other imports
from simple_rl.agents import FixedPolicyAgent

'''
An implementation of 'Enabling Robots to Communicate their Objectives' (Huang et al. AURO2019).

Minor differences from the original algorithm currently include 1) selecting from segments from an optimal trajectory 
in a single environment for demonstrations, rather than selecting from a set of optimal demonstrations in multiple 
environments.
'''

def obtain_summary(mdp_agent, n_demonstrations, weights, wt_vi_candidates, trajectories, visualize=False):
    priors = {}                # up-to-date priors on candidates
    debugging_priors = {}      # a history of updated priors for debugging

    # initialize the prior to be a uniform distribution
    for wt_vi_candidate_pair in wt_vi_candidates:
        wt_candidate = wt_vi_candidate_pair[0]
        priors[wt_candidate.tostring()] = 1. / len(wt_vi_candidates)
        debugging_priors[wt_candidate.tostring()] = [1. / len(wt_vi_candidates)]

    IRL_summary = []
    update_coeff = 0.01 # 10^-5 to 10^5 used by Huang et al. for approximate inference

    for j in range(n_demonstrations):
        cond_posteriors = np.zeros(len(trajectories))
        cond_trajectory_likelihoods_trajectories = []

        for k in range(len(trajectories)):
            Z = 0    # normalization factor
            trajectory = trajectories[k]
            cond_trajectory_likelihoods = {}

            # compute the normalization factor
            for wt_vi_candidate_pair in wt_vi_candidates:
                wt_candidate = wt_vi_candidate_pair[0]
                vi_candidate = wt_vi_candidate_pair[1]
                trajectory_candidate = []  # optimal trajectory for this weight candidate

                # obtain optimal trajectory for this weight candidate
                vi_candidate_policy = FixedPolicyAgent(vi_candidate.policy)
                for sas_prime in trajectory:
                    action = vi_candidate_policy.act(sas_prime[0], 0)
                    s_prime = vi_candidate.mdp._taxi_transition_func(sas_prime[0], action)

                    trajectory_candidate.append((sas_prime[0], action, s_prime))

                # a) exact inference IRL
                # reward_diff = wt_candidate.dot(mdp_agent.accumulate_reward_features(trajectory).T) - wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate).T)
                # if reward_diff >= 0:
                #     cond_trajectory_likelihood = 1
                # else:
                #     cond_trajectory_likelihood = 0

                # b) approximate inference IRL
                # take the abs value in case you're working with partial trajectories, in which the comparative rewards
                # for short term behavior differs from comparative rewards for long term behavior
                reward_diff = abs((wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate).T) \
                               - wt_candidate.dot(mdp_agent.accumulate_reward_features(trajectory).T))[0][0])
                cond_trajectory_likelihood = np.exp(-update_coeff * reward_diff)

                cond_trajectory_likelihoods[wt_candidate.tostring()] = cond_trajectory_likelihood

                Z += cond_trajectory_likelihood * priors[wt_candidate.tostring()]


            cond_trajectory_likelihoods_trajectories.append(cond_trajectory_likelihoods)


            # a) exact inference IRL
            # reward_diff_star = (weights.dot(mdp_agent.accumulate_reward_features(trajectory).T) - \
            #                    weights.dot(mdp_agent.accumulate_reward_features(trajectory).T))[0][0]
            # if reward_diff_star >= 0:
            #     cond_trajectory_likelihood_star = 1
            # else:
            #     cond_trajectory_likelihood_star = 0

            # b) approximate inference IRL
            # this should always be zero, assuming that the agent and the human are using the same reward features
            reward_diff_star = abs((weights.dot(mdp_agent.accumulate_reward_features(trajectory).T) - \
                               weights.dot(mdp_agent.accumulate_reward_features(trajectory).T))[0][0])
            cond_trajectory_likelihood_star = np.exp(-update_coeff * reward_diff_star)

            # calculate what the new condition probability of the true weight vector would be given this demonstration
            cond_posteriors[k] = 1. / Z * cond_trajectory_likelihood_star * priors[weights.tostring()]

        # select the demonstration that maximally increases the conditional posterior probability of the true weight vector
        best_traj = np.argmax(cond_posteriors)
        print(colored('Best trajectory: {}'.format(best_traj), 'red'))
        IRL_summary.append(trajectories[best_traj])
        # remove this demonstration from further consideration
        trajectories.pop(best_traj)

        # update the prior distribution
        for wt_vi_candidate_pair in wt_vi_candidates:
            wt_candidate = wt_vi_candidate_pair[0]
            old_prior = priors[wt_candidate.tostring()]
            updated_prior = old_prior * cond_trajectory_likelihoods_trajectories[best_traj][wt_candidate.tostring()]
            priors[wt_candidate.tostring()] = updated_prior
            debugging_priors[wt_candidate.tostring()].append(updated_prior)

            if np.array_equal(wt_candidate, weights[0]):
                print(colored('wt: {}, prior: {}, updated_prior: {}'.format(np.round(wt_candidate, 2), old_prior, updated_prior), 'red'))
            else:
                print('wt: {}, prior: {}, updated_prior: {}'.format(np.round(wt_candidate, 2), old_prior, updated_prior))

    if visualize:
        # visualize the evolution of the prior distribution with each new demonstration
        for j in range(n_demonstrations):
            x = range(len(wt_vi_candidates))
            y = []
            for wt_vi_candidate_pair in wt_vi_candidates:
                wt_candidate = wt_vi_candidate_pair[0]
                y.append(debugging_priors[wt_candidate.tostring()][j])
            plt.plot(x, y)
            plt.xlabel('Candidate reward weight vectors')
            plt.ylabel('Probability of candidates')
            plt.show()

    return IRL_summary