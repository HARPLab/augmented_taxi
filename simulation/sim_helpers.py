# This script contains functions to run simulation experiments.


import dill as pickle
import itertools
import numpy as np

from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_helpers

import policy_summarization.BEC_helpers as BEC_helpers
import simulation.human_learner_model as hlm
from termcolor import colored
import matplotlib.pyplot as plt
import params_team as params



def get_human_response(env_idx, env_cnst, opt_traj, likelihood_correct_response):

    if params.debug_hm_sampling:
        print('Generating human response for environment constraint: ', env_cnst[0])

    human_model = hlm.cust_pdf_uniform(env_cnst[0], likelihood_correct_response)

    # print('human model kappa: ', human_model.kappa)

    filename = 'models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'
    
    with open(filename, 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    mdp = wt_vi_traj_env[0][1].mdp

    agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
    mdp.set_init_state(opt_traj[0][0])
    traj_opt = mdp_helpers.rollout_policy(mdp, agent)

    # mdp.visualize_state(mdp.get_init_state())
    # plt.show()

    
    # print(traj_opt)
    # print(opt_traj)

    # opt_overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(traj_opt, opt_traj)
    # print('Optimal trajectory overlap percentage: ', opt_overlap_pct)
    
    likely_correct_response_count = 0
    likely_incorrect_response_count = 0

    skip_human_model = True

    loop_count = 0
    max_loop_count = 100
    differing_response_model_idx = []

    while skip_human_model:
        if params.debug_hm_sampling:
            print('Sampling a human model ...')
        cust_samps = human_model.rvs(size=1) # sample a human model (weight vector) from the distribution created from the constraint and the likelihood of sampling a correct response for that constraint
        if params.debug_hm_sampling:
            print('Sampled models: ', cust_samps)

        dot = cust_samps[0].dot(env_cnst[0])
        if dot >= 0:
            # print('Possibly a correct response sampled')
            likely_correct_response_count += 1
            likely_response_type = 'correct'
        else:
            # print('Possibly an incorrect response sampled')
            likely_incorrect_response_count += 1
            likely_response_type = 'incorrect'

        if loop_count == 0:
            initial_likely_response_type = likely_response_type


        for model_idx, human_model_weight in enumerate(cust_samps):

            # print('Sampled model: ', human_model_weight)

            mdp.weights = human_model_weight
            # vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=100)
            vi_human = ValueIteration(mdp, sample_rate=1)
            vi_human.run_vi()

            if not vi_human.stabilized:
                if params.debug_hm_sampling:
                    print(colored('Human model ' + str(model_idx) + ' did not converge and skipping for response generation', 'red'))
                skip_human_model = True
            else:
                if params.debug_hm_sampling:
                    print(colored('Human model ' + str(model_idx) + ' converged', 'green'))
                skip_human_model = False
            
            if not skip_human_model:
                cur_state = mdp.get_init_state()
                # print('Current state: ', cur_state)
                human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
                
                human_traj_rewards = mdp.accumulate_reward_features(human_opt_trajs[0], discount=True)  # just use the first optimal trajectory
                mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
                new_constraint = mu_sa - human_traj_rewards

                # print('New constraint: ', new_constraint)
                if (new_constraint == np.array([0, 0, 0])).all():
                    if params.debug_hm_sampling:
                        print(colored('Correct response sampled', 'blue'))
                    response_type = 'correct'
                else:
                    if params.debug_hm_sampling:
                        print(colored('Incorrect response sampled', 'red'))
                    response_type = 'incorrect'

                # check if the response matches the initial likely guess
                if response_type != initial_likely_response_type and loop_count < max_loop_count:
                    skip_human_model = True
                    differing_response_model_idx.append([model_idx, human_model_weight])
                    if params.debug_hm_sampling:
                        print(colored('Different from initial repsonse. Skipping ...', 'red'))
                        print('Initial likely response type: ', initial_likely_response_type, 'Current response type: ', response_type)
                elif response_type != initial_likely_response_type and loop_count >= max_loop_count:
                    alt_human_model_weight = differing_response_model_idx[0][1] # choose a previously converged model that differs from the initial likely response
                    mdp.weights = alt_human_model_weight
                    vi_human = ValueIteration(mdp, sample_rate=1)
                    vi_human.run_vi()
                    human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
                    if params.debug_hm_sampling:
                        print(colored('Human model chose from differing repsonse. ', 'green'))
                
                
                # mdp.visualize_trajectory(traj_opt)
                # plt.show()
                # mdp.visualize_trajectory(human_opt_trajs[0])
                # plt.show()

        loop_count += 1

    # human_traj_overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(human_opt_trajs[0], traj_opt)
    # print('human traj overlap pct: ', human_traj_overlap_pct)

    # return human_opt_trajs, response_type, likely_correct_response_count, likely_incorrect_response_count, initial_likely_response_type

    return human_opt_trajs[0], response_type





# def get_human_response_old(env_idx, env_cnst, opt_traj, human_history, team_knowledge, team_size = 2, response_distribution = 'correct'):

#     human_traj = []
#     cnst = []

#     # a) find the sub_optimal responses
#     BEC_depth_list = [1]

#     filename = '../models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'
    
#     with open(filename, 'rb') as f:
#         wt_vi_traj_env = pickle.load(f)

#     mdp = wt_vi_traj_env[0][1].mdp
#     agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
#     mdp.set_init_state(opt_traj[0][0])
    

#     constraints_list_correct = []
#     human_trajs_list_correct = []
#     constraints_list_incorrect = []
#     human_trajs_list_incorrect = []


#     for BEC_depth in BEC_depth_list:
#         # print('BEC_depth: ', BEC_depth)
#         action_seq_list = list(itertools.product(mdp.actions, repeat=BEC_depth))

#         traj_opt = mdp_helpers.rollout_policy(mdp, agent)
#         # print('Optimal Trajectory length: ', len(traj_opt))
#         traj_hyp = []

#         for sas_idx in range(len(traj_opt)):
        
#             # reward features of optimal action
#             mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

#             sas = traj_opt[sas_idx]
#             cur_state = sas[0]
#             # if sas_idx > 0:
#             #     traj_hyp = traj_opt[:sas_idx-1]

#             # currently assumes that all actions are executable from all states
#             for action_seq in action_seq_list:
#                 if sas_idx > 0:
#                     traj_hyp = traj_opt[:sas_idx-1]

#                 traj_hyp_human = mdp_helpers.rollout_policy(mdp, agent, cur_state=cur_state, action_seq=action_seq)
#                 traj_hyp.extend(traj_hyp_human)
                
#                 mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)
#                 new_constraint = mu_sa - mu_sb

#                 count = sum(np.array_equal(new_constraint, arr) for arr in constraints_list_correct) + sum(np.array_equal(new_constraint, arr) for arr in constraints_list_incorrect)

#                 # if count < team_size: # one sample trajectory for each constriant is sufficient; but just for a variety gather one trajectory for each person for each constraint, if possible
#                     # print('Hyp traj len: ', len(traj_hyp))
#                     # print('new_constraint: ', new_constraint)
#                 if (new_constraint == np.array([0, 0, 0])).all():
#                     constraints_list_correct.append(env_cnst)
#                     human_trajs_list_correct.append(traj_opt) 
#                 else:
#                     constraints_list_incorrect.append(new_constraint)
#                     human_trajs_list_incorrect.append(traj_hyp)

           
#     print('Constraints list correct: ', len(constraints_list_correct))
#     print('Constraints list incorrect: ', len(constraints_list_incorrect))
    
#     # b) find the counterfactual human responses
#     sample_human_models = BEC_helpers.sample_human_models_uniform([], 8)

#     for model_idx, human_model in enumerate(sample_human_models):

#         mdp.weights = human_model
#         vi_human = ValueIteration(mdp, sample_rate=1)
#         vi_human.run_vi()

#         if not vi_human.stabilized:
#             skip_human_model = True
#             print(colored('Human model ' + str(model_idx) + ' did not converge and skipping for response generation', 'red'))
        
#         if not skip_human_model:
#             human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
#             for human_opt_traj in human_opt_trajs:
#                 human_traj_rewards = mdp.accumulate_reward_features(human_opt_traj, discount=True)
#                 mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
#                 new_constraint = mu_sa - human_traj_rewards

#                 count = sum(np.array_equal(new_constraint, arr) for arr in constraints_list_correct) + sum(np.array_equal(new_constraint, arr) for arr in constraints_list_incorrect)

#                 # if count < team_size:
#                     # print('Hyp traj len: ', len(traj_hyp))
#                     # print('new_constraint: ', new_constraint)
#                 if (new_constraint == np.array([0, 0, 0])).all():
#                     constraints_list_correct.append(env_cnst)
#                     human_trajs_list_correct.append(traj_opt) 
#                 else:
#                     constraints_list_incorrect.append(new_constraint)
#                     human_trajs_list_incorrect.append(human_opt_traj)

#     print('Constraints list correct after human models: ', len(constraints_list_correct))
#     print('Constraints list incorrect after human models: ', len(constraints_list_incorrect))
    

#     # Currently coded for a team size of 2
#     if response_distribution == 'correct':

#         for i in range(team_size):
#             random_index = random.randint(0, len(constraints_list_correct)-1)
#             human_traj.append(human_trajs_list_correct[random_index])
#             cnst.append(constraints_list_correct[random_index])

#             constraints_list_correct.pop(random_index)
#             human_trajs_list_correct.pop(random_index)
        
#     elif response_distribution == 'incorrect':
#         for i in range(team_size):
#             member_id = 'p' + str(i+1)
#             random_index = random.randint(0, len(constraints_list_incorrect)-1)
#             while len([x for x in team_knowledge[member_id] if (x == constraints_list_incorrect[random_index]).all()]) or norm(constraints_list_incorrect[random_index], 1) > 8:  # additional check to ensure constraints are different from existing knowledge (just for generating examples for papers)
#                 random_index = random.randint(0, len(constraints_list_incorrect)-1)

#             human_traj.append(human_trajs_list_incorrect[random_index])
#             cnst.append(constraints_list_incorrect[random_index])

#             constraints_list_correct.append(env_cnst)
#             human_trajs_list_correct.append(traj_opt)

#             constraints_list_incorrect.pop(random_index)
#             human_trajs_list_incorrect.pop(random_index)

#     elif response_distribution == 'mixed':
#         constraints_list_correct_used = []
#         for i in range(team_size):
#             if i%2 == 0:
#                 print('len constraints_list_correct: ', len(constraints_list_correct))
#                 random_index = random.randint(0, len(constraints_list_correct)-1)
#                 human_traj.append(human_trajs_list_correct[random_index])
#                 cnst.append(constraints_list_correct[random_index])
#                 constraints_list_correct_used.append(constraints_list_correct[random_index])
#                 constraints_list_correct.pop(random_index)
#                 human_trajs_list_correct.pop(random_index)
#             else:
#                 indices = [i for i in range(len(constraints_list_incorrect)) if np.array_equal(-constraints_list_incorrect[i], constraints_list_correct_used[-1])]
#                 print('constraints_list_correct_used: ', constraints_list_correct_used)
#                 print('opposing indices: ', indices, 'for constraint: ', constraints_list_correct_used[-1])
#                 print('constraints_list_incorrect: ', constraints_list_incorrect)
#                 if len(indices) > 0: 
#                     member_id = 'p' + str(i+1)
#                     random_index = random.choice(indices) 
#                     while len([x for x in team_knowledge[member_id] if (x == constraints_list_incorrect[random_index]).all()]) or norm(constraints_list_incorrect[random_index], 1) > 8:  # additional check to ensure constraints are different from existing knowledge (just for generating examples for papers)
#                         random_index = random.choice(indices)    # opposing incorrect response
#                 else:
#                     member_id = 'p' + str(i+1)
#                     random_index = random.randint(0, len(constraints_list_incorrect)-1)
#                     while len([x for x in team_knowledge[member_id] if (x == constraints_list_incorrect[random_index]).all()]) or norm(constraints_list_incorrect[random_index], 1) > 8:  # additional check to ensure constraints are different from existing knowledge (just for generating examples for papers)
#                         random_index = random.randint(0, len(constraints_list_incorrect)-1)  # random incorrect response
                
#                 human_traj.append(human_trajs_list_incorrect[random_index])
#                 cnst.append(constraints_list_incorrect[random_index])

#                 constraints_list_incorrect.pop(random_index)
#                 human_trajs_list_incorrect.pop(random_index)

#     human_history.append((env_idx, human_traj, cnst))

#     # print('N of human_traj: ', len(human_traj))

#     # print('Visualizing human trajectory ....')
#     # for ht in human_traj:
#     #     print('human_traj len: ', len(ht))
#     #     print('constraint: ', cnst[human_traj.index(ht)])
#     #     mdp.visualize_trajectory(ht)

#         # Later: Check that test responses are not getting repeated for the same environment
#         # TODO: Check if the trajectories generated are valid - seems like diagonal moves are occuring sometimes



#     return human_traj, human_history 
