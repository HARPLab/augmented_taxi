# This script contains functions to run simulation experiments.


import dill as pickle
import itertools
import numpy as np

from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_helpers

import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz
import simulation.human_learner_model as hlm
from termcolor import colored
import matplotlib.pyplot as plt
import params_team as params
import random


def sample_from_distribution(condition, points, probabilities, points_to_avoid = []):
    # Step 1: Normalize Probabilities
    total_prob = sum(probabilities)
    normalized_probs = [p / total_prob for p in probabilities]

    # Use zip to combine the arrays into pairs
    combined = list(zip(points, normalized_probs))

    # print('Combined: ', combined)

    # Sort based on the first array (array1 in this case)
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # print('Sorted combined: ', sorted_combined)

    # Unzip the sorted pairs back into separate arrays
    points_sorted, probabilities_sorted = zip(*sorted_combined)

    # Step 2: Compute Cumulative Distribution Function (CDF)
    cdf = [sum(probabilities_sorted[:i+1]) for i in range(len(probabilities_sorted))]

    # Step 3: Sampling
    sampling_complete = False
    loop_count = 0
    sampled_point = []

    while not sampling_complete and loop_count < 100:
        r = random.uniform(0, 1)
        print('Loop count: ', loop_count)
        for i, cumulative_prob in enumerate(cdf):
            if cumulative_prob >= r:
                pot_sampled_point = points_sorted[i]
                
                if condition == 'cluster_random' or condition == 'cluster_weight':
                    # Method 1/3: Sample from clusters. Here the points are cluster centers
                    cluster_id = [j for j, x in enumerate(points) if (x == pot_sampled_point).all()]
                    print('Sampled cluster_id: ', cluster_id)
                    if cluster_id not in points_to_avoid:
                        print('Sampled point id: ', i, 'Sampled point: ', pot_sampled_point, 'Sampled point probability: ', probabilities_sorted[i])
                        sampling_complete = True
                        sampled_point = pot_sampled_point
                        break
                elif condition == 'particles':
                    # Method 2: Sample from all particles
                    check_sampled = [j for j, x in enumerate(points_to_avoid) if (x == pot_sampled_point).all()]
                    if len(check_sampled)==0:
                        print('Sampled point id: ', i, 'Sampled point: ', pot_sampled_point, 'Sampled point probability: ', probabilities_sorted[i])
                        sampled_point = pot_sampled_point
                        sampling_complete = True 
                        break   # from for loop



        loop_count += 1

    
    return sampled_point, i, probabilities_sorted[i]


# def sample_from_distribution_cluster(particles, clusters_to_avoid = []):

#     # Step 1: Normalize Probabilities
#     total_prob = sum(particles.cluster_weights)
#     normalized_probs = [p / total_prob for p in particles.cluster_weights]

#     # Use zip to combine the arrays into pairs
#     combined = list(zip(particles.cluster_centers, normalized_probs))

#     # Sort based on the first array (array1 in this case)
#     sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

#     # Unzip the sorted pairs back into separate arrays
#     clusters_sorted, probabilities_sorted = zip(*sorted_combined)

#     # # Step 2: Compute Cumulative Distribution Function (CDF)
#     cdf = [sum(probabilities_sorted[:i+1]) for i in range(len(probabilities_sorted))]

#     # Step 3: Sampling
#     sampling_complete = False
#     loop_count = 0
#     sampled_point = []
    
#     while not sampling_complete and loop_count < 100:
#         r = random.uniform(0, 1)
#         print('Loop count: ', loop_count)
#         for i, cumulative_prob in enumerate(cdf):
#             if cumulative_prob >= r:
#                 sampled_cluster = clusters_sorted[i]

#                 # Method 1: Sample from clusters and then a random sampling of particles based on their weights within the cluster
#                 cluster_id = [j for j, x in enumerate(particles.cluster_weights) if (x == sampled_cluster).all()]
#                 print('Sampled cluster_id: ', cluster_id)
#                 if cluster_id not in clusters_to_avoid:

#                     sampled_point, skip_cluster_flag, particle_prob = sample_from_cluster_prob(particles, cluster_id, mdp)
                    
#                     print('Sampled point id: ', i, 'Sampled cluster: ', pot_sampled_point, 'Sampled point probability: ', probabilities_sorted[i])
#                     sampling_complete = True
#                     sampled_point = pot_sampled_point


#                     break




def sample_from_cluster_prob(particles_to_sample, cluster_id, mdp):
    sampled_model_flag = False
    prev_sampled_flag = False
    skip_cluster_flag = False
    prev_sampled_particles = []

    points_to_sample = particles_to_sample.positions[np.where(particles_to_sample.cluster_assignments == cluster_id)[0]]
    points_prob = particles_to_sample.weights[np.where(particles_to_sample.cluster_assignments == cluster_id)[0]]
    
    # points_to_sample_all = particles_to_sample.positions[np.where(particles_to_sample.cluster_assignments == cluster_id)[0]]
    # points_prob_all = particles_to_sample.weights[np.where(particles_to_sample.cluster_assignments == cluster_id)[0]]

    # indices_sample_from = [j for j,x in enumerate(points_to_sample_all) if (x[0][0] < 0 and x[0][2] < 1)]
    # points_to_sample = points_to_sample_all[indices_sample_from]
    # points_prob = points_prob_all[indices_sample_from]


    # Step 1: Normalize Probabilities
    total_prob = sum(points_prob)
    normalized_probs = [p / total_prob for p in points_prob]

    # Use zip to combine the arrays into pairs
    combined = list(zip(points_to_sample, normalized_probs))

    # print('combined: ', combined)

    # Sort based on the first array (array1 in this case)
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Unzip the sorted pairs back into separate arrays
    points_sorted, probabilities_sorted = zip(*sorted_combined)

    # # Step 2: Compute Cumulative Distribution Function (CDF)
    cdf = [sum(probabilities_sorted[:i+1]) for i in range(len(probabilities_sorted))]
    

    while not sampled_model_flag:
        print('Sampled particle: ' + str(len(prev_sampled_particles)) + ' out of ' + str(len(points_to_sample)) + ' particles')
        if len(prev_sampled_particles) < min(max(0.2*len(points_to_sample), 5), len(points_to_sample)):
            r = random.uniform(0, 1)
            for i, cumulative_prob in enumerate(cdf):
                if cumulative_prob >= r:
                    sampled_point = points_sorted[i]
                    particle_prob = probabilities_sorted[i]
            
                    # check if this point is previsouly sampled
                    if len(prev_sampled_particles) > 0:
                        for prev_particle in prev_sampled_particles:
                            if (sampled_point == prev_particle).all():
                                prev_sampled_flag = True
                                break
                            else:
                                prev_sampled_flag = False
                    
                    if not prev_sampled_flag:
                        prev_sampled_particles.append(sampled_point)
                        if (sampled_point[0][2] < 0):
                            mdp.weights = sampled_point
                            vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=100)
                            vi_human.run_vi()
                            if vi_human.stabilized:
                                sampled_model_flag = True
        else:
            print('All particles exhausted. Skipping cluster ...')
            skip_cluster_flag = True
            sampled_model_flag = True
            sampled_point = []
            particle_prob = []
    
    
    return sampled_point, skip_cluster_flag, particle_prob
                    
            


def sample_from_cluster(particles_to_sample, cluster_id, mdp):
    sampled_model = False
    prev_sampled = False
    skip_cluster_flag = False
    sampled_particles = []
    points_to_sample = particles_to_sample.positions[np.where(particles_to_sample.cluster_assignments == cluster_id)[0]]
    points_prob = particles_to_sample.weights[np.where(particles_to_sample.cluster_assignments == cluster_id)[0]]
    
    while not sampled_model:
        print('Sampled particle: ' + str(len(sampled_particles)) + ' out of ' + str(len(points_to_sample)) + ' particles')
        # if len(sampled_particles) < len(points_to_sample):
        if len(sampled_particles) < min(max(0.2*len(points_to_sample), 5), len(points_to_sample)):
            rand_ind = random.randint(0, len(points_to_sample)-1)
            particle = points_to_sample[rand_ind]
            particle_prob = points_prob[rand_ind]
            
            # check if this point is previsouly sampled
            if len(sampled_particles) > 0:
                for sampled_particle in sampled_particles:
                    if (particle == sampled_particle).all():
                        prev_sampled = True
                        break
                    else:
                        prev_sampled = False
            
            if not prev_sampled:
                sampled_particles.append(particle)
                # check for heuristics - negative step cost (this is valid as we start with this prior)
                if (particle[0][2] < 0):
                    mdp.weights = particle
                    vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=100)
                    vi_human.run_vi()
                    if vi_human.stabilized:
                        sampled_model = True

        else:
            skip_cluster_flag = True
            sampled_model = True
            particle = []
            particle_prob = []
    
    
    return particle, skip_cluster_flag, particle_prob



# def evaluate_sampled_human_model(human_model_weight):
    
    
#     sample_human_models_ref_latllong = cg.cart2latlong(np.array(sample_human_models_ref).squeeze())
#     sample_human_models_latlong = cg.cart2latlong(np.array(sample_human_models).squeeze())
#     distances = haversine_distances(sample_human_models_latlong, sample_human_models_ref_latllong)
#     min_model_idxs = np.argmin(distances, axis=1)






def get_human_response(condition, env_idx, particles_to_sample, opt_traj, test_constraints, learning_factor, args = []):

    if len(args) != 0:
        set_id, member_id, test_constraints, sampled_points_history, response_history, member, constraint_history, set_id_history, skip_model_history, cluster_id_history, point_probability, team_learning_factor_history = args


# def get_human_response(env_idx, particles_to_sample, opt_traj, test_constraints, learning_factor):


    filename = 'models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'
    
    with open(filename, 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    mdp = wt_vi_traj_env[0][1].mdp
    agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
    mdp.set_init_state(opt_traj[0][0])
    traj_opt = mdp_helpers.rollout_policy(mdp, agent)

    skip_human_model = True
    skip_cluster = False

    loop_count = 0
    last_skip_cluster_loop = 0
    max_loop_count = 1
    differing_response_model_idx = []
    clusters_to_avoid = []
    points_to_avoid = []

    # particles_to_sample.cluster()  ## Cluster happens here!
    # print('Cluster centers: ', len(particles_to_sample.cluster_centers))
    # print('Cluster weights: ', len(particles_to_sample.cluster_weights))

    while skip_human_model:

        # # ## Method 1: Sampling from clusters based on weights and random sampling of particles within the cluster
        if condition == 'cluster_random':
            human_model_weight, cluster_id, rew_weight_prob = sample_from_distribution(condition, particles_to_sample.cluster_centers, particles_to_sample.cluster_weights)
            
            if loop_count > 0:
                if skip_cluster and len(clusters_to_avoid) < len(particles_to_sample.cluster_centers):
                    print('Sampling from a new cluster ... Max no. of clusters: ', len(particles_to_sample.cluster_centers), '. Clusters to avoid: ', clusters_to_avoid)
                    clusters_to_avoid.append(cluster_id)
                    cluster_sample = []
                    skip_cluster = False
                    human_model_weight, cluster_id, rew_weight_prob = sample_from_distribution(condition, particles_to_sample.cluster_centers, particles_to_sample.cluster_weights, points_to_avoid = clusters_to_avoid)
                else:
                    print('Sampling from the same cluster ...')
                    human_model_weight, skip_cluster, rew_weight_prob = sample_from_cluster(particles_to_sample, cluster_id, mdp)
                    if loop_count - last_skip_cluster_loop > max_loop_count:
                        skip_cluster = True
                        last_skip_cluster_loop = loop_count

                    if len(clusters_to_avoid) >= len(particles_to_sample.cluster_centers):
                        human_model_weight = []
                        cluster_id = []
                        rew_weight_prob = []
                        skip_human_model = False

    #######################

        # # Method 2: Sampling from all particles
        
        elif condition == 'particles':
            sampled_point_flag = False
            while not sampled_point_flag:
                human_model_weight, cluster_id, rew_weight_prob = sample_from_distribution(condition, particles_to_sample.positions, particles_to_sample.weights, points_to_avoid = points_to_avoid)
                
                print('Sampled point: ', human_model_weight, 'Sampled point probability: ', rew_weight_prob, 'Cluster id: ', cluster_id)
                if len(points_to_avoid) > 0:
                    for point_avd in points_to_avoid:
                        if not (human_model_weight == point_avd).all():
                            sampled_point_flag = True
                            print('Point sampled!')
                            break
                else:
                    sampled_point_flag = True
                    print('Point sampled!')

    ############################

        # Method 3: Sampling from clusters and weight-based sampling of particles within the cluster
        elif condition == 'cluster_weight':
            human_model_weight, cluster_id, rew_weight_prob = sample_from_distribution(condition, particles_to_sample.cluster_centers, particles_to_sample.cluster_weights)
            clusters_to_avoid.append(cluster_id)

            if loop_count > 0:
                if skip_cluster and len(clusters_to_avoid) < len(particles_to_sample.cluster_centers):
                    print('Sampling from a new cluster ... Max no. of clusters: ', len(particles_to_sample.cluster_centers), '. Clusters to avoid: ', clusters_to_avoid)
                    skip_cluster = False
                    human_model_weight, cluster_id, rew_weight_prob = sample_from_distribution(condition, particles_to_sample.cluster_centers, particles_to_sample.cluster_weights, points_to_avoid = clusters_to_avoid)
                    clusters_to_avoid.append(cluster_id)
                else:
                    skip_human_model = False
                    print('Sampling from the same cluster ...')
                    human_model_weight, skip_cluster, rew_weight_prob = sample_from_cluster_prob(particles_to_sample, cluster_id, mdp)
                    
                    print('last_skip_cluster_loop: ', last_skip_cluster_loop, 'loop_count: ', loop_count, 'max_loop_count: ', max_loop_count)
                    if loop_count - last_skip_cluster_loop > max_loop_count:
                        skip_cluster = True
                        last_skip_cluster_loop = loop_count
                    if len(clusters_to_avoid) >= len(particles_to_sample.cluster_centers):
                        human_model_weight = []
                        cluster_id = []
                        rew_weight_prob = []
                        skip_human_model = False

                    
    #######################
        # if params.debug_hm_sampling:
        # print('Sampled models: ', cust_samps)

        # for model_idx, human_model_weight in enumerate(cust_samps):
        # print('human_model_weight: ', human_model_weight, 'cluster_id: ', cluster_id)
        # print('human_model_weight: ', human_model_weight)
        
        if len(human_model_weight) != 0:
        # if (human_model_weight != [0, 0, 0]):
            points_to_avoid.append(human_model_weight)

            mdp.weights = human_model_weight
            # vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=100)
            vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=100)
            vi_human.run_vi()

            if not vi_human.stabilized:
                print(colored('Human model with weight, ' + str(human_model_weight) + ', did not converge and skipping for response generation', 'red'))
                skip_human_model = True
            else:
                print(colored('Human model with weight, ' + str(human_model_weight) + ', converged', 'green'))
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


                # mdp.visualize_trajectory(traj_opt)
                # plt.show()
                # mdp.visualize_trajectory(human_opt_trajs[0])
                # plt.show()


            else:
                response_type = 'NA'
                # visualize skipped trajectories
                cur_state = mdp.get_init_state()
                human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
                # mdp.visualize_trajectory(human_opt_trajs[0])
                # plt.show()
            
            # check for constraint satisfaction
            if len(args) != 0:
                constraint_flag = True
                for constraint in test_constraints[0]:
                    if constraint.dot(human_model_weight.T) < 0:
                        constraint_flag = False
            else:
                constraint_flag = True
                for constraint in test_constraints:
                    if constraint.dot(human_model_weight.T) < 0:
                        constraint_flag = False
            
            if len(args) != 0:
                sampled_points_history.append(human_model_weight)
                response_history.append(response_type)
                member.append(member_id)
                constraint_history.append(constraint_flag)
                set_id_history.append(set_id)
                cluster_id_history.append(cluster_id)
                skip_model_history.append(skip_human_model)
                point_probability.append(rew_weight_prob)
                team_learning_factor_history.append(learning_factor)

            loop_count += 1
        
        else:
            human_opt_trajs = [None]
            response_type = 'NA'
            break

    # human_traj_overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(human_opt_trajs[0], traj_opt)
    # print('human traj overlap pct: ', human_traj_overlap_pct)

    # return human_opt_trajs, response_type, likely_correct_response_count, likely_incorrect_response_count, initial_likely_response_type

    # plot cluster centers and sampled particle
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # particles_to_sample.plot(fig=fig, ax=ax, cluster_centers = particles_to_sample.cluster_centers, cluster_weights = particles_to_sample.cluster_weights)
    # ax.scatter(human_model_weight[0][0], human_model_weight[0][1], human_model_weight[0][2], c='g', marker='o', s = 100)
    # BEC_viz.visualize_planes(test_constraints, fig=fig, ax=ax, alpha=0.5)

    # plt.show()



    if len(args) != 0:
        return human_model_weight, human_opt_trajs[0], response_type, sampled_points_history, response_history, member, constraint_history, set_id_history, skip_model_history, cluster_id_history, point_probability, team_learning_factor_history
    else:
        return human_opt_trajs[0], response_type



# def get_human_response_old(env_idx, env_cnst, opt_traj, likelihood_correct_response):

#     if params.debug_hm_sampling:
#         print('Generating human response for environment constraint: ', env_cnst[0])

#     human_model = hlm.cust_pdf_uniform(env_cnst[0], likelihood_correct_response)

#     # print('human model kappa: ', human_model.kappa)

#     filename = 'models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'
    
#     with open(filename, 'rb') as f:
#         wt_vi_traj_env = pickle.load(f)

#     mdp = wt_vi_traj_env[0][1].mdp

#     agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
#     mdp.set_init_state(opt_traj[0][0])
#     traj_opt = mdp_helpers.rollout_policy(mdp, agent)

#     # mdp.visualize_state(mdp.get_init_state())
#     # plt.show()

    
#     # print(traj_opt)
#     # print(opt_traj)

#     # opt_overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(traj_opt, opt_traj)
#     # print('Optimal trajectory overlap percentage: ', opt_overlap_pct)
    
#     likely_correct_response_count = 0
#     likely_incorrect_response_count = 0

#     skip_human_model = True

#     loop_count = 0
#     max_loop_count = 100
#     differing_response_model_idx = []

#     while skip_human_model:
#         if params.debug_hm_sampling:
#             print('Sampling a human model ...')
#         cust_samps = human_model.rvs(size=1) # sample a human model (weight vector) from the distribution created from the constraint and the likelihood of sampling a correct response for that constraint
#         if params.debug_hm_sampling:
#             print('Sampled models: ', cust_samps)

#         dot = cust_samps[0].dot(env_cnst[0])
#         if dot >= 0:
#             # print('Possibly a correct response sampled')
#             likely_correct_response_count += 1
#             likely_response_type = 'correct'
#         else:
#             # print('Possibly an incorrect response sampled')
#             likely_incorrect_response_count += 1
#             likely_response_type = 'incorrect'

#         if loop_count == 0:
#             initial_likely_response_type = likely_response_type


#         for model_idx, human_model_weight in enumerate(cust_samps):

#             # print('Sampled model: ', human_model_weight)

#             mdp.weights = human_model_weight
#             # vi_human = ValueIteration(mdp, sample_rate=1, max_iterations=100)
#             vi_human = ValueIteration(mdp, sample_rate=1)
#             vi_human.run_vi()

#             if not vi_human.stabilized:
#                 if params.debug_hm_sampling:
#                     print(colored('Human model ' + str(model_idx) + ' did not converge and skipping for response generation', 'red'))
#                 skip_human_model = True
#             else:
#                 if params.debug_hm_sampling:
#                     print(colored('Human model ' + str(model_idx) + ' converged', 'green'))
#                 skip_human_model = False
            
#             if not skip_human_model:
#                 cur_state = mdp.get_init_state()
#                 # print('Current state: ', cur_state)
#                 human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
                
#                 human_traj_rewards = mdp.accumulate_reward_features(human_opt_trajs[0], discount=True)  # just use the first optimal trajectory
#                 mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
#                 new_constraint = mu_sa - human_traj_rewards

#                 # print('New constraint: ', new_constraint)
#                 if (new_constraint == np.array([0, 0, 0])).all():
#                     if params.debug_hm_sampling:
#                         print(colored('Correct response sampled', 'blue'))
#                     response_type = 'correct'
#                 else:
#                     if params.debug_hm_sampling:
#                         print(colored('Incorrect response sampled', 'red'))
#                     response_type = 'incorrect'

#                 # check if the response matches the initial likely guess
#                 if response_type != initial_likely_response_type and loop_count < max_loop_count:
#                     skip_human_model = True
#                     differing_response_model_idx.append([model_idx, human_model_weight])
#                     if params.debug_hm_sampling:
#                         print(colored('Different from initial repsonse. Skipping ...', 'red'))
#                         print('Initial likely response type: ', initial_likely_response_type, 'Current response type: ', response_type)
#                 elif response_type != initial_likely_response_type and loop_count >= max_loop_count:
#                     alt_human_model_weight = differing_response_model_idx[0][1] # choose a previously converged model that differs from the initial likely response
#                     mdp.weights = alt_human_model_weight
#                     vi_human = ValueIteration(mdp, sample_rate=1)
#                     vi_human.run_vi()
#                     human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
#                     if params.debug_hm_sampling:
#                         print(colored('Human model chose from differing repsonse. ', 'green'))
                
                
#                 # mdp.visualize_trajectory(traj_opt)
#                 # plt.show()
#                 # mdp.visualize_trajectory(human_opt_trajs[0])
#                 # plt.show()

#         loop_count += 1

#     # human_traj_overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(human_opt_trajs[0], traj_opt)
#     # print('human traj overlap pct: ', human_traj_overlap_pct)

#     # return human_opt_trajs, response_type, likely_correct_response_count, likely_incorrect_response_count, initial_likely_response_type

#     return human_opt_trajs[0], response_type





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
