#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np
import copy
from termcolor import colored
from pathos.multiprocessing import ProcessPool as Pool
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
import difflib

# Other imports.
sys.path.append("simple_rl")
import params
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import make_mdp
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers
from policy_summarization import BEC
import policy_summarization.multiprocessing_helpers as mp_helpers
from simple_rl.utils import mdp_helpers
import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz
from policy_summarization import particle_filter as pf
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'


import pdb

def generate_agent(mdp_class, data_loc, mdp_parameters, visualize=False):
    try:
        with open('models/' + data_loc + '/vi_agent.pickle', 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        cf_data_dir = 'models/' + data_loc
        os.makedirs(cf_data_dir, exist_ok=True)

        mdp_agent = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)
        vi_agent = ValueIteration(mdp_agent, sample_rate=1)
        vi_agent.run_vi()

        with open('models/' + data_loc + '/vi_agent.pickle', 'wb') as f:
            pickle.dump((mdp_agent, vi_agent), f)

    # Visualize agent
    if visualize:
        fixed_agent = FixedPolicyAgent(vi_agent.policy)
        mdp_agent.visualize_agent(fixed_agent)
        mdp_agent.reset()  # reset the current state to the initial state
        mdp_agent.visualize_interaction()

def obtain_summary(mdp_class, data_loc, mdp_parameters, weights, step_cost_flag, summary_variant, pool, n_train_demos, n_human_models, n_particles, prior, posterior, obj_func_proportion, hardcode_envs=False):
    if hardcode_envs:
        # using 4 hardcoded environments
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool, hardcode_envs=True)

        vi_traj_triplets = []
        for i in range(4):
            env_filename = mp_helpers.lookup_env_filename(data_loc, i)

            with open(env_filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)

            mdp = wt_vi_traj_env[0][1].mdp
            agent = wt_vi_traj_env[0][1]
            weights = mdp.weights
            trajectory = mdp_helpers.rollout_policy(mdp, agent)

            vi_traj_triplets.append((i, agent, trajectory))
    else:
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool)

    try:
        with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count = pickle.load(f)
    except:
        if hardcode_envs:
            # use demo BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count = BEC.extract_constraints(data_loc, step_cost_flag, pool, vi_traj_triplets=vi_traj_triplets, print_flag=True)
        else:
            # use policy BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count = BEC.extract_constraints(data_loc, step_cost_flag, pool, print_flag=True)
        with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
            pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count), f)

    try:
        with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
            min_BEC_constraints, BEC_lengths_record = pickle.load(f)
    except:
        min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(policy_constraints, min_subset_constraints_record, env_record, weights, step_cost_flag, pool)

        with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
            pickle.dump((min_BEC_constraints, BEC_lengths_record), f)

    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary, visited_env_traj_idxs, particles = pickle.load(f)
    except:
        # SCOT_summary = BEC.obtain_SCOT_summaries(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag)
        # initialize particle filter
        particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
        particles = pf.Particles(particle_positions)
        particles.update(prior)
        print(colored('entropy: {}'.format(particles.calc_entropy()), 'blue'))

        if summary_variant == 'particle_filter':
            BEC_summary, visited_env_traj_idxs, particles = BEC.obtain_summary_particle_filter(data_loc, particles, summary_variant, min_subset_constraints_record,
                                           min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool,
                                           n_human_models, consistent_state_count)

        elif summary_variant == 'proposed' or summary_variant == 'counterfactual_only':
            BEC_summary, visited_env_traj_idxs = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count, n_train_demos=n_train_demos, prior=prior, obj_func_proportion=obj_func_proportion)
        elif summary_variant == 'feature_only' or summary_variant == 'baseline':
            BEC_summary = BEC.obtain_summary(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=n_train_demos)
        else:
            raise AssertionError("Unknown summary variant.")

        if summary_variant != 'particle_filter':
            # update the particle filter model according to the generated summary
            for unit in BEC_summary:
                for summary in unit:
                    particles.update(summary[3])

        if len(BEC_summary) > 0:
            with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
                pickle.dump((BEC_summary, visited_env_traj_idxs, particles), f)

    # BEC.visualize_summary(BEC_summary, weights, step_cost_flag)
    #
    # for summary in BEC_summary:
    #     best_mdp = summary[0]
    #     best_traj = summary[1]
    #
    #     with open('models/augmented_taxi/info_gains_' + str(0) + '. pickle', 'rb') as f:
    #         info_gains_record = pickle.load(f)
    #
    #     with open('models/' + data_loc + '/counterfactual_data_' + str(0) + '/model' + str(
    #             select_model) + '/cf_data_env' + str(
    #             best_env_idx).zfill(5) + '.pickle', 'rb') as f:
    #         best_human_trajs_record_env, constraints_env = pickle.load(f)
    #

    # # constraint visualization
    # BEC_summary = list(itertools.chain(*BEC_summary))
    # constraints_record = prior
    # for summary in BEC_summary:
    #     constraints_record.extend(summary[3])
    #     # constraints_record = summary[3]
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.set_facecolor('white')
    #     ax.xaxis.pane.fill = False
    #     ax.yaxis.pane.fill = False
    #     ax.zaxis.pane.fill = False
    #
    #     solid_angle = BEC_helpers.calc_solid_angles([constraints_record])[0]
    #     print(solid_angle)
    #
    #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_record)
    #     poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    #
    #     min_constraints = BEC_helpers.remove_redundant_constraints(constraints_record, weights, step_cost_flag)
    #     print(min_constraints)
    #     for constraints in [min_constraints]:
    #         BEC_viz.visualize_planes(constraints, fig=fig, ax=ax)
    #
    #     # visualizing uninformed prior
    #     # ieqs2 = BEC_helpers.constraints_to_halfspace_matrix_sage([[]])
    #     # poly2 = Polyhedron.Polyhedron(ieqs=ieqs2)
    #     # BEC_viz.visualize_spherical_polygon(poly2, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #     #
    #     # visualize spherical polygon
    #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #
    #     # ieqs_posterior = BEC_helpers.constraints_to_halfspace_matrix_sage(posterior)
    #     # poly_posterior = Polyhedron.Polyhedron(ieqs=ieqs_posterior)  # automatically finds the minimal H-representation
    #     # BEC_viz.visualize_spherical_polygon(poly_posterior, fig=fig, ax=ax, plot_ref_sphere=False, color='g')
    #
    #     ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100)
    #     if mdp_class == 'augmented_taxi2':
    #         ax.set_xlabel('$\mathregular{w_0}$: Mud')
    #         ax.set_ylabel('$\mathregular{w_1}$: Recharge')
    #     elif mdp_class == 'two_goal2':
    #         ax.set_xlabel('X: Goal 1 (grey)')
    #         ax.set_ylabel('Y: Goal 2 (green)')
    #     else:
    #         ax.set_xlabel('X: Goal')
    #         ax.set_ylabel('Y: Skateboard')
    #     ax.set_zlabel('$\mathregular{w_2}$: Action')
    #
    #     ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    #     ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    #     ax.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    #
    #     # if matplotlib.get_backend() == 'TkAgg':
    #     #     mng = plt.get_current_fig_manager()
    #     #     mng.resize(*mng.window.maxsize())
    #
    #     plt.show()

    # # particle filter visualization
    # from numpy.random import seed
    # seed(2)
    #
    # particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    # particles = pf.Particles(particle_positions)
    #
    # constraints_running = prior
    #
    # # print(particles.calc_info_gain(prior))
    # particles.update(prior)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_facecolor('white')
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #
    # particles.plot(fig=fig, ax=ax)
    # BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
    #
    # # visualize spherical polygon
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #
    # # visualize the ground truth weight
    # w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    # w_normalized = w / np.linalg.norm(w[0, :], ord=2)
    # ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='r', s=100)
    #
    # plt.show()
    #
    # for j, summary in enumerate(BEC_summary):
    #     print(j)
    #
    #     constraints = summary[3]
    #
    #     constraints_running.extend(constraints)
    #     constraints_running = BEC_helpers.remove_redundant_constraints(constraints_running, None, False)
    #
    #     particles.update(constraints)
    #     print('Entropy: {}'.format(particles.calc_entropy()))
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.set_facecolor('white')
    #     ax.xaxis.pane.fill = False
    #     ax.yaxis.pane.fill = False
    #     ax.zaxis.pane.fill = False
    #
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #
    #     particles.plot(fig=fig, ax=ax)
    #     BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
    #
    #     # visualize spherical polygon
    #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
    #     poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #
    #     # visualize the ground truth weight
    #     w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    #     w_normalized = w / np.linalg.norm(w[0, :], ord=2)
    #     ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='b', s=100)
    #
    #     plt.show()

    return BEC_summary, visited_env_traj_idxs, particles

def obtain_test_environments(mdp_class, data_loc, mdp_parameters, weights, BEC_params, step_cost_flag, n_human_models, prior, posterior, summary=None, use_counterfactual=True, visualize_test_env=False):
    '''
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    '''
    # use generalized area of the BEC region to select test environments
    try:
        with open('models/' + data_loc + '/test_environments.pickle', 'rb') as f:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(f)
    except:
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool)

        try:
            with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
                policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = pickle.load(f)
        except:
            # use policy BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = BEC.extract_constraints(
                data_loc, step_cost_flag, pool, print_flag=True)
            with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
                pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record,
                             consistent_state_count), f)

        try:
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
                min_BEC_constraints, BEC_lengths_record = pickle.load(f)
        except:
            min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(policy_constraints,
                                                                                  min_subset_constraints_record,
                                                                                  env_record, weights, step_cost_flag,
                                                                                  pool)

            with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
                pickle.dump((min_BEC_constraints, BEC_lengths_record), f)

        equal_prior_posterior = True
        counterfactual_folder_idx = 0
        if len(prior) == len(posterior):
            for j, posterior_constraint in enumerate(posterior):
                if not np.array_equal(posterior_constraint, prior[j]):
                    equal_prior_posterior = False
                    counterfactual_folder_idx = -1
        else:
            equal_prior_posterior = False
            counterfactual_folder_idx = -1

        try:
            with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'rb') as f:
                min_subset_constraints_record_counterfactual = pickle.load(f)

            with open('models/' + data_loc + '/BEC_lengths_counterfactual.pickle', 'rb') as f:
                BEC_lengths_record_counterfactual = pickle.load(f)

        except:
            if not equal_prior_posterior:
                sample_human_models = BEC_helpers.sample_human_models_uniform(posterior, n_human_models)
                print("Obtaining counterfactual data for human models sampled from the posterior: ")
                for model_idx, human_model in enumerate(sample_human_models):
                    cf_data_dir = 'models/' + data_loc + '/counterfactual_data_-1/model' + str(model_idx)
                    os.makedirs(cf_data_dir, exist_ok=True)

                    print(colored('Model #: {}'.format(model_idx), 'red'))
                    print(colored('Model val: {}'.format(human_model), 'red'))

                    # assuming that I'm considering human models jointly
                    pool.restart()
                    n_processed_envs = len(os.listdir(cf_data_dir))
                    args = [
                        (data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]),
                         traj_record[i], posterior, step_cost_flag, counterfactual_folder_idx, None,
                         True) for i in range(n_processed_envs, len(traj_record))]
                    _ = list(tqdm(pool.imap(BEC.compute_counterfactuals, args), total=len(args)))
                    pool.close()
                    pool.join()
                    pool.terminate()

            try:
                with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'rb') as f:
                    min_subset_constraints_record_counterfactual = pickle.load(f)
            except:
                pool.restart()
                args = [(data_loc, i, min_subset_constraints_record[i], n_human_models, counterfactual_folder_idx, weights, step_cost_flag) for i in range(len(min_subset_constraints_record))]
                # combine the human counterfactual and one-step deviation constraints
                min_subset_constraints_record_counterfactual = list(tqdm(pool.imap(BEC_helpers.combine_counterfactual_constraints, args), total=len(args)))
                pool.close()
                pool.join()
                pool.terminate()

                with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'wb') as f:
                    pickle.dump(min_subset_constraints_record_counterfactual, f)

            # take the overlap of the human posterior with BEC of suboptimal trajectories of one-step deviation
            pool.restart()
            args = [(i, n_human_models, min_subset_constraints, prior, posterior, data_loc, counterfactual_folder_idx, weights, traj_record[i], step_cost_flag, pool)
                    for i, min_subset_constraints in enumerate(min_subset_constraints_record_counterfactual)]

            print("Obtaining overlap in BEC area between posterior human model and potential test demonstrations: ")
            BEC_lengths_record_counterfactual = list(tqdm(pool.imap(BEC.overlap_demo_BEC_and_human_posterior, args), total=len(args)))
            pool.close()
            pool.join()
            pool.terminate()
            with open('models/' + data_loc + '/BEC_lengths_counterfactual.pickle', 'wb') as f:
                pickle.dump(BEC_lengths_record_counterfactual, f)

        if use_counterfactual:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = \
                ps_helpers.obtain_test_environments(data_loc, min_subset_constraints_record_counterfactual, env_record, traj_record, weights, BEC_lengths_record_counterfactual, BEC_params['n_test_demos'], BEC_params['test_difficulty'], step_cost_flag, summary=summary)
        else:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = \
                ps_helpers.obtain_test_environments(data_loc, min_subset_constraints_record, env_record,
                                                    traj_record, weights, BEC_lengths_record,
                                                    BEC_params['n_test_demos'], BEC_params['test_difficulty'],
                                                    step_cost_flag, summary=summary)

        with open('models/' + data_loc + '/test_environments.pickle', 'wb') as f:
            pickle.dump((test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers), f)

    if visualize_test_env:
        BEC.visualize_test_envs(posterior, test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers, weights,
                                step_cost_flag)
    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints

def calculate_new_pos (new_cop, pointer, step_size, currx, curry, currstate, counter, last):
    if counter == 0:
        (nextx, nexty, nextstate) = new_cop[pointer + 1]
    elif counter == 1:
        (nextx, nexty, nextstate) = new_cop[-1]

    else:
        (nextx, nexty, nextstate) = (last.get_agent_x(), last.get_agent_y(), last)
    
    #going left
    if nextx < currx:    
        action = "left"
        (newx, newy) = (currx - step_size, curry)
        #need to make changes if wrapping around
        if (newx < nextx):
            diff = nextx - newx
            #this should only happen on not last part of this segment
            if (counter == 0):
                (twox, twoy, twostate) = new_cop[pointer + 2]
            elif (counter == 1):
                (twox, twoy, twostate) = (last.get_agent_x(), last.get_agent_y(), last)
            new_direction = (twox - nextx, twoy - nexty) 
            if twox > nextx: 
                newx = nextx + diff
            elif twoy > nexty:
                newx = nextx
                newy = nexty + diff
            elif twoy < nexty:
                newx = nextx
                newy = nexty - diff
            pointer += 1
    #going right
    elif currx < nextx:  
        action = "right"
        (newx, newy) = (currx + step_size, curry)
        #need to make changes if wrapping around
        if (newx > nextx):
            diff = newx - nextx
            #this should only happen on not last part of this segment
            if (counter == 0):
                (twox, twoy, twostate) = new_cop[pointer + 2]
            elif (counter == 1):
                (twox, twoy, twostate) = (last.get_agent_x(), last.get_agent_y(), last)
            new_direction = (twox - nextx, twoy - nexty) 
            if twox < nextx:
                newx = nextx - diff
            elif twoy > nexty: 
                newx = nextx
                newy = nexty + diff
            elif twoy < nexty:
                newx = nextx
                newy = nexty - diff
            pointer += 1
    #going up
    elif nexty > curry: 
        action = "up"
        (newx, newy) = (currx, curry + step_size)
        #need to make changes if wrapping around
        if (newy > nexty):
            diff = newy - nexty
            #this should only happen on not last part of this segment
            if (counter == 0):
                (twox, twoy, twostate) = new_cop[pointer + 2]
            elif (counter == 1):
                (twox, twoy, twostate) = (last.get_agent_x(), last.get_agent_y(), last)
            new_direction = (twox - nextx, twoy - nexty) 
            if twox < nextx: 
                newy = nexty
                newx = nextx - diff
            elif twox > nextx:
                newy = nexty
                newx = nextx + diff
            elif twoy < nexty: 
                newy = nexty - diff
            pointer += 1
    #going down
    elif curry > nexty:  
        action = "down"
        (newx, newy) = (currx, curry - step_size)
        #need to make changes if wrapping around
        if (newy < nexty):
            diff = nexty - newy
            #this should only happen on not last part of this segment
            if (counter == 0):
                (twox, twoy, twostate) = new_cop[pointer + 2]
            elif (counter == 1):
                (twox, twoy, twostate) = (last.get_agent_x(), last.get_agent_y(), last)
            new_direction = (twox - nextx, twoy - nexty) 
            if twox < nextx: 
                newy = nexty
                newx = nextx - diff
            elif twox > nextx:
                newy = nexty
                newx = nextx + diff
            elif twoy > nexty: 
                newy = nexty + diff
            pointer += 1
    else:
        return
    editedstate = copy.deepcopy(currstate)
    editedstate.objects["agent"][0]["x"] = newx
    editedstate.objects["agent"][0]["y"] = newy

    return newx, newy, editedstate, action



def normalize_trajs(opt_traj, human_traj):
    opt_traj_currs = [currstate for (prevstate, action, currstate) in opt_traj]
    human_traj_currs = [currstate for (prevstate, action, currstate) in human_traj]
    matcher = difflib.SequenceMatcher(None, opt_traj_currs, human_traj_currs, autojunk=False)
    matches = matcher.get_matching_blocks()
    anchor_points = []

    for match in matches:
        for i in range(match[2]):
            anchor_points.append((match[0] + i, match[1] + i))
    
    for pt in anchor_points:
        (opt, hum) = pt
            
    normalized_opt_traj = []
    normalized_human_traj = []

    for i in range(len(anchor_points)):
        (opt_idx, human_idx) = anchor_points[i]

        if i == 0:
            prev_opt_idx = 0
            prev_human_idx = 0
        else:
            (prev_opt_idx, prev_human_idx) = anchor_points[i - 1]
             
        diff_opt = opt_idx - prev_opt_idx + 1
        diff_human = human_idx - prev_human_idx + 1

        #all good no normalization needed for this segment
        if (diff_opt == diff_human):
            normalized_human_traj = normalized_human_traj + [human_traj[l] for l in range (prev_human_idx, human_idx)]
            normalized_opt_traj = normalized_opt_traj + [opt_traj[l] for l in range (prev_opt_idx, opt_idx)]
            continue

        else: 
            opt_start_state = opt_traj[prev_opt_idx][0]
            human_start_state = human_traj[prev_human_idx][0]
            new_cop = []  #creating trajectory in terms of (x, y) coordinates
            holder = []
            pointer = 0
            #need to expand optimal trajectory
            if (diff_opt < diff_human):
                step_size = diff_opt/diff_human

                opt_segment = [] #going to hold normalized segment in terms of states
                for j in range(prev_opt_idx, opt_idx + 1):
                    new_cop.append((opt_traj[j][0].get_agent_x(), opt_traj[j][0].get_agent_y(), opt_traj[j][0]))

                counter = 0
                last = opt_traj[opt_idx][2]
                for k in range(len(new_cop) + 1):
                    if k == 0:
                        (currx, curry, currstate) = new_cop[k]
                    else:
                        (currx, curry, currstate) = holder[-1]
                    if (k == len(new_cop) - 1):
                        counter = 1
                    elif (k == len(new_cop)):
                        counter = 2

                    newx, newy, editedstate, action = calculate_new_pos(new_cop, pointer, step_size, currx, curry, currstate, counter, last)
                    opt_segment.append((currstate, action, editedstate))
                    holder.append((newx, newy, editedstate))
                
                normalized_opt_traj = normalized_opt_traj + opt_segment
                normalized_human_traj = normalized_human_traj + [human_traj[l] for l in range (prev_human_idx, human_idx)]
                
            #need to expand human trajectory
            else:
                step_size = diff_human/diff_opt

                human_segment = [] #going to hold normalized segment in terms of states
                for j in range(prev_human_idx, human_idx + 1):
                    new_cop.append((human_traj[j][0].get_agent_x(), human_traj[j][0].get_agent_y(), human_traj[j][0]))

                counter = 0
                last = human_traj[human_idx][2]
                for k in range(len(new_cop) + 1):
                    if k == 0:
                        (currx, curry, currstate) = new_cop[k]
                    else:
                        (currx, curry, currstate) = holder[-1]
                    if (k == len(new_cop) - 1):
                        counter = 1
                    elif (k == len(new_cop)):
                        counter = 2
                    newx, newy, editedstate, action = calculate_new_pos(new_cop, pointer, step_size, currx, curry, currstate, counter, last)
                    human_segment.append((currstate, action, editedstate))
                    holder.append((newx, newy, editedstate))


                normalized_human_traj = normalized_human_traj + human_segment
                normalized_opt_traj = normalized_opt_traj + [opt_traj[l] for l in range (prev_opt_idx, opt_idx)]

    normalized_human_traj.append(human_traj[-1])
    normalized_opt_traj.append(opt_traj[-1])

    return normalized_opt_traj, normalized_human_traj                
                    

        




def obtain_unit_tests(mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, prior, n_particles, n_human_models, data_loc, weights, step_cost_flag, visualize_pf_transition=False):
    # todo: maybe pass in some of these objects later
    with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count = pickle.load(
            f)

    # initialize a particle filter model of human
    particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    particles = pf.Particles(particle_positions)
    particles.update(prior)
    particles_prev = copy.deepcopy(particles)

    # run through the pre-selected units
    for unit_idx, unit in enumerate(BEC_summary):
        print("Here are the demonstrations for this unit")
        unit_constraints = []
        running_variable_filter = unit[0][4]

        # show each demonstration that is part of this unit
        for subunit in unit:
            subunit[0].visualize_trajectory(subunit[1])
            unit_constraints.extend(subunit[3])

            # update particle filter with demonstration's constraint
            particles.update(subunit[3])
            # visualize the updated particle filter
            if visualize_pf_transition:
                BEC_viz.visualize_pf_transition(subunit[3], particles_prev, particles, mdp_class, weights)
                particles_prev = copy.deepcopy(particles)

        # obtain the constraints conveyed by the unit's demonstrations
        min_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, weights, step_cost_flag)
        # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
        preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(data_loc, unit, visited_env_traj_idxs, min_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter)

        # with open('models/' + data_loc + '/preliminary_tests.pickle', 'wb') as f:
        #     pickle.dump((preliminary_tests, visited_env_traj_idxs), f)
        # with open('models/' + data_loc + '/preliminary_tests.pickle', 'rb') as f:
        #     preliminary_tests, visited_env_traj_idxs = pickle.load(f)

        # query the human's response to the diagnostic tests
        for test in preliminary_tests:
            test_mdp = test[0]
            opt_traj = test[1]
            test_constraints = test[3]
            test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

            print("Here is a diagnostic test for this unit")
            human_traj, human_history = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent
            # with open('models/' + data_loc + '/human_traj.pickle', 'wb') as f:
            #     pickle.dump((human_traj, human_history), f)
            # with open('models/' + data_loc + '/human_traj.pickle', 'rb') as f:
            #     human_traj, human_history = pickle.load(f)

            human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
            opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

            if (human_feature_count == opt_feature_count).all():
                print("You got the diagnostic test right")

                particles.update(test_constraints)
                if visualize_pf_transition:
                    BEC_viz.visualize_pf_transition(test_constraints, particles_prev, particles, mdp_class, weights)
                    particles_prev = copy.deepcopy(particles)

            else:
                print("You got the diagnostic test wrong. Here's the correct answer")
                failed_BEC_constraint = opt_feature_count - human_feature_count
                print("Failed BEC constraint: {}".format(failed_BEC_constraint))

                particles.update([-failed_BEC_constraint])
                if visualize_pf_transition:
                    BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles_prev, particles, mdp_class, weights)
                    particles_prev = copy.deepcopy(particles)

                normalized_opt_traj, normalized_human_traj = normalize_trajs(opt_traj, human_traj)
                test_mdp.visualize_trajectory_comparison(normalized_opt_traj, normalized_human_traj)

                print("Here is a remedial demonstration that might be helpful")

                remedial_instruction, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool, particles, n_human_models, failed_BEC_constraint, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter, consistent_state_count, step_cost_flag)
                remedial_mdp, remedial_traj, _, remedial_constraint, _ = remedial_instruction[0]
                remedial_mdp.visualize_trajectory(remedial_traj)
                test_history.extend(remedial_instruction)

                particles.update([remedial_constraint])
                if visualize_pf_transition:
                    BEC_viz.visualize_pf_transition([remedial_constraint], particles_prev, particles, mdp_class, weights)
                    particles_prev = copy.deepcopy(particles)

                with open('models/' + data_loc + '/remedial_instruction.pickle', 'wb') as f:
                    pickle.dump(remedial_instruction, f)

                remedial_test_correct = False

                print("Here is a remedial test to see if you've correctly learned the lesson")
                while not remedial_test_correct:

                    remedial_test, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool,
                                                                                                     particles,
                                                                                                     n_human_models,
                                                                                                     failed_BEC_constraint,
                                                                                                     min_subset_constraints_record,
                                                                                                     env_record,
                                                                                                     traj_record,
                                                                                                     traj_features_record,
                                                                                                     test_history,
                                                                                                     visited_env_traj_idxs,
                                                                                                     running_variable_filter,
                                                                                                     consistent_state_count,
                                                                                                     step_cost_flag, type='testing')

                    remedial_mdp, remedial_traj, _, _, _ = remedial_test[0]
                    test_history.extend(remedial_test)

                    human_traj, human_history = remedial_mdp.visualize_interaction(
                        keys_map=params.keys_map)  # the latter is simply the gridworld locations of the agent
                    # with open('models/' + data_loc + '/human_traj.pickle', 'wb') as f:
                    #     pickle.dump((human_traj, human_history), f)
                    # with open('models/' + data_loc + '/human_traj.pickle', 'rb') as f:
                    #     human_traj, human_history = pickle.load(f)

                    human_feature_count = remedial_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = remedial_mdp.accumulate_reward_features(remedial_traj, discount=True)

                    if (human_feature_count == opt_feature_count).all():
                        print("You got the remedial test correct")
                        remedial_test_correct = True

                        particles.update([failed_BEC_constraint])
                        if visualize_pf_transition:
                            BEC_viz.visualize_pf_transition([failed_BEC_constraint], particles_prev, particles, mdp_class,
                                                            weights)
                            particles_prev = copy.deepcopy(particles)
                    else:
                        failed_remedial_constraint = opt_feature_count - human_feature_count
                        print("You got the remedial test wrong. Here's the correct answer")

                        normalized_remedial_traj, normalized_human_traj = normalize_trajs(remedial_traj, human_traj)
                        remedial_mdp.visualize_trajectory_comparison(normalized_remedial_traj, normalized_human_traj)

                        particles.update([-failed_remedial_constraint])
                        if visualize_pf_transition:
                            BEC_viz.visualize_pf_transition([-failed_remedial_constraint], particles_prev, particles, mdp_class,
                                                            weights)
                            particles_prev = copy.deepcopy(particles)


    return preliminary_tests, visited_env_traj_idxs

if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

    with open('models/' + params.data_loc['BEC'] + '/params.pickle', 'wb') as f:
        pickle.dump(params, f)

    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent(params.mdp_class, params.data_loc['base'], params.mdp_parameters, visualize=True)

    # b) obtain a BEC summary of the agent's policy
    BEC_summary, visited_env_traj_idxs, particles_summary = obtain_summary(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'],
                            params.step_cost_flag, params.BEC['summary_variant'], pool, params.BEC['n_train_demos'],
                            params.BEC['n_human_models'], params.BEC['n_particles'], params.prior, params.posterior, params.BEC['obj_func_proportion'])

    unit_tests = obtain_unit_tests(params.mdp_class,BEC_summary, visited_env_traj_idxs, particles_summary, pool, params.prior, params.BEC['n_particles'], params.BEC['n_human_models'], params.data_loc['BEC'], params.weights['val'], params.step_cost_flag)

    # c) obtain test environments
    # obtain_test_environments(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
    #                          params.step_cost_flag, params.BEC['n_human_models'], params.prior, params.posterior, summary=BEC_summary, visualize_test_env=True, use_counterfactual=True)