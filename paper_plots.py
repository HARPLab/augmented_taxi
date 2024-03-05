# Plots for IJCAI paper

import pandas as pd
import ast
import json
import numpy as np
import seaborn as sns
import math
import os
from termcolor import colored
import warnings
import textwrap
import pickle
from ast import literal_eval
import copy
import sage.all
import sage.geometry.polyhedron.base as Polyhedron



import teams.teams_helpers as team_helpers
import params_team as params
import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz
import teams.utils_teams as utils_teams
import policy_summarization.probability_utils as p_utils


import multiprocessing



import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors



plt.rcParams['figure.figsize'] = [15, 10]
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'  # Import additional LaTeX packages if needed
# plt.rcParams['font.family'] = 'serif'


def label_axes(ax, mdp_class, weights=None, view_params = None):
    fs = 16
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if weights is not None:
        ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=30)
    if mdp_class == 'augmented_taxi2':
        ax.set_xlabel(r'$\mathregular{w_0}$: Mud', fontsize = fs, labelpad=15)
        ax.set_ylabel(r'$\mathregular{w_1}$: Recharge', fontsize = fs, labelpad=15)
    elif mdp_class == 'colored_tiles':
        ax.set_xlabel('X: Tile A (brown)')
        ax.set_ylabel('Y: Tile B (green)')
    else:
        ax.set_xlabel('X: Goal')
        ax.set_ylabel('Y: Skateboard')
    ax.set_zlabel('$\mathregular{w_2}$: Action', fontsize = fs, labelpad=7)

    # ax.view_init(elev=16, azim=-160)
    if not view_params is None:
        elev = view_params[0]
        azim = view_params[1]
    else:
        elev=16
        azim = -160
    ax.view_init(elev=elev, azim=azim)

    # Set ticks for x, y, and z axes
    # ax.set_xticks(np.arange(-1, 1, 0.5))  # Specify x ticks
    # ax.set_yticks(np.arange(-1, 1, 0.5))  # Specify y ticks
    # ax.set_zticks(np.arange(-1, 1, 0.5))  # Specify z ticks

    ax.set_xticks(np.arange(-1, 1, 1))  # Specify x ticks
    ax.set_yticks(np.arange(-1, 1, 1))  # Specify y ticks
    ax.set_zticks(np.arange(-1, 1, 1))  # Specify z ticks


    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
##########################
    

def visualize_transition(constraints, particles, mdp_class, weights=None, fig=None, text=None, knowledge_type = 'common_knowledge', plot_filename = 'transition', vars_filename = 'sim_run'):

    # From BEC_viz.visualize_pf_transition function
   
    '''
    Visualize the change in particle filter due to constraints
    '''
    if fig == None:
        fig = plt.figure()

   
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax3 = fig.add_subplot(1, 2, 2, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    fs = 20
    ax1.title.set_text('Particles before demonstrations')
    ax3.title.set_text('Particles after demonstrations')
    ax1.title.set_fontsize(fs)
    ax3.title.set_fontsize(fs)

    fig.suptitle(text, fontsize=30)

    # plot particles before and after the constraints
    particles.plot(fig=fig, ax=ax1, plot_prev=True)
    particles.plot(fig=fig, ax=ax3)


    view_params = plot_constraints_views(constraints, fig, [ax3], knowledge_type = knowledge_type, viz_planes=False)

    label_axes(ax1, mdp_class, weights, view_params = view_params)
    label_axes(ax3, mdp_class, weights, view_params = view_params)

    # # Add what constraints are being shown in the demo to the plot
    # if len(constraints) > 0:
    #     x_loc = 0.5
    #     y_loc = 0.1
    #     fig.text(0.2, y_loc, 'Constraints in this demo: ', fontsize=20)
    #     for cnst in constraints:
    #         fig.text(x_loc, y_loc, str(cnst), fontsize=20)
    #         y_loc -= 0.05


    # https://stackoverflow.com/questions/41167196/using-matplotlib-3d-axes-how-to-drag-two-axes-at-once
    # link the pan of the three axes together
    def on_move(event):
        if event.inaxes == ax1:
            # ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            ax3.view_init(elev=ax1.elev, azim=ax1.azim)
        # elif event.inaxes == ax2:
        #     ax1.view_init(elev=ax2.elev, azim=ax2.azim)
        #     ax3.view_init(elev=ax2.elev, azim=ax2.azim)
        elif event.inaxes == ax3:
            ax1.view_init(elev=ax3.elev, azim=ax3.azim)
            # ax2.view_init(elev=ax3.elev, azim=ax3.azim)
            return
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()

    # if params.show_plots_flag:
    #     plt.show()
    # if params.save_plots_flag:
    #     plt.savefig('plots/' + vars_filename + '_' + text +'.png', dpi=300)
    #     plt.close()
    # fig.savefig('plots/' + plot_filename +'.png', dpi=300)
    # plt.pause(10)
    # plt.close()
###########################
    
def plot_constraints_views(plot_constraints, fig, axs_list, knowledge_type = 'common_knowledge', viz_planes=True):

    # if viz_planes:
    #     view_params_vals = [[16, -160], [8, -120], [2, -60]]
    # else:
    #     view_params_vals = [[16, -160], [11, -108], [2, -60]]  # better view for common and joint belief

    view_params_vals = [[16, -160], [8, -110], [2, -60]]

    print('plot_constraints: ', plot_constraints,'axs: ', axs_list, 'knowledge_type: ', knowledge_type)

    if len(plot_constraints) > 0:
        for ax_n in axs_list:
            if knowledge_type == 'joint_knowledge':
                joint_constraints = plot_constraints
                
                for constraint in joint_constraints:
                    if viz_planes:
                        BEC_viz.visualize_planes(constraint, fig=fig, ax=ax_n)
                # for last set of constraints
                if constraint[0][0] == 0:
                    view_params = view_params_vals[0]
                elif constraint[0][1] == 0:
                    view_params = view_params_vals[1]
                elif constraint[0][2] == 0:
                    view_params = view_params_vals[2]
                else:
                    view_params = view_params_vals[0]
            else:
                if len(plot_constraints) > 1:
                    if viz_planes:
                        BEC_viz.visualize_planes(plot_constraints, fig=fig, ax=ax_n)
                    if plot_constraints[0][0][0] == 0:
                        view_params = view_params_vals[0]
                    elif plot_constraints[0][0][1] == 0:
                        view_params = view_params_vals[1]
                    elif plot_constraints[0][0][2] == 0:
                        view_params = view_params_vals[2]
                    else:
                        view_params = view_params_vals[0]
                else:
                    print('Only one constraints: ', plot_constraints)
                    if viz_planes:
                        BEC_viz.visualize_planes(plot_constraints, fig=fig, ax=ax_n)
                    if plot_constraints[0][0][0] == 0:
                        view_params = view_params_vals[0]
                    elif plot_constraints[0][0][1] == 0:
                        view_params = view_params_vals[1]
                    elif plot_constraints[0][0][2] == 0:
                        view_params = view_params_vals[2]
                    else:
                        view_params = view_params_vals[0]

    return view_params

######################
    

def visualize_team_knowledge_constraints(team_knowledge, weights, step_cost_flag, particles_team_teacher = None, kc_id = None, fig2=None, text=None, plot_min_constraints_flag = False, plot_text_flag = False, min_unit_constraints = [], plot_filename = 'team_knowledge_constraints', fig_title = None, vars_filename = 'sim_run'):




    ###########  functions ###################

    view_params_vals = [[16, -160], [8, -120], [2, -60]]



    # print(colored('Plotting team knowledge constraints..', 'blue'))
    
    if fig2 == None:
        fig2 = plt.figure()
        ax_a = fig2.add_subplot(1, 4, 1, projection='3d')
        ax_b = fig2.add_subplot(1, 4, 2, projection='3d')
        ax_c = fig2.add_subplot(1, 4, 3, projection='3d')
        ax_d = fig2.add_subplot(1, 4, 4, projection='3d')
        fig2.suptitle(fig_title, fontsize=16)

    # if kc_id is None:
    #     kc_id_list = range(len(team_knowledge['p1']))
    # else:
    #     kc_id_list = [kc_id]
    

    plot_id = 0
    view_params_flag = False
    for knowledge_id, knowledge_type  in enumerate(team_knowledge):

        # # plot constraints - updated
        # plot_constraints = []
        # if knowledge_type == 'joint_knowledge':
        #     # for each player
        #     for mem_id in range(len(team_knowledge[knowledge_type][kc_id_list[0]])):
        #         plot_constraints.append(copy.deepcopy(team_knowledge[knowledge_type][kc_id_list[0]][mem_id]))

        #         for rem_kc_id in kc_id_list[1:]:
        #             plot_constraints[mem_id].extend(team_knowledge[knowledge_type][rem_kc_id][mem_id])
        # else:
        #     for kc_index in kc_id_list:
        #         # print('team_knowledge[knowledge_type][kc_index]: ', team_knowledge[knowledge_type][kc_index])
        #         if len(plot_constraints) == 0:
        #             plot_constraints = copy.deepcopy(team_knowledge[knowledge_type][kc_index])
        #         else:
        #             plot_constraints.extend(team_knowledge[knowledge_type][kc_index])

        plot_constraints = copy.deepcopy(team_knowledge[knowledge_type])
        print('knowledge_type: ', knowledge_type, 'plot_constraints: ', plot_constraints)
        # # set view based on first constraint of first person
        # if not view_params_flag:
        #     if plot_constraints[0][0][0] == 0:
        #         view_params = view_params_vals[0]
        #     elif plot_constraints[0][0][1] == 0:
        #         view_params = view_params_vals[1]
        #     elif plot_constraints[0][0][2] == 0:
        #         view_params = view_params_vals[2]
        #     else:
        #         view_params = view_params_vals[0]
        #     view_params_flag = True


        # choose plot axes
        if knowledge_type == 'common_knowledge':
            plot_ax = ax_c
        elif knowledge_type == 'joint_knowledge':
            plot_ax = ax_d
        elif knowledge_type == 'p1':
            plot_ax = ax_a
        elif knowledge_type == 'p2':
            plot_ax = ax_b

        view_params = plot_constraints_views(plot_constraints, fig2, [plot_ax], knowledge_type = knowledge_type, viz_planes=False)


        # # Get the "tab10" colormap
        color_palette = plt.get_cmap("tab10")

        # # color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        color_list = ['tab:blue', 'tab:red']
        print('color_list: ', color_list)

        # plot knowledge constraints
        if knowledge_type != 'joint_knowledge':            
            # plot unit constraints
            if knowledge_type == 'p1':
                utils_teams.visualize_planes_team(plot_constraints, fig=fig2, ax=plot_ax, color='tab:blue')
            elif knowledge_type == 'p2':
                utils_teams.visualize_planes_team(plot_constraints, fig=fig2, ax=plot_ax, color='tab:red')
            else:
                utils_teams.visualize_planes_team(plot_constraints, fig=fig2, ax=plot_ax, color=color_list)
            ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(plot_constraints)
            poly = Polyhedron.Polyhedron(ieqs=ieqs)
            BEC_viz.visualize_spherical_polygon(poly, fig=fig2, ax=plot_ax, plot_ref_sphere=False, alpha=0.5)

        else:
            for i in range(len(plot_constraints)):
                cnsts = plot_constraints[i]
                # cnsts = BEC_helpers.remove_redundant_constraints(cnsts, weights, step_cost_flag)  
                utils_teams.visualize_planes_team([cnsts], fig=fig2, ax=plot_ax, color=color_list[i])
                ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage([cnsts])
                poly = Polyhedron.Polyhedron(ieqs=ieqs)
                BEC_viz.visualize_spherical_polygon(poly, fig=fig2, ax=plot_ax, plot_ref_sphere=False, alpha=0.5)

            plot_ax.set_title(knowledge_type)


        # plot particles
        if particles_team_teacher is not None:
            particles_team_teacher[knowledge_type].plot(fig=fig2, ax=plot_ax)

        plot_ax.set_title(knowledge_type)

        plot_id += 1


    
    label_axes(ax_a, params.mdp_class, weights, view_params = view_params)
    label_axes(ax_b, params.mdp_class, weights, view_params = view_params)
    label_axes(ax_c, params.mdp_class, weights, view_params = view_params)
    label_axes(ax_d, params.mdp_class, weights, view_params = view_params)


    # https://stackoverflow.com/questions/41167196/using-matplotlib-3d-axes-how-to-drag-two-axes-at-once
    # link the pan of the three axes together
    def on_move(event):

        if event.inaxes == ax_a:
            ax_b.view_init(elev=ax_a.elev, azim=ax_a.azim)
            ax_c.view_init(elev=ax_a.elev, azim=ax_a.azim)
            ax_d.view_init(elev=ax_a.elev, azim=ax_a.azim)
        elif event.inaxes == ax_b:
            ax_a.view_init(elev=ax_b.elev, azim=ax_b.azim)
            ax_c.view_init(elev=ax_b.elev, azim=ax_b.azim)
            ax_d.view_init(elev=ax_b.elev, azim=ax_b.azim)
        elif event.inaxes == ax_c:
            ax_a.view_init(elev=ax_c.elev, azim=ax_c.azim)
            ax_b.view_init(elev=ax_c.elev, azim=ax_c.azim)
            ax_d.view_init(elev=ax_c.elev, azim=ax_c.azim)
        elif event.inaxes == ax_d:
            ax_a.view_init(elev=ax_d.elev, azim=ax_d.azim)
            ax_b.view_init(elev=ax_d.elev, azim=ax_d.azim)
            ax_c.view_init(elev=ax_d.elev, azim=ax_d.azim)
            return
        
        fig2.canvas.draw_idle()
    
    
    plt.show()



################################

def visualize_transition_w_feedback(test_constraints, demo_constraints, particles, learning_factor, mdp_class, model_type = 'low_noise', weights=None, fig=None, text=None, knowledge_type = 'common_knowledge', plot_filename = 'transition', vars_filename = 'sim_run', plot_constraints=None):

    # From BEC_viz.visualize_pf_transition function
   
    '''
    Visualize the change in particle filter due to constraints
    '''
    if fig == None:
        fig = plt.figure()

   
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    
    fs = 20
    ax1.title.set_text('After demonstrations')
    ax2.title.set_text('After tests')
    ax3.title.set_text('After feedback')
    ax1.title.set_fontsize(fs)
    ax2.title.set_fontsize(fs)
    ax3.title.set_fontsize(fs)

    fig.suptitle(text, fontsize=30)

    # plot particles before and after the constraints
    info_gain = {}
    entropy = {}
    # info_gain['demo'] = particles.calc_info_gain(demo_constraints, learning_factor, model_type)
    # entropy['before_demo'] = particles.calc_entropy()

    # update demo constraints
    particles.update(demo_constraints, learning_factor, model_type = model_type)        
    # entropy['after_demo'] = particles.calc_entropy()
    particles.plot(fig=fig, ax=ax1)

    # if knowledge_type != 'joint_knowledge':
    #     info_gain['test'] = particles.calc_info_gain(test_constraints, learning_factor, model_type)

    # update test constraints
    if knowledge_type == 'joint_knowledge':
        # particles.update_jk(test_constraints, learning_factor, model_type = model_type)
        particles, prob_individual = update_jk(particles, test_constraints, learning_factor, model_type = model_type)

        fig2, ax_p2 = plt.subplots(1,3, figsize=(15,5))
        prob_individual_pd = pd.DataFrame(prob_individual, columns = ['p1', 'p2', 'p3'])
        print(type(prob_individual_pd))
        print(prob_individual_pd)
        # sns.lineplot(data=prob_individual_pd, y='p1', ax=ax_p2[0])
        # sns.lineplot(data=prob_individual_pd, y='p2', ax=ax_p2[1])
        # sns.lineplot(data=prob_individual_pd, y='p3', ax=ax_p2[2])
        ax_p2[0].plot(prob_individual_pd['p1'])
        ax_p2[1].plot(prob_individual_pd['p2'])
        ax_p2[2].plot(prob_individual_pd['p3'])

        
    elif knowledge_type == 'jk_1':
        particles, _ = update_jk(particles, test_constraints, learning_factor, model_type = model_type, member_id = 0)
    elif knowledge_type == 'jk_2':
        particles, _ = update_jk(particles, test_constraints, learning_factor, model_type = model_type, member_id = 1)
    elif knowledge_type == 'jk_3':
        particles, _ = update_jk(particles, test_constraints, learning_factor, model_type = model_type, member_id = 2)
    else:
        particles.update(test_constraints, learning_factor, model_type = model_type)

    # entropy['after_test'] = particles.calc_entropy()
    particles.plot(fig=fig, ax=ax2)

    # info_gain['feedback'] = particles.calc_info_gain(demo_constraints, learning_factor, model_type)

    # update particles with feedback
    particles.update(demo_constraints, learning_factor, model_type=model_type)
    particles.plot(fig=fig, ax=ax3)
    
    # entropy['after_feedback'] = particles.calc_entropy()
    

    view_params = plot_constraints_views(demo_constraints, fig, [ax3], knowledge_type = knowledge_type, viz_planes=False)

    utils_teams.visualize_planes_team(demo_constraints, fig=fig, ax=ax1)

    if knowledge_type == 'joint_knowledge':
        utils_teams.visualize_planes_team(plot_constraints, fig=fig, ax=ax2)
    else:
        utils_teams.visualize_planes_team(test_constraints, fig=fig, ax=ax2)
    utils_teams.visualize_planes_team(demo_constraints, fig=fig, ax=ax3)


    label_axes(ax1, mdp_class, weights, view_params = view_params)
    label_axes(ax2, mdp_class, weights, view_params = view_params)
    label_axes(ax3, mdp_class, weights, view_params = view_params)

    # Add what constraints are being shown in the demo to the plot
    x_loc = 0.5
    y_loc = 0.1
    # for cnst in info_gain:
    #     fig.text(x_loc, y_loc, 'info gain for' + cnst + ': ' + str(info_gain[cnst]), fontsize=14)
    #     y_loc -= 0.05

    # for ent in entropy:
    #     fig.text(x_loc, y_loc, 'entropy for' + ent + ': ' + str(ent[cnst]), fontsize=14)
    #     y_loc -= 0.05

    # print('info_gain: ', info_gain)
    # print('entropy: ', entropy)


    # https://stackoverflow.com/questions/41167196/using-matplotlib-3d-axes-how-to-drag-two-axes-at-once
    # link the pan of the three axes together
    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            ax3.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            ax3.view_init(elev=ax2.elev, azim=ax2.azim)
        elif event.inaxes == ax3:
            ax1.view_init(elev=ax3.elev, azim=ax3.azim)
            ax2.view_init(elev=ax3.elev, azim=ax3.azim)
            return
        fig.canvas.draw_idle()

    # def on_move(event):
    #     if event.inaxes == ax1:
    #         ax2.view_init(elev=ax1.elev, azim=ax1.azim)
    #     elif event.inaxes == ax2:
    #         ax1.view_init(elev=ax2.elev, azim=ax2.azim)
    #         return
    #     fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()

    plt.savefig('plots/' + vars_filename + '_' + text +'.png', dpi=300)
    plt.close()

    # if params.show_plots_flag:
    #     plt.show()
    # if params.save_plots_flag:
    #     plt.savefig('plots/' + vars_filename + '_' + text +'.png', dpi=300)
    #     plt.close()
    # fig.savefig('plots/' + plot_filename +'.png', dpi=300)
    # plt.pause(10)
    # plt.close()
###########################
    
def update_jk(particles, joint_knowledge, learning_factor, model_type = 'low_noise',  c=0.5, reset_threshold_prob=0.001, member_id = None):
        
    # joint_constraints = joint_knowledge[0]
    joint_constraints = joint_knowledge

    print('Update JK with constraints: {}'.format(joint_constraints))
    
    particles.weights_prev = copy.deepcopy(particles.weights)
    particles.positions_prev = copy.deepcopy(particles.positions)

    particles, prob_individual = reweight_jk(particles, joint_constraints, learning_factor, member_id) # particles reweighted even if one of the original constraints is satisfied.

    particles.weights = particles.weights / np.sum(particles.weights)  # normalize weights

    # Only resample particles
    n_eff = particles.calc_n_eff(particles.weights)
    # print('n_eff: {}'.format(n_eff))
    if n_eff < c * len(particles.weights):
        # a) use systematic resampling
        # resample_indexes = p_utils.systematic_resample(self.weights)
        # self.resample_from_index(resample_indexes)

        # b) use KLD resampling
        resample_indexes = particles.KLD_resampling()
        print('Resampling...')
        particles.resample_from_index(np.array(resample_indexes), model_type)

    particles.binned = False

    return particles, prob_individual


############################
def reweight_jk(particles, constraints, learning_factor, member_id, plot_particles_flag = False):
    '''
    :param constraints: normal of constraints / mean direction of VMF
    :param k: concentration parameter of VMF
    :return: probability of x under this composite distribution (uniform + VMF)
    '''

    
    plot_particles_flag = False
    # cnt = 0

    u_pdf_scaled = learning_factor/(2*np.pi)
    VMF_kappa, x1 = particles.solve_for_distribution_params(learning_factor)
    
    prob_individual_list = []
    for j, x in enumerate(particles.positions):
        
        prob, prob_individual = observation_probability_jk(u_pdf_scaled, VMF_kappa, x, constraints, plot_particles_flag, particles.x1, particles.x2, member_id)
        particles.weights[j] = particles.weights[j] * prob
        prob_individual_list.extend(prob_individual)
            
    return particles, prob_individual_list


def observation_probability_jk(u_pdf_scaled, VMF_kappa, x, joint_constraints, plot_particles_flag, x1, x2, member_id):
    prob = 1
    p = x.shape[1]

    # check if atleast one of the constraints is satisfied, i.e. none of the inverse constraints are satisfied
    team_constraint_satisfied_flag = []
    prob_individual = np.ones([1, len(joint_constraints)])
    
    ind_id = 0
    for individual_constraint in joint_constraints:
        # print('Individual constraints: ', individual_constraint)
        team_constraint_satisfied_flag.append(True)
        
        # check if the constraints for each individual member are satisfied
        for constraint in individual_constraint:
            dot = constraint.dot(x.T)

            if dot < 0:
                team_constraint_satisfied_flag[-1] = False
                # prob_individual[0, ind_id] *= p_utils.VMF_pdf(constraint, VMF_kappa, p, x, dot=dot) # use the von Mises dist
                prob_individual[0, ind_id] *= p_utils.VMF_pdf(constraint, VMF_kappa, p, x, dot=dot) * x1 * x2 # use the von Mises dist
                # prob_individual[0, ind_id] *= p_utils.VMF_pdf(constraint, 4, p, x, dot=dot) * x1 * x2 # use the von Mises dist
            else:
                prob_individual[0, ind_id] *= u_pdf_scaled # use the uniform dist

        ind_id += 1

    # Method 2: Use maximum probability distribution
    if member_id is not None:
        prob = prob_individual[0, member_id]
    else:
        prob = np.max(prob_individual)



    return prob, prob_individual
            

##########################

if __name__ == "__main__":

    # lock = multiprocessing.Lock()

    ## plot prior PF distribution and updated PF distribution
    params.team_size = 1
    team_prior, teacher_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, teacher_learning_factor=[0.8, 0.8, 0.8], team_prior = params.team_prior)
    
    # # plot_pf = copy.deepcopy(teacher_pf['p1'])
    # entire KC
    demo_constraints = [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])]

    # test_constraints = [[np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])], \
    #                     [np.array([[3,  0,  -2]])], \
    #                     [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])] ]
    
    # # just one constraint
    # demo_constraints = [np.array([[-1,  0,  2]])]

    # test_constraints = [[np.array([[-1,  0,  2]])], \
    #                     [np.array([[-2,  0,  1]])] ]

    teacher_pf['p1'].update(demo_constraints, 0.9, model_type = 'noise')


    # test_pf_1  = copy.deepcopy(teacher_pf['joint_knowledge'])
    # test_pf_2  = copy.deepcopy(teacher_pf['joint_knowledge'])
    # test_pf_3  = copy.deepcopy(teacher_pf['joint_knowledge'])
    
    

    # # plot_pf.plot()
    # # plt.show()
    visualize_transition(demo_constraints, teacher_pf['p1'], params.mdp_class, weights = params.weights['val'])

    ##################


    ### plot team knowledge constraints

    # team_knowledge = {'p1': [np.array([[-1,  0,  2]])], 
    #                 'p2': [np.array([[-5,  0,  2]])], 
    #                 'common_knowledge': [np.array([[-1,  0,  2]]), np.array([[-5,  0,  2]])], 
    #                 'joint_knowledge': [np.array([[-1,  0,  2]]), np.array([[-5,  0,  2]])]}


    # visualize_team_knowledge_constraints(team_knowledge, params.weights['val'], params.step_cost_flag)


    ##################
    # learning_factor = 0.8
    # test_constraints_team = []
    # for id in range(params.team_size):
    #     member = 'p' + str(id+1)
    #     # visualize_transition_w_feedback(test_constraints[id], demo_constraints, teacher_pf[member], learning_factor, params.mdp_class, \
    #     #                                 model_type = 'med_noise', weights = params.weights['val'], text=member)

    #     cnst_flag = True
    #     if len(test_constraints_team) > 1:
            
    #         for cnst in test_constraints_team:
    #             for cnst2 in test_constraints[id]:
    #                 if (cnst == cnst2).all():
    #                     cnst_flag = False
    #                     break
            
    #     if cnst_flag:
    #         test_constraints_team.extend(test_constraints[id])

    # print('test_constraints_team: ', test_constraints_team)


    # # visualize_transition_w_feedback(test_constraints_team, demo_constraints, teacher_pf['common_knowledge'], learning_factor, params.mdp_class, \
    # #                                 model_type = 'med_noise', weights = params.weights['val'], text='common')
    # visualize_transition_w_feedback(test_constraints, demo_constraints, teacher_pf['common_knowledge'], learning_factor, params.mdp_class, \
    #                                 model_type = 'med_noise', text = 'joint', weights = params.weights['val'], knowledge_type='joint_knowledge', plot_constraints=test_constraints_team)

    # visualize_transition_w_feedback(test_constraints[0], demo_constraints, test_pf_1, learning_factor, params.mdp_class, text = 'joint_1', \
    #                                 model_type = 'med_noise', weights = params.weights['val'], knowledge_type='jk_1')
    # visualize_transition_w_feedback(test_constraints[1], demo_constraints, test_pf_2, learning_factor, params.mdp_class, text = 'joint_2', \
    #                                 model_type = 'med_noise', weights = params.weights['val'], knowledge_type='jk_2')
    # visualize_transition_w_feedback(test_constraints[2], demo_constraints, test_pf_3, learning_factor, params.mdp_class, text = 'joint_3', \
    #                                 model_type = 'med_noise', weights = params.weights['val'], knowledge_type='jk_3')



    
