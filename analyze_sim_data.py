# Analyze simulation data

import pandas as pd
import ast
import json
import teams.teams_helpers as teams_helpers
import params_team as params
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import math
import os
from termcolor import colored
import warnings
import textwrap
import pickle
from ast import literal_eval
import copy

plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 14

# Set the seaborn color palette to colorblind
sns.set_palette("colorblind")

# Get the colorblind palette with the desired number of colors
colorblind_palette = sns.color_palette("colorblind", 6)

# Set the default color cycle for Matplotlib using rcParams
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colorblind_palette)



import teams.teams_helpers as team_helpers
import params_team as params
# import simulation.sim_helpers as sim_helpers
from teams import particle_filter_team as pf_team
import policy_summarization.BEC_visualization as BEC_viz
import policy_summarization.BEC_helpers as BEC_helpers

import sage.geometry.polyhedron.base as Polyhedron
import sage.all
from statsmodels.graphics.factorplots import interaction_plot
from pandas.api.types import CategoricalDtype


# from SALib.analyze import sobol
from pingouin import partial_corr
import teams.utils_teams as utils_teams
import random

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn import metrics
from scipy.stats import bartlett, levene

from policy_summarization import computational_geometry as cg

import ast
from statsmodels.stats.anova import anova_lm

warnings.simplefilter(action='ignore', category=FutureWarning)


def str_to_dict(string, var_type = None, splitter = ', '):
    # remove the curly braces from the string
    # print('String before: ', string)
    string = string.strip('[]')
    string = string.strip('{}')
 
    # split the string into key-value pairs
    pairs = string.split(splitter)
 
    # use a dictionary comprehension to create
    # the dictionary, converting the values to
    # integers and removing the quotes from the keys
    dict = {}
    # print(pairs)
    for i in range(len(pairs)):
        pair = pairs[i]
        # print('pair: ', pair)
        key, value = pair.split(': ')
        
        key = key.strip(' \' ')
        # print('key: ', key)
        # print('value: ', value)

        if splitter == ', ':
            value = value.strip(' \' ')  # strip is used only for leading and trailing characters
            value = value.strip('[]')
            value = value.strip('array')
            value = value.strip('([])')
            # print('value 1: ', value)
        else:
            value = value.replace('array', '')
            value = value.replace('(', '')
            value = value.replace(')', '')
            # print('Knowledge_type: ', key)
            # print('Type before: ', type(value))
            # print('Check before: ', value)
            # value = ast.literal_eval(value)
            value = eval(value)
            # print('value 2: ', value)
            if key == 'joint_knowledge':
                value_copy = []
                for val in value:
                    v_c = []
                    for v in val:
                        v_c.extend(np.array([v]))
                    value_copy.append(v_c)
                value = value_copy
            else:   
                value = [np.array(v) for v in value]
            # print('Type after: ', type(value))
            # print('Check after: ', value)
        
        if var_type == 'float':
            dict[key] = float(value)
        elif var_type == 'array':
            dict[key] = np.array(value, dtype=np.float32)
        else:
            # print('data type: ', type(value))   
            dict[key] = value

    # return {key[1:-2]: float(value.strip(' \' ')) for key, value in (pair.split(': ') for pair in pairs)}
    
    return dict
 

def normalize_knowledge(knowledge):
    knowledge['loop_count_normalized'] = np.zeros(len(knowledge))
    for i in range(len(knowledge)):
        knowledge['loop_count_normalized'][i] = knowledge['loop_count'][i]/max(knowledge['loop_count'])

    return knowledge




def sim_study_data_analysis(path, files, file_prefix_list, data_condition = None, runs_to_exclude_list=[], runs_to_analyze_list = [], vars_filename_prefix = '', max_trials=-1):

    ### data
    analysis_run_id = 1

    trial_data_list = []
    interaction_data_list = []
    particles_prob_list = []
    concept_data_list = []
    learning_incomplete_runs_list = []

    if data_condition is not None:
        vars_filename_prefix = vars_filename_prefix + '_' + data_condition

    for file in files:
        # check if file is a valid pickle file
        print('Checking file: ', file, '. file prefix list: ', file_prefix_list)
        for file_prefix in file_prefix_list:
            if file_prefix in file and '.pickle' in file and '.png' not in file and '.csv' not in file and 'counterfactuals' not in file:
                run_file_flag = True
                break
            else:
                run_file_flag = False

        # check if file needs to be excluded
        for runs_to_exclude in runs_to_exclude_list:
            if runs_to_exclude in file:
                run_file_flag = False
                break

        if run_file_flag:
            with open(path + '/' + file, 'rb') as f:
                sim_vars = pickle.load(f)
            print('Opening file: ', file)
            
            
            # bec_final = sim_vars['BEC_knowledge_level'][len(sim_vars)-1]
            study_id = sim_vars['study_id'][0]
            sim_run_id = sim_vars['run_no'][0]

            # check if learning was completed based on final BEC knowledge
            # learning_complete = True
            # for k_type, k_val in bec_final.items():
            #     if k_val[0] != 1:
            #         learning_complete = False
            #         break

            # check if learning was completed based on sim status flag
            if sim_vars['sim_status'].iloc[-1] == 'Teaching complete':
                learning_complete = True
            else:
                learning_complete = False
            
            
            if learning_complete:
                print(colored('Reading file: ' + file + '. Run id: ' + str(analysis_run_id), 'blue' ))

                # update sim vars
                if max_trials != -1:
                    last_interaction_loop = min(max_trials, len(sim_vars))
                else:
                    last_interaction_loop = len(sim_vars)

                sim_vars = sim_vars.iloc[:last_interaction_loop]
                ###########
                    
                # compile trial data
                trial_data_dict = {}
                trial_data_dict['run_no'] = sim_run_id
                trial_data_dict['analysis_run_id'] = analysis_run_id
                trial_data_dict['study_id'] = study_id
                trial_data_dict['demo_strategy'] = sim_vars['demo_strategy'][0]
                trial_data_dict['team_composition'] = str(sim_vars['team_composition'][0])
                trial_data_dict['max_loop_count'] = sim_vars['loop_count'].iloc[-1]
                trial_data_dict['file_name'] = file
                trial_data_dict['sim_status'] = sim_vars['sim_status'].iloc[-1]
                

                # check for duplicate trials
                kc_id_list = sim_vars['knowledge_comp_id']
                run_start_id = []
                for id in range(len(kc_id_list)-1):
                    if kc_id_list[id+1] - kc_id_list[id] < 0:
                        print(colored('Duplicate trials found for file: ' + file + '. Run id: ' + str(sim_run_id), 'red' ))
                        run_start_id = [id+1]
                        break
                
                if len(run_start_id) != 0:
                    sim_vars = sim_vars.iloc[run_start_id[0]:].reset_index(drop=True)
                    print(sim_vars['knowledge_comp_id'])



                update_id = 1
                prev_concept_end_id = -1





                for i in range(last_interaction_loop):

                    interaction_data_dict = {}
                    interaction_data_dict['run_no'] = sim_run_id
                    interaction_data_dict['analysis_run_id'] = analysis_run_id
                    interaction_data_dict['study_id'] = study_id
                    interaction_data_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                    interaction_data_dict['team_composition'] = str(sim_vars['team_composition'][i])
                    interaction_data_dict['loop_count'] = int(sim_vars['loop_count'][i])
                    interaction_data_dict['test_constraints'] = sim_vars['test_constraints'][i]

                    interaction_data_list.append(interaction_data_dict)

                    for var_name in ['particles_prob_teacher_before_demo', 'particles_prob_learner_before_demo', 'particles_prob_teacher_after_demo', 'particles_prob_learner_after_demo', \
                                    'particles_prob_teacher_before_test', 'particles_prob_learner_before_test', 'particles_prob_teacher_after_test', 'particles_prob_teacher_after_feedback', 'particles_prob_learner_after_feedback']:

                        # ##  each test has probability separately
                        # for prob_dict in sim_vars[var_name][i]:
                        ## joint test has probability separately
                        team_particles_probability_dict = {}
                        for key, val in sim_vars[var_name][i].items():
                            # print('key: ', key, ' val: ', val)
                            team_particles_probability_dict[key] = float(val)

                        learner_BEC_prob = []
                        learner_prob = []
                        for p_id, player in enumerate(team_particles_probability_dict):
                            particles_probability_dict = {}
                            particles_probability_dict['analysis_run_id'] = analysis_run_id
                            particles_probability_dict['run_no'] = sim_run_id
                            particles_probability_dict['study_id'] = study_id
                            particles_probability_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                            particles_probability_dict['team_composition'] = str(interaction_data_dict['team_composition'])
                            particles_probability_dict['loop_count'] = int(sim_vars['loop_count'][i])
                            particles_probability_dict['test_constraints'] = interaction_data_dict['test_constraints']
                            particles_probability_dict['loop_count'] = int(sim_vars['loop_count'][i])
                            particles_probability_dict['update_type'] = var_name
                            particles_probability_dict['kc_id'] = sim_vars['knowledge_comp_id'][i]
                            particles_probability_dict['player_id'] = player
                            particles_probability_dict['learning_factor'] = sim_vars['team_learning_factor'][i][p_id]
                            particles_probability_dict['kc_prob'] = float(team_particles_probability_dict[player])


                            particles_teacher_current = sim_vars['particles_team_teacher_final'][i][player]
                            particles_learner_current = sim_vars['particles_team_learner_final'][i][player]
                            min_BEC_constraints = sim_vars['min_BEC_constraints'][i]
                            particles_learner_current.calc_particles_probability(min_BEC_constraints)
                            BEC_prob = particles_learner_current.particles_prob_correct
                            particles_probability_dict['BEC_prob'] = float(BEC_prob)

                            min_KC_constraints = sim_vars['min_KC_constraints'][i]
                            particles_learner_current.calc_particles_probability(min_KC_constraints)
                            KC_prob = particles_learner_current.particles_prob_correct
                            particles_probability_dict['calc_KC_prob'] = float(KC_prob)

                            if var_name == 'particles_prob_learner_after_feedback':
                                learner_BEC_prob.append(BEC_prob)
                                learner_prob.append(float(team_particles_probability_dict[player]))

                            # particles_prob = particles_prob.append(particles_probability_dict, ignore_index=True, sort=False)
                            particles_prob_list.append(particles_probability_dict)

                    #################
                            
                    # check for end of concept
                    if i < len(sim_vars)-1:
                        current_kc_id = sim_vars['knowledge_comp_id'][i]
                        next_kc_id = sim_vars['knowledge_comp_id'][i+1]

                        if current_kc_id != next_kc_id:
                            concept_data_dict = {}
                            concept_data_dict['run_no'] = sim_run_id
                            concept_data_dict['analysis_run_id'] = analysis_run_id
                            concept_data_dict['study_id'] = study_id
                            concept_data_dict['demo_strategy'] = str(sim_vars['demo_strategy'][i])
                            concept_data_dict['team_composition'] = str(sim_vars['team_composition'][i])
                            concept_data_dict['kc_id'] = int(current_kc_id)
                            concept_data_dict['N_interactions_learn_concept'] = int(i - prev_concept_end_id)
                            concept_data_dict['loop_count'] = int(sim_vars['loop_count'][i])
                            concept_data_dict['test_constraints'] = str(sim_vars['test_constraints'][i])
                            concept_data_dict['avg_prob_learner_after_feedback'] = np.mean(learner_prob)
                            concept_data_dict['avg_BEC_prob_learner_after_feedback'] = np.mean(learner_BEC_prob)

                            #print('learner_prob: ', learner_prob, 'learner_BEC_prob:', learner_BEC_prob)
                            #print(np.mean(learner_prob), np.mean(learner_BEC_prob))


                            concept_data_list.append(concept_data_dict)

                            prev_concept_end_id = i

                    elif i == len(sim_vars)-1:
                        current_kc_id = sim_vars['knowledge_comp_id'][i]

                        concept_data_dict = {}
                        concept_data_dict['run_no'] = sim_run_id
                        concept_data_dict['analysis_run_id'] = analysis_run_id
                        concept_data_dict['study_id'] = study_id
                        concept_data_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        concept_data_dict['team_composition'] = sim_vars['team_composition'][i]
                        concept_data_dict['kc_id'] = current_kc_id
                        concept_data_dict['N_interactions_learn_concept'] = i - prev_concept_end_id
                        concept_data_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        concept_data_dict['test_constraints'] = sim_vars['test_constraints'][i]
                        concept_data_dict['avg_prob_learner_after_feedback'] = np.mean(learner_prob)
                        concept_data_dict['avg_BEC_prob_learner_after_feedback'] = np.mean(learner_BEC_prob)

                        concept_data_list.append(concept_data_dict)

                        prev_concept_end_id = i

                ############
                trial_data_dict['average_team_knowledge'] = np.mean(learner_BEC_prob)  # team knowledge after the last interaction period

                trial_data_list.append(trial_data_dict)

                analysis_run_id += 1

                #####################
            else:
                print(colored('Learning incomplete for file: ' + file + '. Run id: ' + str(sim_run_id), 'red' ))
                learning_incomplete_runs_dict = {}
                learning_incomplete_runs_dict['run_no'] = sim_run_id
                learning_incomplete_runs_dict['study_id'] = study_id
                learning_incomplete_runs_dict['file_name'] = file
                learning_incomplete_runs_dict['team_composition'] = str(sim_vars['team_composition'][0])
                learning_incomplete_runs_dict['demo_strategy'] = sim_vars['demo_strategy'][0]
                learning_incomplete_runs_dict['max_loop_count'] = int(sim_vars['loop_count'].iloc[-1])
                learning_incomplete_runs_dict['sim_status'] = str(sim_vars['sim_status'].iloc[-1])
                learning_incomplete_runs_list.append(learning_incomplete_runs_dict)
            
            #############  Read all simulation trials data  ##########


    trial_data = pd.DataFrame(trial_data_list)
    interaction_data = pd.DataFrame(interaction_data_list)
    particles_prob = pd.DataFrame(particles_prob_list)  
    concept_data = pd.DataFrame(concept_data_list)
    learning_incomplete_runs = pd.DataFrame(learning_incomplete_runs_list)





    ## save files
    if max_trials != -1:
        with open(path + '/' + vars_filename_prefix + '_' + str(max_trials) + '_' + 'trial_data.pickle', 'wb') as f:
            pickle.dump(trial_data, f)

        trial_data.to_csv(path + '/' + vars_filename_prefix + '_' + str(max_trials) + '_' + 'trial_data.csv', index=False)


    else:
        with open(path + '/' + vars_filename_prefix + '_' + 'trial_data.pickle', 'wb') as f:
            pickle.dump(trial_data, f)

        with open(path + '/' + vars_filename_prefix + '_' + 'interaction_data.pickle', 'wb') as f:
            pickle.dump(interaction_data, f)
        
        with open(path + '/' + vars_filename_prefix + '_' + 'particles_prob.pickle', 'wb') as f:
            pickle.dump(particles_prob, f)

        with open(path + '/' + vars_filename_prefix + '_' + 'concept_data.pickle', 'wb') as f:
            pickle.dump(concept_data, f)

        with open(path + '/' + vars_filename_prefix + '_' + 'learning_incomplete_runs.pickle', 'wb') as f:
            pickle.dump(learning_incomplete_runs, f)

        trial_data.to_csv(path + '/' + vars_filename_prefix + '_' + 'trial_data.csv', index=False)
        interaction_data.to_csv(path + '/' + vars_filename_prefix + '_' + 'interaction_data.csv', index=False)
        particles_prob.to_csv(path + '/' + vars_filename_prefix + '_' + 'particles_prob.csv', index=False)
        concept_data.to_csv(path + '/' + vars_filename_prefix + '_' + 'concept_data.csv', index=False)
        learning_incomplete_runs.to_csv(path + '/' + vars_filename_prefix + '_' + 'learning_incomplete_runs.csv', index=False)

    
    print(colored('Number of trials processed: ' + str(len(trial_data)), 'blue'))
    print(colored('Number of incomplete learning runs: ' + str(len(learning_incomplete_runs)), 'red'))


    # ##############################################
        
    # ## Choose conditions to plot
    # know_list_full = ['individual', 'common_knowledge', 'joint_knowledge']
    # team_mix_full = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    # demo_list = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']
    
    # know_list = know_list_full[0:]
    # team_mix = team_mix_full[0:]

    # # # plot interaction count for both experimental conditions
    # f_c, ax_c = plt.subplots(ncols=1)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'max_loop_count', hue = 'team_composition', ax=ax_c, errorbar=('se',1)).set(title='Max number of interactions')

    # # interaction count for each experimental condition
    # f2, ax_2 = plt.subplots(ncols=2)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'max_loop_count', ax=ax_2[0], errorbar=('se',1)).set(title='Max number of interactions vs. Demo Strategy')
    # sns.barplot(data = trial_data, x = 'team_composition', y = 'max_loop_count', ax=ax_2[1], errorbar=('se',1)).set(title='Max number of interactions vs. Team composition')
    # plt.show()

    # #####################


    # f3, ax3 = plt.subplots(ncols=1)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'average_team_knowledge', hue = 'team_composition', ax=ax3, errorbar=('se',1)).set(title='Knowledge')

    # f4, ax_4 = plt.subplots(ncols=2)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'average_team_knowledge', ax=ax_4[0], errorbar=('se',1)).set(title='Knowledge vs. Demo Strategy')
    # sns.barplot(data = trial_data, x = 'team_composition', y = 'average_team_knowledge', ax=ax_4[1], errorbar=('se',1)).set(title='Knowledge vs. Team composition')
    # plt.show()



#################
def squarest_rectangle(area):
        # Initialize variables to store the dimensions of the squarest rectangle
        length = 1
        width = area
        
        # Initialize the minimum difference between length and width
        min_difference = abs(length - width)

        while min_difference > 2:
            area += 1
        
            # Iterate over all possible lengths up to the square root of the area
            for l in range(1, int(area**0.5) + 1):
                # Check if the length divides the area evenly
                if area % l == 0:
                    w = area // l  # Calculate the corresponding width
                    
                    # Calculate the difference between length and width
                    difference = abs(l - w)
                    
                    # Update dimensions if the current rectangle is "squarer"
                    if difference < min_difference:
                        length = l
                        width = w
                        min_difference = difference

        return length, width

#########################
def label_axes(ax, mdp_class, weights=None, view_params = None):
    fs = 10
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if weights is not None:
        ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100/2)
    if mdp_class == 'augmented_taxi2':
        ax.set_xlabel(r'$\mathregular{w_0}$: Mud', fontsize = fs)
        ax.set_ylabel(r'$\mathregular{w_1}$: Recharge', fontsize = fs)
    elif mdp_class == 'colored_tiles':
        ax.set_xlabel('X: Tile A (brown)')
        ax.set_ylabel('Y: Tile B (green)')
    else:
        ax.set_xlabel('X: Goal')
        ax.set_ylabel('Y: Skateboard')
    ax.set_zlabel('$\mathregular{w_2}$: Action', fontsize = fs)

    # ax.view_init(elev=16, azim=-160)
    if not view_params is None:
        elev = view_params[0]
        azim = view_params[1]
    else:
        elev = 30
        azim = -140
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

########################
    
def sim_study_plots(path, vars_filename_prefix = 'analysis'):

    team_mix_full = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    demo_list = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']






    try:
        # with open(path + '/' + vars_filename_prefix+ '_trial_data_subset_N15_checked.pickle', 'rb') as f:
        #     trial_data_subset = pickle.load(f)

        with open(path + '/' + vars_filename_prefix + '_6_trial_data_N15_checked.pickle', 'rb') as f:
            trial_data_subset = pickle.load(f)
    except:
        # For counterbalanced conditions
        print('To calculate subset data')

        with open(path + '/' + vars_filename_prefix+ '_6_trial_data.pickle', 'rb') as f:
            trial_data = pickle.load(f)

        with open(path + '/' + vars_filename_prefix+ '_trial_data_subset_N15_checked.pickle', 'rb') as f:
            full_trial_data_subset = pickle.load(f)

        unique_analysis_run_id = full_trial_data_subset['analysis_run_id'].unique()

        trial_data_subset = trial_data[trial_data['analysis_run_id'].isin(unique_analysis_run_id)]

        with open(path + '/' + vars_filename_prefix + '_6_trial_data_N15_checked.pickle', 'wb') as f:
            pickle.dump(trial_data_subset, f)
            
        # # choose subset of counterbalanced conditions
        # N=15 # number of conditions for each combination
        
        # subset_index = []
        # for team in team_mix_full:
        #     for demo in demo_list:
        #         index = trial_data[(trial_data['team_composition'] == str(team)) & (trial_data['demo_strategy'] == demo)].index
        #         print(len(index))
        #         index_to_choose = random.sample(list(index), N)
        #         subset_index.extend(index_to_choose)
        # trial_data_subset = trial_data.loc[subset_index]

        # with open(path + '/' + vars_filename_prefix + '_trial_data_subset_N15.pickle', 'wb') as f:
        #     pickle.dump(trial_data_subset, f)

        # trial_data_subset.to_csv(path + '/' + vars_filename_prefix + '_trial_data_subset_N15.csv', index=False)
    ###############
    
    trial_data = copy.deepcopy(trial_data_subset)
    

    # with open(path + '/' + vars_filename_prefix+ '_trial_data_subset.pickle', 'rb') as f:
    #     trial_data = pickle.load(f)

    with open(path + '/' + vars_filename_prefix+ '_interaction_data.pickle', 'rb') as f:
        interaction_data = pickle.load(f)

    with open(path + '/' + vars_filename_prefix+ '_particles_prob.pickle', 'rb') as f:
        prob_data = pickle.load(f)

    with open(path + '/' + vars_filename_prefix+ '_concept_data.pickle', 'rb') as f:
        concept_data = pickle.load(f)

    # Combine with baseline
    with open(path + '/analysis_baseline_trial_data_teams.pickle', 'rb') as f:
        trial_data_base = pickle.load(f)
    
    trial_data_w_baseline = pd.concat([trial_data_base, trial_data], ignore_index=True)

    # trial_data.to_csv(path + '/analysis_trial_data_combined.csv', index=False)

    # with open(path + '/analysis_trial_data_combined.pickle', 'wb') as f:

    ##################

    outliers = [14, 229] # identified from the boxplot; analysis_run_id; for N6 counterfactuals, N15 subset
    trial_data = trial_data[~trial_data['analysis_run_id'].isin(outliers)]

    print(len(trial_data), trial_data['analysis_run_id'])


    #     pickle.dump(trial_data, f)

    #####################################


    # # plot interaction count for both experimental conditions
    # f_c, ax_c = plt.subplots(ncols=1)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'max_loop_count', hue = 'team_composition', ax=ax_c, errorbar=('se',1)).set(title='Max number of interactions')

    # # interaction count for each experimental condition
    # f2, ax_2 = plt.subplots(ncols=2)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'max_loop_count', ax=ax_2[0], errorbar=('se',1)).set(title='Max number of interactions vs. Demo Strategy')
    # sns.barplot(data = trial_data, x = 'team_composition', y = 'max_loop_count', ax=ax_2[1], errorbar=('se',1)).set(title='Max number of interactions vs. Team composition')
    # plt.show()

    # # # # #####################

    # f3, ax3 = plt.subplots(ncols=1)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'average_team_knowledge', hue = 'team_composition', ax=ax3, errorbar=('se',1)).set(title='Knowledge')

    # f4, ax_4 = plt.subplots(ncols=2)
    # sns.barplot(data = trial_data, x = 'demo_strategy', y = 'average_team_knowledge', ax=ax_4[0], errorbar=('se',1)).set(title='Knowledge vs. Demo Strategy')
    # sns.barplot(data = trial_data, x = 'team_composition', y = 'average_team_knowledge', ax=ax_4[1], errorbar=('se',1)).set(title='Knowledge vs. Team composition')
    # plt.show()

    ###################
    # ### Paper figure
    demo_labels = ['Ind. (low)', 'Ind. (high)', 'Common', 'Joint']
    order_list = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']

    f_c, ax_c = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    
    sns.barplot(data = trial_data, x = 'demo_strategy', y = 'max_loop_count', ax=ax_c[0,0], order=order_list, errorbar=('se',1)).set(title='Max number of interactions to learn')
    ax_c[0,0].set_xlabel('Demo Strategy')
    ax_c[0,0].set_ylabel('No. interactions')
    ax_c[0,0].set_xticks(ticks=[0, 1, 2, 3], labels=demo_labels, fontsize=18)

    sns.barplot(data = trial_data, x = 'demo_strategy', y = 'average_team_knowledge', ax=ax_c[0,1], order=order_list, errorbar=('se',1)).set(title='Average team knowledge')
    ax_c[0,1].set_xlabel('Demo Strategy')
    ax_c[0,1].set_ylabel('Average team knowledge, $\mathregular{p_{BEC}}$')
    ax_c[0,1].set_xticks(ticks=[0, 1, 2, 3], labels=demo_labels, fontsize=18)


    sns.barplot(data = trial_data, x = 'team_composition', y = 'max_loop_count', ax=ax_c[1,0], errorbar=('se',1)).set(title='Max number of interactions to learn')
    ax_c[1,0].set_xlabel('Team Composition')
    # ax_c[1,0].set_ylabel('No. interactions')

    sns.barplot(data = trial_data, x = 'team_composition', y = 'average_team_knowledge', ax=ax_c[1,1], errorbar=('se',1)).set(title='Average team knowledge')
    ax_c[1,1].set_xlabel('Team Composition')
    # ax_c[1,1].set_ylabel('Average team knowledge, $\mathregular{p_{BEC}}$')    
    
    plt.show()
    #############
    # Violin plot - not uses

    # f2, axv_2 = plt.subplots(nrows=2, ncols=2)
    
    
    # sns.catplot(data=trial_data_w_baseline, x='demo_strategy', y='max_loop_count', kind="violin", cut=0)
    # axv_2_1 = plt.gca()
    # # axv_2_1.set_title('Max interactions to learn vs. Demo Strategy', fontsize=20)
    # axv_2_1.set_xlabel('Demo Strategy')
    # axv_2_1.set_ylabel('No. interactions')
    # axv_2_1.set_xticks(ticks=[0, 1, 2, 3, 4], labels=demo_labels, fontsize=18)
    # # axv_2_1.legend(title='Demo Strategy', title_fontsize='20', fontsize='20', labels=demo_labels)

    # sns.catplot(data=trial_data_w_baseline, x="demo_strategy", y="average_team_knowledge", kind="violin", cut=0)
    # axv_2_2 = plt.gca()
    # # axv_2_2.set_title('Average team knowledge vs. Demo Strategy', fontsize=20)
    # axv_2_2.set_xlabel('Demo Strategy')
    # axv_2_2.set_ylabel('Average team knowledge, $\mathregular{p_{BEC}}$')
    # axv_2_2.set_xticks(ticks=[0, 1, 2, 3, 4], labels=demo_labels, fontsize=18)

    # plt.show()

    # #####################
    # # Learning dynamics over time
    # team_mix_full = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    # # outliers = [103, 192, 316] run ids
    # outliers = [113, 211, 350] # analysis run ids


    # # categorize low and high learners
    # learner_data_list = []
    # prob_data['BEC_prob'] = prob_data['BEC_prob'].astype(float)
    # run_id_list = prob_data['analysis_run_id'].unique()
    # learner_data = pd.DataFrame()

    # for analysis_run_id in run_id_list:
    #     run_data = prob_data[(prob_data['analysis_run_id'] == analysis_run_id) & (prob_data['update_type'] == 'particles_prob_learner_after_feedback')].reset_index(drop=True)
    #     # print('run_data: ', run_data)
    #     team_composition = ast.literal_eval(run_data['team_composition'].iloc[0])
    #     team_composition = [int(x) for x in team_composition]

    #     loop_id_list = run_data['loop_count'].unique()

    #     # for loop_id in loop_id_list:
    #     loop_id = loop_id_list[-1]
    #     loop_data = run_data[run_data['loop_count'] == loop_id]
    #     # print('loop_data: ', loop_data)
        
    #     for member_id in range(len(team_composition)):
    #         member = 'p' + str(member_id+1)
    #         member_data = loop_data[loop_data['player_id'] == member]
    #         # print('member_data: ', member_data)
    #         if team_composition[member_id] == 0:
    #             member_data['learner_type'] = 'naive'
    #         else:
    #             member_data['learner_type'] = 'proficient'

    #         # learner_data_list.append(member_data.to_dict())
    #         learner_data = pd.concat([learner_data, member_data], ignore_index=True)

        
    # # learner_data = pd.DataFrame(learner_data_list)

    # with open(path + '/' + vars_filename_prefix + '_learner_data.pickle', 'wb') as f:
    #     pickle.dump(learner_data, f)

    # learner_data.to_csv(path + '/' + vars_filename_prefix + '_learner_data.csv', index=False)

    # ###############################

    # print(learner_data)

    # ## learner data
    # f6, ax6 = plt.subplots(ncols=1)
    # sns.barplot(data = learner_data, x = 'demo_strategy', y = 'BEC_prob', hue = 'learner_type', ax=ax6, errorbar=('se',1)).set(title='Learner data vs. Demo Strategy')


    # f7, ax7 = plt.subplots(ncols=1)
    # sns.barplot(data = learner_data, x = 'team_composition', y = 'BEC_prob', hue = 'learner_type', ax=ax7, errorbar=('se',1)).set(title='Learner data vs. Team Composition')


    # plt.show()
    ##############################   

    # Convert list to NumPy array
        
    #     ind_index_in_team = prob_data['player'][i].index(2)

    # prob_data = prob_data.drop(prob_data[prob_data['analysis_run_id'].isin(outliers)].index)
    # prob_data_plot = prob_data[prob_data['update_type'] == 'particles_prob_learner_after_feedback'].groupby(['analysis_run_id', 'team_composition', 'demo_strategy', 'loop_count']).agg({'BEC_prob': 'mean'}).reset_index()
    # print(prob_data_plot)

    # for i in range(4):
    #     f5, ax5 = plt.subplots(ncols=1)

    #     plot_data = prob_data_plot[(prob_data_plot['team_composition'] == str(team_mix_full[i]))]
    #     print('plot_data: ', len(plot_data), 'i: ', i)
    #     sns.lineplot(data = plot_data, x = 'loop_count', y = 'BEC_prob', hue = 'demo_strategy', hue_order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'], ax=ax5).set(title='Learning dynamics: ' + str(team_mix_full[i]))
    #     plt.show()


    #############
    # print(concept_data.dtypes)

    # concept_data['demo_strategy'] = concept_data['demo_strategy'].astype('string')
    # concept_data['team_composition'] = concept_data['team_composition'].astype('string')

    # run_list = trial_data['analysis_run_id'].unique()

    # index = concept_data['analysis_run_id'].isin(run_list)

    # concept_data = concept_data[index]


    # Concept data
    # f6, ax6 = plt.subplots(ncols=1)
    # sns.barplot(data = concept_data, x = 'demo_strategy', y = 'N_interactions_learn_concept', hue = 'team_composition', ax=ax6, errorbar=('se',1)).set(title='Concept learning')

    # f7, ax7 = plt.subplots(ncols=1)
    # sns.barplot(data = concept_data, x = 'kc_id', y = 'N_interactions_learn_concept', hue = 'demo_strategy', ax=ax7, errorbar=('se',1)).set(title='Concept learning')
    
    # f8, ax8 = plt.subplots(ncols=1)
    # sns.barplot(data = concept_data, x = 'kc_id', y = 'N_interactions_learn_concept', hue = 'team_composition', ax=ax8, errorbar=('se',1)).set(title='Concept learning')
    
    # plt.show()

    # kc_id_list = concept_data['kc_id'].unique()

    # for kc_id in kc_id_list:
    #     kc_data = concept_data[concept_data['kc_id'] == kc_id]

    #     f7, ax7 = plt.subplots(ncols=1)
    #     sns.barplot(data = kc_data, x = 'kc_id', y = 'avg_BEC_prob_learner_after_feedback', hue='demo_strategy', ax=ax7, errorbar=('se',1)).set(title='Concept learning for ' + str(kc_id))

    #     f8, ax8 = plt.subplots(ncols=1)
    #     sns.barplot(data = kc_data, x = 'kc_id', y = 'avg_BEC_prob_learner_after_feedback', hue='team_composition', ax=ax8, errorbar=('se',1)).set(title='Concept learning for ' + str(kc_id))

    #     plt.show()
    #################
        
    # ## individual through interactions
    # colors = sns.color_palette("colorblind", 7).as_hex()
    # players_color_dict = {'p1': str(colors[0]), 'p2': str(colors[1]), 'p3': str(colors[2])}
    # color_dict = {'particles_prob_learner_before_demo': str(colors[0]), 'particles_prob_learner_after_demo': str(colors[1]), 'particles_prob_learner_before_test': str(colors[2]),  'particles_prob_learner_after_feedback': str(colors[3])}
    # run_id_list = [1130]
    # run_filepath = 'N_diagnostic_tests/sim_study/03_02_sim_study_test_final_2_study_200_run_1130.pickle'
    # study_id = prob_data['study_id'].iloc[0]
    # kc_id_list = prob_data['kc_id'].unique()
    # print('kc_id_list: ', kc_id_list)




    # for run_id in run_id_list:
    #     plot_data = prob_data[(prob_data['run_no']==run_id) & \
    #                                ((prob_data['update_type']=='particles_prob_learner_before_demo') | (prob_data['update_type']=='particles_prob_learner_after_demo') | \
    #                                 (prob_data['update_type']=='particles_prob_learner_before_test') | (prob_data['update_type']=='particles_prob_learner_after_feedback'))]
        
    #     f_p, ax_p = plt.subplots(nrows=2, figsize=(15, 10))

    #     player_id_list = plot_data['player_id'].unique()
    #     interaction_id_list = plot_data['loop_count'].unique()

    #     with open(path + '/' + run_filepath, 'rb') as f:
    #         pf_data = pickle.load(f)


            





    #     int_fill_flag=False
    #     for player_id in player_id_list:
    #         plot_data_player = plot_data[plot_data['player_id']==player_id]
    #         interaction_number = np.arange(1, len(plot_data_player)+1)
    #         plot_data_player['interaction_number'] = interaction_number
    #         plot_data_player = plot_data_player.reset_index(drop=True)

    #         # get particles prob
    #         prob_particles = [0]
    #         for id, row in pf_data.iterrows():
    #             learner_pf_after_demo = row['particles_team_learner_after_demos'][player_id]
    #             learner_pf_after_feedback = row['particles_team_learner_final'][player_id]



    #             learner_pf_after_demo.calc_particles_probability(row['min_BEC_constraints'])
    #             prob_particles.append(learner_pf_after_demo.particles_prob_correct) # after demo
    #             prob_particles.append(learner_pf_after_demo.particles_prob_correct) # test
                
    #             learner_pf_after_feedback.calc_particles_probability(row['min_BEC_constraints'])
    #             prob_particles.append(learner_pf_after_feedback.particles_prob_correct) # after feedback

    #             if id != len(pf_data)-1:
    #                 prob_particles.append(learner_pf_after_demo.particles_prob_correct) # before demo


    #         print('player:', player_id, 'prob_particles: ', prob_particles)

    #         plot_data_player['BEC_prob_pf'] = prob_particles

    #         for kc_id in kc_id_list:
    #             plot_data_player_kc = plot_data_player[plot_data_player['kc_id']==kc_id]
    #             ax_p[0].axvline(x=plot_data_player_kc['interaction_number'].iloc[-1], color='black', linestyle='--')
    #             ax_p[1].axvline(x=plot_data_player_kc['interaction_number'].iloc[-1], color='black', linestyle='--')

    #         # plot for each interaction number
    #         for int_id in interaction_id_list:
    #             plot_data_int = plot_data_player[plot_data_player['loop_count']==int_id]
                    
    #             # plot for each KC of each player
    #             sns.lineplot(data = plot_data_int, x = 'interaction_number', y = 'kc_prob', ax=ax_p[0], color=players_color_dict[player_id]) 
    #             sns.lineplot(data = plot_data_int, x = 'interaction_number', y = 'BEC_prob_pf', ax=ax_p[1], color=players_color_dict[player_id])

    #         xtick_values = []
    #         xtick_pos = []
    #         for id, row in plot_data_player.iterrows():
    #             if row['update_type'] == 'particles_prob_learner_after_feedback':
    #                 xtick_pos.append(row['interaction_number'])
    #                 xtick_values.append(str(row['loop_count']))
    #             else:
    #                 xtick_pos.append(row['interaction_number'])
    #                 xtick_values.append('')


    #         ax_p[0].set_xticks(xtick_pos)
    #         ax_p[1].set_xticks(xtick_pos)

    #         ax_p[0].set_xticklabels(xtick_values)
    #         ax_p[1].set_xticklabels(xtick_values)

    #         if not int_fill_flag:
    #             for id, row in plot_data_player.iterrows():
    #                 if row['update_type'] == 'particles_prob_learner_after_feedback':
    #                     interaction_type = 'Feedback'
    #                 elif row['update_type'] == 'particles_prob_learner_before_test':
    #                     interaction_type = 'Test'
    #                 elif row['update_type'] == 'particles_prob_learner_after_demo':
    #                     interaction_type = 'Demo'
    #                 else:
    #                     interaction_type = ''
                    
    #                 # print('id:', id)
    #                 if id > 0:
    #                     # plt.axvline(x=row['combined_demo_id'], color='red', linestyle='--')
    #                     ax_p[0].axvspan(plot_data_player['interaction_number'].iloc[id-1], row['interaction_number'], alpha=0.2, color=color_dict[row['update_type']])
    #                     ax_p[0].text(row['interaction_number']-0.5, 0.1, interaction_type, rotation=90, fontsize=11, weight="bold")

    #                     ax_p[1].axvspan(plot_data_player['interaction_number'].iloc[id-1], row['interaction_number'], alpha=0.2, color=color_dict[row['update_type']])
    #                     ax_p[1].text(row['interaction_number']-0.5, 0.1, interaction_type, rotation=90, fontsize=11, weight="bold")
    #             int_fill_flag = True

    #     filename = 'Study ' + str(study_id) + '. Run ' + str(run_id) + '. Team mix ' + str(plot_data['team_composition'].iloc[0]) + '. Demo strategy ' + str(plot_data['demo_strategy'].iloc[0])
    #     f_p.suptitle(filename)


    #     plt.show()
    #     plt.savefig('models/augmented_taxi2/' + filename + '.png')
    #################################3
    ## plot distributions

    # # plot particle distribution
    # path = 'models/augmented_taxi2'
    # file = ''
    # with open(path + '/' + file, 'rb') as f:
    #     sim_vars = pickle.load(f)

    # ## 
    # for i in range(len(sim_vars)):
    #     learner_particles = sim_vars['particles_team_learner_final'][i]

    #     fig = plt.figure(figsize=(18, 14))
        
    #     row_len, col_len = squarest_rectangle(len(sim_vars))
    #     print('row_len: ', row_len, '. col_len: ', col_len)
    #     axs = np.array([fig.add_subplot(row_len, col_len, i+1, projection='3d') for i in range(len(sim_vars))])   
    #     axs2 = np.array([fig2.add_subplot(row_len, col_len, i+1, projection='3d') for i in range(len(sim_vars))])
        
    #     row_id, col_id = 0, 0
    #     for id, row in sim_vars.iterrows(): 

    #         test_constraints = row['test_constraints'] 

    #         learner_particles.calc_particles_probability(test_constraints)
    #         learner_pcr = learner_particles.particles_prob_correct

    #         learner_particles.plot(fig=fig, ax=axs[id])
    #         BEC_viz.visualize_planes(test_constraints, fig=fig, ax=axs[id])
    #         ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(test_constraints)
    #         poly = Polyhedron.Polyhedron(ieqs=ieqs)
    #         BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=axs[id], plot_ref_sphere=False, alpha=0.75)
    #         axs[id].set_title('id ' + str(id) + ': ' + row['interaction_type'], fontsize=11)


    #         if col_id < col_len-1:
    #             col_id += 1
    #         else:
    #             col_id = 0
    #             row_id += 1

    # plt.show()
        

    #     # if viz_flag:
        #     fig.savefig(path + '/' + vars_filename_prefix + '_learner_particles.png')
        #     fig2.savefig(path + '/' + vars_filename_prefix + '_teacher_particles.png')

    #####################







########################


def run_analysis_script(path, files, file_prefix_list, runs_to_exclude_list=[], runs_to_analyze_list = [], vars_filename_prefix = ''):
    

    ### load previously saved data
    try:
        with open (path + '/' + vars_filename_prefix + '_' + 'team_knowledge_level_long2.pickle', 'rb') as f:
            team_knowledge_level_long = pickle.load(f)

        with open (path + '/' + vars_filename_prefix + '_' + 'interaction_count2.pickle', 'rb') as f:
            interaction_count = pickle.load(f)

        with open (path + '/' + vars_filename_prefix + '_' + 'run_data2.pickle', 'rb') as f:
            run_data = pickle.load(f)

        with open (path + '/' + vars_filename_prefix + '_' + 'particles_prob2.pickle', 'rb') as f:
            particles_prob = pickle.load(f)
        
        
    except:
    ### analyze data
        team_unit_knowledge_level = pd.DataFrame()
        team_BEC_knowledge_level_expected = pd.DataFrame()
        team_BEC_knowledge_level = pd.DataFrame()
        team_knowledge = pd.DataFrame()
        learning_rate = pd.DataFrame()
        likelihood_correct_response = pd.DataFrame()
        particles_prob = pd.DataFrame()
        learning_incomplete_runs = pd.DataFrame()

        team_unit_knowledge_level_list = []
        team_BEC_knowledge_level_expected_list = []
        team_BEC_knowledge_level_list = []
        learning_incomplete_runs_list = []
        run_data_list = []
        particles_prob_list = []
        interaction_count_list = []


        learning_complete_history = []
        run_history = []

        # intiialize unique simulation run id
        run_id = 0

        # initialize particles
        params.tean_size = 1
        # particles_team_learner = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = [0.8], team_prior = params.team_prior, pf_flag='learner', vars_filename='', model_type='no_noise')
        # particles_initial = copy.deepcopy(particles_team_learner['p1'])


        for file in files:

            # check if file is a valid csv
            for file_prefix in file_prefix_list:
                if file_prefix in file and '.pickle' in file:
                    run_file_flag = True
                    break
                else:
                    run_file_flag = False

            # check if file needs to be excluded
            for runs_to_exclude in runs_to_exclude_list:
                if runs_to_exclude in file:
                    run_file_flag = False
                    break

            
            if run_file_flag:

                # sim_vars = pd.read_csv(path + '/' + file)

                # print('bec_final: ', bec_final)
                # bec_final = sim_vars['BEC_knowledge_level'][len(sim_vars)-1]
                # BEC_team_knowledge_final= str_to_dict(bec_final, var_type = 'float')
                # # check if learning was completed
                # learning_complete = True
                # for k_type, k_val in BEC_team_knowledge_final.items():
                #     if k_val != 1:
                #         learning_complete = False
                #         break
                
                with open(path + '/' + file, 'rb') as f:
                    sim_vars = pickle.load(f)
                
                bec_final = sim_vars['BEC_knowledge_level'][len(sim_vars)-1]
                study_id = sim_vars['study_id'][0]
                sim_run_id = sim_vars['run_no'][0]

                # check if learning was completed
                learning_complete = True
                for k_type, k_val in bec_final.items():
                    if k_val[0] != 1:
                        learning_complete = False
                        break


                # print('learning_complete flag: ', learning_complete)
                
                if learning_complete:

                    print(colored('Reading file: ' + file + '. Run id: ' + str(run_id), 'blue' ))
                    
                    update_id = 1





                    for i in range(len(sim_vars)):
                        # unit knowledge level
                        tk = sim_vars['unit_knowledge_level'][i]
                        bec_k = sim_vars['BEC_knowledge_level'][i]
                        bec_k_e = sim_vars['BEC_knowledge_level_expected'][i]
                        tkc = sim_vars['team_knowledge'][i]
                        # particles_prob = sim_vars['particles_prob_learner_demo'][i]
                        # ilv = sim_vars['initial_likelihood_vars'][i]
                        # lcr = sim_vars['likelihood_correct_response'][i]


                        # if type(tk) == str and type(bec_k) == str and type(bec_k_e) == str and type(tkc) == str and type(ilv) == str and type(lcr) == str:
                        # if type(tk) == str and type(bec_k) == str and type(bec_k_e) == str and type(tkc) == str:
                        

                        ## common vars
                        test_constraints = sim_vars['test_constraints'][i]
                        team_composition = str(sim_vars['team_composition'][i])
                        # print('team_composition: ', type(team_composition))
                        # print('test_constraints: ', test_constraints)


                        ### Unit knowledge level
                        # unit_knowledge_dict = str_to_dict(tk, var_type = 'float')

                        
                        unit_knowledge_dict = {}
                        for key, val in tk.items():
                            unit_knowledge_dict[key] = float(val[0])

                        unit_knowledge_dict['run_no'] = run_id
                        unit_knowledge_dict['study_id'] = study_id
                        unit_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        unit_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        unit_knowledge_dict['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        unit_knowledge_dict['file_name'] = file
                        # print('unit_knowledge_dict: ', unit_knowledge_dict)
                        # team_unit_knowledge_level = team_unit_knowledge_level.append(unit_knowledge_dict, ignore_index=True, sort=False)
                        team_unit_knowledge_level_list.append(unit_knowledge_dict)

                        ### BEC knowledge level 
                        # BEC_team_knowledge_dict = str_to_dict(bec_k, var_type = 'float')
                        
                        BEC_team_knowledge_dict = {}
                        for key, val in bec_k.items():
                            BEC_team_knowledge_dict[key] = float(val[0])

                        BEC_team_knowledge_dict['run_no'] = run_id
                        BEC_team_knowledge_dict['study_id'] = study_id
                        BEC_team_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        BEC_team_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        BEC_team_knowledge_dict['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        BEC_team_knowledge_dict['team_composition'] = team_composition
                        BEC_team_knowledge_dict['file_name'] = file

                        # print('BEC_team_knowledge_dict: ', BEC_team_knowledge_dict)
                        # team_BEC_knowledge_level = team_BEC_knowledge_level.append(BEC_team_knowledge_dict, ignore_index=True, sort=False)
                        team_BEC_knowledge_level_list.append(BEC_team_knowledge_dict)
                    
                        ### expected BEC knowledge level
                        # BEC_team_knowledge_dict_expected = str_to_dict(bec_k_e, var_type = 'float')

                        BEC_team_knowledge_dict_expected = {}
                        for key, val in bec_k_e.items():
                            BEC_team_knowledge_dict_expected[key] = float(val[0])

                        BEC_team_knowledge_dict_expected['run_no'] = run_id
                        BEC_team_knowledge_dict_expected['study_id'] = study_id
                        BEC_team_knowledge_dict_expected['loop_count'] = int(sim_vars['loop_count'][i])
                        BEC_team_knowledge_dict_expected['demo_strategy'] = sim_vars['demo_strategy'][i]
                        BEC_team_knowledge_dict_expected['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        BEC_team_knowledge_dict_expected['file_name'] = file
                        # team_BEC_knowledge_level_expected = team_BEC_knowledge_level_expected.append(BEC_team_knowledge_dict_expected, ignore_index=True, sort=False)
                        team_BEC_knowledge_level_expected_list.append(BEC_team_knowledge_dict_expected)

                        # # knowledge mix condition
                        # learning_rate_dict = str_to_dict(ilv, var_type = 'array', splitter='),')
                        # learning_rate_dict['run_no'] = run_id
                        # learning_rate_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        # learning_rate_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        # learning_rate = learning_rate.append(learning_rate_dict, ignore_index=True)
                        # # print('learning_rate_dict: ', learning_rate_dict)

                        ### team knowledge constraints
                        # team_knowledge_dict = str_to_dict(tkc, splitter = ', \'')
                        
                        ## not working - to be fixed later; not being used currently
                        # team_knowledge_dict = {}
                        # for key, val in bec_k_e.items():
                        #     print('key: ', key, ' val: ', val)
                        #     team_knowledge_dict[key] = float(val[0])
                        # team_knowledge_dict['run_no'] = run_id
                        # team_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        # team_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        # team_knowledge_dict['file_name'] = file

                        # print('team_knowledge_dict: ', team_knowledge_dict)
                        # team_knowledge = team_knowledge.append(team_knowledge_dict, ignore_index=True)

                        # # likelihood response
                        # team_likelihood_correct_response_dict = {}
                        # # print('lcr before: ', lcr)
                        # lcr = lcr.strip('[]')
                        # lcr = lcr.split(' ')
                        # lcr = [i for i in lcr if i != '']
                        # # print('lcr: ', lcr)
                        # lcr_array = np.array(list(lcr), dtype=float)
                        # # print('lcr_array: ', lcr_array)
                        # team_likelihood_correct_response_dict['p1'] = lcr_array[0]
                        # team_likelihood_correct_response_dict['p2'] = lcr_array[1]
                        # team_likelihood_correct_response_dict['p3'] = lcr_array[2]
                        # team_likelihood_correct_response_dict['common_knowledge'] = []
                        # team_likelihood_correct_response_dict['joint_knowledge'] = []
                        # team_likelihood_correct_response_dict['run_no'] = run_id
                        # team_likelihood_correct_response_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        # team_likelihood_correct_response_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        # likelihood_correct_response = likelihood_correct_response.append(team_likelihood_correct_response_dict, ignore_index=True)


                        ### knowledge component and learning factor
                        kc_variables = sim_vars['variable_filter'][i]
                        # print('kc_variables: ', type(kc_variables))
                        if i==0:
                            current_kc_variable = kc_variables
                        if (current_kc_variable != kc_variables).any():
                            update_id = 1  #reset update id for new KC
                            current_kc_variable = kc_variables

                        if (kc_variables == [[0, 1, 0]]).all():
                            kc_id = 1
                        elif (kc_variables == [[1, 0, 0]]).all():
                            kc_id = 2
                        elif (kc_variables == [[0, 0, 0]]).all():
                            kc_id = 3
                        else:
                            print(colored('Unrecognized variable filter: ' + kc_variables, 'red'))

                        # kc_variables = sim_vars['variable_filter'][i]
                        # print('kc_variables: ', type(kc_variables))
                        # if i==0:
                        #     current_kc_variable = kc_variables
                        # if (current_kc_variable != kc_variables):
                        #     update_id = 1  #reset update id for new KC
                        #     current_kc_variable = kc_variables

                        # if (kc_variables == '[[0. 1. 0.]]'):
                        #     kc_id = 1
                        # elif (kc_variables == '[[1. 0. 0.]]'):
                        #     kc_id = 2
                        # elif (kc_variables == '[[0. 0. 0.]]'):
                        #     kc_id = 3
                        # else:
                        #     print(colored('Unregognized variable filter: ' + kc_variables, 'red'))

                        
                        lf = sim_vars['team_learning_factor'][i]

                        lf_array = lf

                        # # print('lf: ', lf)
                        # lf = lf.strip('[]')
                        # lf = lf.split(' ')
                        # lf = [i for i in lf if i != '']
                        # # print('lcr: ', lcr)
                        # lf_array = np.array(list(lf), dtype=float)
                        # # print('lf_array: ', lf_array)

                        

                        # #### particles probability
                        # if 'particles_prob_learner_demo' in sim_vars.columns:
                        #     # team_particles_probability_dict = str_to_dict(sim_vars['particles_prob_learner_demo'][i], var_type=float)
                        #     var_name = 'particles_prob_learner_demo'
                        # elif 'particles_prob_learner_demos' in sim_vars.columns:
                        #     # team_particles_probability_dict = str_to_dict(sim_vars['particles_prob_learner_demos'][i], var_type=float)
                        #     var_name = 'particles_prob_learner_demos'
                        # elif 'particles_prob_learner_before_test' in sim_vars.columns:
                        #     # team_particles_probability_dict = str_to_dict(sim_vars['particles_prob_learner_before_test'][i], var_type=float)
                        #     var_name = 'particles_prob_learner_before_test'

                        for var_name in ['particles_prob_teacher_before_demo', 'particles_prob_learner_before_demo', 'particles_prob_teacher_after_demo', 'particles_prob_learner_after_demo', \
                                         'particles_prob_teacher_before_test', 'particles_prob_learner_before_test', 'particles_prob_teacher_after_test', 'particles_prob_teacher_after_feedback', 'particles_prob_learner_after_feedback']:

                        
                            # test_id = 1
                            # print('sim_vars[var_name][i]:', sim_vars[var_name][i])

                            # ##  each test has probability separately
                            # for prob_dict in sim_vars[var_name][i]:
                            ## joint test has probability separately
                            team_particles_probability_dict = {}
                            for key, val in sim_vars[var_name][i].items():
                                print('key: ', key, ' val: ', val)
                                team_particles_probability_dict[key] = float(val)

                            
                            # for older trials; only one probability
                            # if True:
                            #     team_particles_probability_dict = str_to_dict(sim_vars[var_name][i], var_type=float)
                            ###########

                            # print('team_particles_probability_dict: ', team_particles_probability_dict)

                            for p_id, player in enumerate(team_particles_probability_dict):
                                particles_probability_dict = {}
                                particles_probability_dict['run_no'] = run_id
                                particles_probability_dict['study_id'] = study_id
                                particles_probability_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                                particles_probability_dict['team_composition'] = team_composition
                                particles_probability_dict['loop_count'] = int(sim_vars['loop_count'][i])
                                # particles_probability_dict['test_id'] = test_id
                                particles_probability_dict['test_constraints'] = test_constraints
                                particles_probability_dict['update_id'] = update_id
                                particles_probability_dict['update_type'] = var_name
                                particles_probability_dict['kc_id'] = kc_variables
                                particles_probability_dict['player_id'] = player
                                particles_probability_dict['learning_factor'] = lf_array[p_id]
                                particles_probability_dict['kc_prob'] = float(team_particles_probability_dict[player])
                                particles_probability_dict['file_name'] = file


                                particles_teacher_current = sim_vars['particles_team_teacher_final'][i][player]
                                particles_learner_current = sim_vars['particles_team_learner_final'][i][player]

                                # print(particles_teacher_current.positions)
                                # print(particles_learner_current.positions)
                                # print(particles_teacher_current.weights)
                                # print(particles_learner_current.weights)
                                min_BEC_constraints = sim_vars['min_BEC_constraints'][i]

                                
                                particles_learner_current.calc_particles_probability(min_BEC_constraints)
                                BEC_prob = particles_learner_current.particles_prob_correct
                                # print('min_BEC_constraints: ', min_BEC_constraints, 'BEC_prob: ', BEC_prob)
                                particles_probability_dict['BEC_prob'] = BEC_prob

                                # particles_prob = particles_prob.append(particles_probability_dict, ignore_index=True, sort=False)
                                particles_prob_list.append(particles_probability_dict)

                                # test_id += 1
                        ################   
                        
                        update_id += 1

                        # else:
                        #     print(colored('Some non-string variables...','red'))


                    # # BEC constraints
                    # bec = sim_vars['min_BEC_constraints'][i]
                    # bec = bec.replace('array', '')
                    # bec = bec.replace('(', '')
                    # bec = bec.replace(')', '')
                    # bec = ast.literal_eval(bec)
                    # bec_copy = []
                    # for val in bec:
                    #     bec_copy.extend(np.array([val]))
                    # bec = bec_copy

                    # plot knowledge constraints
                    # teams_helpers.visualize_team_knowledge_constraints(bec, team_knowledge_dict, unit_knowledge_dict, BEC_team_knowledge_dict, params.mdp_class, weights=params.weights['val'])

                        # team_BEC_knowledge_level = normalize_knowledge(team_BEC_knowledge_level)

                else:
                    print(colored('Learning incomplete for file: ' + file + '. Run id: ' + str(run_id), 'red' ))
                    learning_incomplete_runs_dict = {}
                    learning_incomplete_runs_dict['run_no'] = run_id
                    learning_incomplete_runs_dict['study_id'] = study_id
                    learning_incomplete_runs_dict['file_name'] = file
                    learning_incomplete_runs_dict['team_composition'] = str(sim_vars['team_composition'][0])
                    learning_incomplete_runs_dict['demo_strategy'] = sim_vars['demo_strategy'][0]
                    learning_incomplete_runs_dict['max_loop_count'] = sim_vars['loop_count'].iloc[-1]
                    # learning_incomplete_runs = learning_incomplete_runs.append(learning_incomplete_runs_dict, ignore_index=True, sort=False)
                    learning_incomplete_runs_list.append(learning_incomplete_runs_dict)
                
                ######
                run_history.append(run_id)
                learning_complete_history.append(learning_complete)
                run_id += 1
                
        # datafraames
        team_unit_knowledge_level = pd.DataFrame(team_unit_knowledge_level_list)
        team_BEC_knowledge_level = pd.DataFrame(team_BEC_knowledge_level_list)
        team_BEC_knowledge_level_expected = pd.DataFrame(team_BEC_knowledge_level_expected_list)
        learning_incomplete_runs = pd.DataFrame(learning_incomplete_runs_list)
        particles_prob = pd.DataFrame(particles_prob_list)



        #### process team knowledge data
        BEC_knowledge_level = []
        BEC_knowledge_level_expected = []
        knowledge_type = []
        ind_knowledge_type = []
        normalized_loop_count = []
        lcr_var = []
        team_mix_var = []
        team_knowledge_level_min = team_BEC_knowledge_level.copy(deep=True)
        # print('team_knowledge_level_min columns: ', team_knowledge_level_min.columns)
        team_knowledge_level_min = team_knowledge_level_min.drop(['p1', 'p2', 'p3', 'common_knowledge', 'joint_knowledge'], axis=1)


        #### Long format data of team knowledge level
        team_knowledge_level_long = pd.DataFrame()
        for know_id in ['p1', 'p2', 'p3', 'common_knowledge', 'joint_knowledge']:
            team_knowledge_level_long = pd.concat([team_knowledge_level_long, team_knowledge_level_min])
            BEC_knowledge_level.extend(team_BEC_knowledge_level[know_id])
            BEC_knowledge_level_expected.extend(team_BEC_knowledge_level_expected[know_id])
            ind_knowledge_type.extend([know_id]*len(team_BEC_knowledge_level[know_id]))
            # lcr_var.extend(likelihood_correct_response[know_id])
            # team_mix_var.extend(learning_rate_index)
            
            if 'p' in know_id:
                knowledge_type.extend(['individual']*len(team_BEC_knowledge_level[know_id]))
            else:
                knowledge_type.extend([know_id]*len(team_BEC_knowledge_level[know_id]))

        normalized_loop_count = team_knowledge_level_long['loop_count'].copy(deep=True)
        team_knowledge_level_long['BEC_knowledge_level'] = BEC_knowledge_level
        team_knowledge_level_long['BEC_knowledge_level_expected'] = BEC_knowledge_level_expected
        team_knowledge_level_long['knowledge_type'] = knowledge_type
        team_knowledge_level_long['ind_knowledge_type'] = ind_knowledge_type
        # team_knowledge_level_long['likelihood_correct_response'] = lcr_var
        team_knowledge_level_long['normalized_loop_count'] = normalized_loop_count
        # team_knowledge_level_long['team_mix'] = team_mix_var

        
        team_knowledge_level_long.to_csv(path + '/' + vars_filename_prefix + '_' + 'team_knowledge_level_long.csv')
        with open (path + '/' + vars_filename_prefix + '_' + 'team_knowledge_level_long.pickle', 'wb') as f:
            pickle.dump(team_knowledge_level_long, f)
        team_knowledge_level_long.describe(include='all').to_csv(path + '/' + vars_filename_prefix + '_' + 'descriptives.csv')
        # team_knowledge_level_long = pd.read_csv('models/augmented_taxi2/team_knowledge_level_long.csv')

        
        ############## concept-wise interaction count
        concept_ids = team_BEC_knowledge_level['knowledge_comp_id'].unique()
        unique_ids = team_knowledge_level_long['run_no'].unique()
        interaction_count = pd.DataFrame()

        team_BEC_knowledge_level.to_csv(path + '/' + vars_filename_prefix + '_' + 'team_BEC_knowledge_level.csv')
        # print(team_BEC_knowledge_level)

        # for id in unique_ids:
        for id in range(len(run_history)):
            interaction_count_dict = {}
            print('Run: ', run_history[id])
            for c_id in range(len(concept_ids)):
                interaction_count_dict['run_no'] = run_history[id]
                interaction_count_dict['learning_complete_flag'] = learning_complete_history[id]
                if learning_complete_history[id]:
                    interaction_count_dict['demo_strategy'] = ''.join(str(element) for element in team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == run_history[id]]['demo_strategy'].iloc[0])
                    interaction_count_dict['team_composition'] = ''.join(str(element) for element in team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == run_history[id]]['team_composition'].iloc[0])

                    run_team_knowledge_data = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == run_history[id]]
                    
                    if concept_ids[c_id] <= run_team_knowledge_data['knowledge_comp_id'].iloc[-1]:
                        loop_id_list = []
                        for idx, row in team_BEC_knowledge_level.iterrows():
                            # print('row[run_no]: ', row['run_no'], ' run_history[id]: ', run_history[id], ' row[knowledge_comp_id]: ', row['knowledge_comp_id'], ' concept_ids[c_id]: ', concept_ids[c_id])
                            if row['run_no'] == run_history[id] and row['knowledge_comp_id'] == concept_ids[c_id]:
                                loop_id_list.append(row['loop_count'])
                        
                        max_loop_count = max(loop_id_list)-min(loop_id_list)+1
                    
                else:
                    interaction_count_dict['demo_strategy'] = ''
                    interaction_count_dict['team_composition'] = ''
                    max_loop_count = 0


                interaction_count_dict['Int_end_id_concept_'+str(concept_ids[c_id])] = max(loop_id_list)
                interaction_count_dict['N_int_concept_'+str(concept_ids[c_id])] = max_loop_count    

            # interaction_count = interaction_count.append(interaction_count_dict, ignore_index=True)
            interaction_count_list.append(interaction_count_dict)

        interaction_count = pd.DataFrame(interaction_count_list)

        # print('interaction_count: ', interaction_count)
        interaction_count.to_csv(path + '/' + vars_filename_prefix + '_' + 'interaction_count.csv')
        with open (path + '/' + vars_filename_prefix + '_' + 'interaction_count.pickle', 'wb') as f:
            pickle.dump(interaction_count, f)
        ######################################################################
        

        ## run-wise data
        # unique_ids = team_knowledge_level_long['run_no'].unique()
        
        run_data = pd.DataFrame()
        # for id in unique_ids:
        for id in range(len(run_history)):
            run_data_dict = {}
            run_data_dict['run_no'] = run_history[id]
            run_data_dict['learning_complete_flag'] = learning_complete_history[id]
            if learning_complete_history[id]:
                run_data_dict['demo_strategy'] = ''.join(str(element) for element in team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['demo_strategy'].iloc[0])
                run_data_dict['max_loop_count'] = int(team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['loop_count'].iloc[-1])
                run_data_dict['team_composition'] = ''.join(str(element) for element in team_knowledge_level_long[team_knowledge_level_long['run_no'] == id]['team_composition'].iloc[0])
                run_data_dict['file_name'] = ''.join(str(element) for element in team_knowledge_level_long[team_knowledge_level_long['run_no'] == id]['file_name'].iloc[0])
            else:
                run_data_dict['demo_strategy'] = ''
                run_data_dict['max_loop_count'] = 0
                run_data_dict['team_composition'] = ''
                run_data_dict['file_name'] = ''

            # run_data = run_data.append(run_data_dict, ignore_index=True)
            run_data_list.append(run_data_dict)

        run_data = pd.DataFrame(run_data_list)
        run_data.to_csv(path + '/' + vars_filename_prefix + '_' + 'run_data.csv')
        with open (path + '/' + vars_filename_prefix + '_' + 'run_data.pickle', 'wb') as f:
            pickle.dump(run_data, f)

        print(colored('Number of runs processed: ' + str(len(run_data)), 'red'))

        # run_data = pd.read_csv('models/augmented_taxi2/run_data.csv')

        print('Incomplete runs: ', learning_incomplete_runs)
        learning_incomplete_runs.to_csv(path + '/' + vars_filename_prefix + '_' + 'learning_incomplete_runs.csv')
        with open (path + '/' + vars_filename_prefix + '_' + 'learning_incomplete_runs.pickle', 'wb') as f:
            pickle.dump(learning_incomplete_runs, f)
        
        ###
        particles_prob.to_csv(path + '/' + vars_filename_prefix + '_' + 'particles_prob.csv')
        with open (path + '/' + vars_filename_prefix + '_' + 'particles_prob.pickle', 'wb') as f:
            pickle.dump(particles_prob, f)

        ## normalize loop count

        # for id in unique_ids:
        #     idx = team_knowledge_level_long[(team_knowledge_level_long['run_no'] == id)].index
        #     max_loop_count = np.max(team_knowledge_level_long.loc[idx, 'loop_count'])
        #     team_knowledge_level_long.loc[idx, 'normalized_loop_count'] = team_knowledge_level_long.loc[idx, 'loop_count']/max_loop_count


        # print(team_knowledge_level_long)
        # print(knowledge_type)

    ###################################################################################################################

    ##############################################   Plots    ##########################################################



    # for know_id in ['p1', 'p2', 'p3', 'common_knowledge', 'joint_knowledge']:
        # sns.lineplot(data = team_BEC_knowledge_level, x = 'loop_count', y = know_id, hue = 'demo_strategy').set(title='Knowledge level for ' + know_id)
        # plt.show()
        # plt.savefig('models/augmented_taxi2/BEC_knowledge_level_' + know_id + '.png')
        # plt.close()
    
    
    # f, ax = plt.subplots(nrows=2,ncols=3, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # row_id = 0
    # col_id = 0
    # for team_mix_cond in [[0, 1, 1]]:
    #     col_id = 0
    #     for know_id in ['individual', 'common_knowledge', 'joint_knowledge']:    
    #         plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) & (team_knowledge_level_long['team_composition']==str(team_mix_cond))]
    #         # plot_data.to_csv('models/augmented_taxi2/plot_data_' + know_id + '_' + str(team_mix_cond) + '.csv')
    #         print('plot_data: ', type(plot_data))
    #         sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #         col_id += 1 
    #     row_id += 1


    ##########
    
    ## Choose conditions to plot
    know_list_full = ['individual', 'common_knowledge', 'joint_knowledge']
    team_mix_full = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    demo_list = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']
    

    know_list = know_list_full[0:]
    team_mix = team_mix_full[0:]

    # # Plot knowledge level for each combination of team composition and knowledge type
    # col_id = 0
    # for team_mix_cond in team_mix:
    #     col_id = 0
    #     f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #     for know_id in know_list:    
    #         plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) & (team_knowledge_level_long['team_composition']==str(team_mix_cond))]
            
    #         # plot_data.to_csv('models/augmented_taxi2/plot_data_' + know_id + '_' + str(team_mix_cond) + '.csv')
    #         # print('Plotting  ', 'row_id: ', row_id, ' col_id: ', col_id, ' know_id: ', know_id, ' team_mix_cond: ', team_mix_cond)
    #         sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #         # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))

    #         col_id += 1 
        
    # plt.show()
            
    # plt.savefig('models/augmented_taxi2/BEC_knowledge_level_' + know_id + '.png')
    # plt.close()
    ########
        
    # # ## plot knowledge level for all conditions
    # f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # col_id = 0
    # for know_id in know_list:  
    #     plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) ]
    #     sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for a team mix: ' + str(team_mix_cond))
    #     # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #     col_id += 1 
    # # plt.show()
    ########

    # ## plot knowledge level for all team composition
    # for team_mix_cond in team_mix:
    #     f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #     col_id = 0
    #     for know_id in know_list:  
    #         plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) &(team_knowledge_level_long['team_composition']==str(team_mix_cond))]
    #         sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for a team mix: ' + str(team_mix_cond))
    #         # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #         col_id += 1 
    #     # plt.show()

    # # plot knowledge for demo strategy
    # for demo_id in demo_list:
    #     f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #     col_id = 0
    #     for know_id in know_list: 
    #         plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) &(team_knowledge_level_long['demo_strategy']==str(demo_id))]
    #         sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for demo strategy: ' + str(demo_id))
    #         # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #         col_id += 1 
    #     # plt.show()
    
    ########################

    ## Another knowledge level plot
    # sea = sns.FacetGrid(team_knowledge_level_long, row = "knowledge_type")
    # sea.map(sns.lineplot, "loop_count", "BEC_knowledge_level", "demo_strategy")
    # sea.set_axis_labels("Number of interaction sets/lessons", "Knowledge level")
    # sea.add_legend()

    # sea = sns.FacetGrid(team_knowledge_level_long, row = "knowledge_type")
    # sea.map(sns.lineplot, "normalized_loop_count", "BEC_knowledge_level", "demo_strategy")
    # sea.set_axis_labels("Number of interaction sets/lessons", "Knowledge level")
    # sea.add_legend()
    # plt.show(block='False')

    # # for each run
    # for id in unique_ids:
    #     idx = team_knowledge_level_long[(team_knowledge_level_long['run_no'] == id)].index
    #     sea = sns.FacetGrid(team_knowledge_level_long.loc[idx,:], row = "knowledge_type")
    #     sea.map(sns.lineplot, "normalized_loop_count", "BEC_knowledge_level", "demo_strategy")
    #     sea.set_axis_labels("Number of interaction sets/lessons", "Knowledge level")
    #     sea.add_legend()
    #     plt.show(block='False')


    # ## plot team BEC knowledge level
    # for id in unique_ids:
    #     idx = team_BEC_knowledge_level[(team_BEC_knowledge_level['run_no'] == id)].index

    #     # print(team_BEC_knowledge_level.columns)

    #     fig, axs = plt.subplots(2, 3, figsize=(9, 5), layout='constrained',
    #                     sharex=True, sharey=True)
    #     fig.suptitle('Run: ' + str(id))
    #     for nn, ax in enumerate(axs.flat):
    #         column_name = team_BEC_knowledge_level.columns[nn]
    #         y = team_BEC_knowledge_level.loc[idx,column_name]
    #         line, = ax.plot(team_BEC_knowledge_level.loc[idx, 'loop_count'], y, lw=2.5)
    #         ax.set_title(column_name, fontsize='small', loc='center')


    #         # plot verticle lines for visualizng end of concepts
    #         for kc_id in team_BEC_knowledge_level['knowledge_comp_id'].unique():
    #             max_idx = team_BEC_knowledge_level[(team_BEC_knowledge_level['knowledge_comp_id'] == kc_id) & (team_BEC_knowledge_level['run_no'] == id)].index.max()
    #             # print('kc_id:', kc_id, ' max_idx: ', max_idx)
    #             if not math.isnan(max_idx):
    #                 ax.axvline(team_BEC_knowledge_level.loc[max_idx, 'loop_count'], color='k', linestyle='--', linewidth=1)

    #         if nn == 4:
    #             break

    #     fig.supxlabel('Interaction Number')
    #     fig.supylabel('BEC Knowledge Level')


    ## Plot Unit knowledge level
    # fig2, axs2 = plt.subplots(2, 3, figsize=(9, 5), layout='constrained',
    #                 sharex=True, sharey=True)
    # for nn, ax in enumerate(axs2.flat):
    #     column_name = team_unit_knowledge_level.columns[nn]
    #     y = team_unit_knowledge_level.loc[idx,column_name]
    #     line, = ax.plot(team_unit_knowledge_level.loc[idx, 'loop_count'], y, lw=2.5)
    #     ax.set_title(column_name, fontsize='small', loc='center')

    #     # plot verticle lines for visualizng end of concepts
    #     for kc_id in team_BEC_knowledge_level['knowledge_comp_id'].unique():
    #         max_idx = team_BEC_knowledge_level[(team_BEC_knowledge_level['knowledge_comp_id'] == kc_id) & (team_BEC_knowledge_level['run_no'] == id)].index.max()
    #         ax.axvline(team_BEC_knowledge_level.loc[max_idx, 'loop_count'], color='k', linestyle='--', linewidth=1)

    #     if nn == 4:
    #         break

    # fig2.supxlabel('Interaction Number')
    # fig2.supylabel('Unit Knowledge Level')

        
    #############
    # # plot interaction count for concepts
    # var_list = []
    # for kc_id in range(1,4):
    #     var_list.append('N_int_concept_'+str(kc_id))

    # kc_data_long = pd.melt(interaction_count, id_vars=['run_no', 'learning_complete_flag', 'demo_strategy', 'team_composition'], value_vars=var_list, var_name='knowledge_comp_id', value_name='N_interactions')
    
    # for id in range(len(kc_data_long)):
    #     kc_data_long.loc[id, 'knowledge_comp_id'] = kc_data_long.loc[id, 'knowledge_comp_id'].split('_')[-1]

    # print(kc_data_long)

    # f_ic, ax_ic = plt.subplots(ncols=1)
    # sns.barplot(data = kc_data_long, x = 'knowledge_comp_id', y = 'N_interactions', hue = 'demo_strategy', ax=ax_ic, errorbar=('se',1)).set(title='Interaction count for concepts')

    # plt.show()
    ###################
    # # Number of interactions histogram - overall
    # f_sbc, ax_sbc = plt.subplots(ncols=1)
    # sns.histplot(data = run_data, x='max_loop_count', hue='learning_complete_flag', ax=ax_sbc).set(title='Distribution of maximum interactions for learning')

    # # plot interaction count for both experimental conditions
    f_c, ax_c = plt.subplots(ncols=1)
    sns.barplot(data = run_data, x = 'demo_strategy', y = 'max_loop_count', hue = 'team_composition', ax=ax_c, errorbar=('se',1)).set(title='Max number of interactions')

    # interaction count for each experimental condition
    f2, ax_2 = plt.subplots(ncols=2)
    sns.barplot(data = run_data, x = 'demo_strategy', y = 'max_loop_count', ax=ax_2[0], errorbar=('se',1)).set(title='Max number of interactions vs. Demo Strategy')
    sns.barplot(data = run_data, x = 'team_composition', y = 'max_loop_count', ax=ax_2[1], errorbar=('se',1)).set(title='Max number of interactions vs. Team composition')
    plt.show()
    
    #############

    ### plot probability of particles in the correct side of test
    # simulation run conditions
    dem_strategy_list = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']
    kc_id_list = ['[[0. 1. 0.]]', '[[1. 0. 0.]]', '[[0. 0. 0.]]']
    team_mix_list = ['[0, 0, 0]', '[0, 0, 2]', '[0, 2, 2]', '[2, 2, 2]']

    # print('particles_prob: ', particles_prob)
    
    

    # # plotting seprately for each team condition and demo strategy
    # for team_composition in team_mix_list:
    #     for dem_strategy in dem_strategy_list:

    #         trial_data = particles_prob[(particles_prob['team_composition']==str(team_composition)) & (particles_prob['demo_strategy']==dem_strategy)]

    #         if len(trial_data) > 0:
    #             f3, ax3 = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10,6))
    #             plt.subplots_adjust(wspace=0.1, hspace=0.5)  
            
    #             kc_no = 0
    #             for kc_id in kc_id_list:
    #                 print('team_composition: ', team_composition, ' dem_strategy: ', dem_strategy)

    #                 plot_data = particles_prob[(particles_prob['team_composition']==str(team_composition)) & (particles_prob['demo_strategy']==dem_strategy) & (particles_prob['kc_id']==kc_id)]
    #                 if len(plot_data) > 0:
    #                     print('plot_data: ', plot_data)
    #                     plot_title = 'Learning factor vs. particles_probability for kc ' +  kc_id + ', team mix: ' + str(team_composition) + ' and a demo strategy: ' + dem_strategy
    #                     wrapped_title = "\n".join(textwrap.wrap(plot_title, 40))
    #                     sns.lineplot(plot_data, x = 'learning_factor', y = 'particles_prob', ax=ax3[0, kc_no], legend=True).set(title=wrapped_title)
    #                     plot_title = 'Updatewise Learning factor vs. particles_probability for kc ' +  kc_id + ', team mix: ' + str(team_composition) + ' and a demo strategy: ' + dem_strategy
    #                     wrapped_title = "\n".join(textwrap.wrap(plot_title, 40))
    #                     sns.lineplot(plot_data, x = 'learning_factor', y = 'particles_prob', hue = 'update_id', ax=ax3[1, kc_no], legend=True).set(title=wrapped_title)
    #                     kc_no += 1
        
    #             # plt.show()

    # plot individual run learning dynamics
    # if len(files) == 1:






    ########


    # ## plotting for all experiment conditions

    # f3, ax3 = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.5)  

    # kc_no = 0
    # for kc_id in kc_id_list:
    #     print('kc_id: ', kc_id, 'particles_prob[kc_id]: ', type(particles_prob['kc_id']))
    #     plot_data_idx = []
    #     for idx, row in particles_prob.iterrows():
    #         if (row['kc_id'] == kc_id):
    #             plot_data_idx.append(idx)
    #     plot_data = particles_prob.loc[plot_data_idx]
    #     # plot_data = particles_prob[(particles_prob['kc_id'][0]==kc_id).all()]
    #     if len(plot_data) > 0:
    #         print('plot_data: ', plot_data)
    #         plot_title = 'Learning factor vs. particles_probability for kc ' +  kc_id 
    #         wrapped_title = "\n".join(textwrap.wrap(plot_title, 40))
    #         sns.lineplot(plot_data, x = 'learning_factor', y = 'particles_prob', ax=ax3[0, kc_no], legend=True).set(title=wrapped_title)
    #         plot_title = 'Learning factor vs. particles_probability for kc ' +  kc_id
    #         wrapped_title = "\n".join(textwrap.wrap(plot_title, 40))
    #         sns.lineplot(plot_data, x = 'learning_factor', y = 'particles_prob', hue = 'update_id', ax=ax3[1, kc_no], legend=True).set(title=wrapped_title)
    #         kc_no += 1

    #################

    # # Plot knowledge level for each combination of team composition and knowledge type
    # f_p, ax_p = plt.subplots(nrows = 2, ncols=4, sharex=True, sharey=True, figsize=(10,6))
    # for team_id in range(len(team_mix)):
    #     team_mix_cond = team_mix[team_id]
    #     plt.subplots_adjust(wspace=0.1, hspace=0.1)   
        
    #     # print('Plotting  ', 'row_id: ', row_id, ' col_id: ', col_id, ' know_id: ', know_id, ' team_mix_cond: ', team_mix_cond)
    #     plot_data = particles_prob[(particles_prob['update_type']=='particles_prob_learner_after_feedback') & (particles_prob['team_composition']==str(team_mix_cond))]
    #     sns.lineplot(plot_data, x = 'loop_count', y = 'kc_prob', hue = 'demo_strategy', ax=ax_p[0, team_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']).set(title='Prob. particles learn concept')
    #     sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_prob', hue = 'demo_strategy', ax=ax_p[1, team_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']).set(title='Prob. particles learn entire reward')

    #############
    ## inidividual prob at end of interaction
    # run_id_list = particles_prob['run_no'].unique() 

    # for run_id in run_id_list:
    #     plot_data = particles_prob[(particles_prob['update_type']=='particles_prob_learner_after_feedback') & (particles_prob['run_no']==run_id)]

    #     print('plot_data: ', plot_data)

    #     f_p, ax_p = plt.subplots(ncols=2)

    #     sns.lineplot(plot_data, x = 'loop_count', y = 'kc_prob', hue='player_id', ax=ax_p[0], errorbar=('se', 1), err_style="band", hue_order = ['p1','p2','p3'])
    #     sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_prob', hue='player_id', ax=ax_p[1], errorbar=('se', 1), err_style="band", hue_order = ['p1','p2','p3'])

    #     # ax_p[0].set_title('Prob. particles learn concept')
    #     # ax_p[1].set_title('Prob. particles learn entire reward')
    #     f_p.suptitle('Run: ' + str(run_id))
    #     plt.show()
    ##############

    ## individual through interactions
    colors = sns.color_palette("colorblind", 7).as_hex()
    color_dict = {'particles_prob_learner_before_demo': str(colors[0]), 'particles_prob_learner_after_demo': str(colors[1]), 'particles_prob_learner_before_test': str(colors[2]),  'particles_prob_learner_after_feedback': str(colors[3])}
    run_id_list = particles_prob['run_no'].unique()
    study_id = particles_prob['study_id'].iloc[0]

    for run_id in run_id_list:
        plot_data = particles_prob[(particles_prob['run_no']==run_id) & \
                                   ((particles_prob['update_type']=='particles_prob_learner_before_demo') | (particles_prob['update_type']=='particles_prob_learner_after_demo') | \
                                    (particles_prob['update_type']=='particles_prob_learner_before_test') | (particles_prob['update_type']=='particles_prob_learner_after_feedback'))]
        
        f_p, ax_p = plt.subplots(nrows=2, figsize=(15, 10))

        player_id_list = plot_data['player_id'].unique()

        int_fill_flag=False
        for player_id in player_id_list:
            plot_data_player = plot_data[plot_data['player_id']==player_id]
            interaction_number = np.arange(1, len(plot_data_player)+1)
            plot_data_player['interaction_number'] = interaction_number
            plot_data_player = plot_data_player.reset_index(drop=True)

            # plot
            sns.lineplot(plot_data_player, x = 'interaction_number', y = 'kc_prob', ax=ax_p[0]) 
            sns.lineplot(plot_data_player, x = 'interaction_number', y = 'BEC_prob', ax=ax_p[1])


            if not int_fill_flag:
                for id, row in plot_data_player.iterrows():
                    print('id:', id)
                    if id > 0:
                        # plt.axvline(x=row['combined_demo_id'], color='red', linestyle='--')
                        ax_p[0].axvspan(plot_data_player['interaction_number'].iloc[id-1], row['interaction_number'], alpha=0.2, color=color_dict[row['update_type']])
                        ax_p[0].text(row['interaction_number']-0.5, 0.3, row['update_type'], rotation=90, fontsize=10, weight="bold")

                        ax_p[1].axvspan(plot_data_player['interaction_number'].iloc[id-1], row['interaction_number'], alpha=0.2, color=color_dict[row['update_type']])
                        ax_p[1].text(row['interaction_number']-0.5, 0.3, row['update_type'], rotation=90, fontsize=10, weight="bold")
                int_fill_flag = True

        filename = 'Study ' + str(study_id) + '. Run ' + str(run_id) + '. Team mix ' + str(plot_data['team_composition'].iloc[0]) + '. Demo strategy ' + str(plot_data['demo_strategy'].iloc[0])
        f_p.suptitle(filename)


        # plt.show()
        plt.savefig('models/augmented_taxi2/' + filename + '.png')



    # plt.show()
            
    # plt.savefig('models/augmented_taxi2/BEC_knowledge_level_' + know_id + '.png')
    # plt.close()
    ########
        
    # ## plot knowledge level for all conditions
    # f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # col_id = 0
    # for know_id in know_list:  
    #     plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) ]
    #     sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for a team mix: ' + str(team_mix_cond))
    #     # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #     col_id += 1 
    # # plt.show()
    # ########

    # ## plot knowledge level for all team composition
    # for team_mix_cond in team_mix:
    #     f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #     col_id = 0
    #     for know_id in know_list:  
    #         plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) &(team_knowledge_level_long['team_composition']==str(team_mix_cond))]
    #         sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for a team mix: ' + str(team_mix_cond))
    #         # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #         col_id += 1 
    #     # plt.show()

    # # plot knowledge for demo strategy
    # for demo_id in demo_list:
    #     f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #     col_id = 0
    #     for know_id in know_list: 
    #         plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) &(team_knowledge_level_long['demo_strategy']==str(demo_id))]
    #         sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for demo strategy: ' + str(demo_id))
    #         # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #         col_id += 1 
    #     # plt.show()

    # plt.show()
    #############

################################################################################




def analyze_concept_data(file):

    # pd.read_csv(file).describe(include='all').to_csv('models/augmented_taxi2/descriptives_concept_data.csv') 
    team_mix_var = [0, 1, 1]

    concept_data = pd.read_csv(file)

    

    concept_data_plot = concept_data[concept_data['team_mix']==str(team_mix_var)]

    concept_data_plot_N_int_melt = pd.melt(concept_data_plot, id_vars=['run_no', 'demo_strategy', 'team_mix'], value_vars=['N_int_concept_1', 'N_int_concept_2', 'N_int_concept_3'], var_name='Concept_id', value_name='N_int_concept')
    concept_data_plot_max_int_melt = pd.melt(concept_data_plot, id_vars=['run_no', 'demo_strategy', 'team_mix'], value_vars=['Int_end_id_concept_1', 'Int_end_id_concept_2', 'Int_end_id_concept_3'], var_name='Concept_id', value_name='Max_int_concept')

    
    # Create a stacked bar chart with error bars

    ## print(concept_data_plot)

    # concept_ids = [0, 1, 2]
    # fig, ax = plt.subplots()
    # Plot the stacked bars
    # bar_width = 0.5

    # print(concept_data[(concept_data['team_mix']==team_mix_var).all(), 'N_int_concept_1'].mean())
    # print(concept_data[concept_data['team_mix']==team_mix_var, 'N_int_concept_1'].sem())

    # bar1 = ax.barh(concept_ids[0],  concept_data_plot['N_int_concept_1'].mean(), bar_width, label='Concept ' + str(concept_ids[0]), xerr= concept_data_plot['N_int_concept_1'].sem(), capsize=5)
    # bar2 = ax.barh(concept_ids[1],  concept_data_plot['N_int_concept_2'].mean(), bar_width, left=concept_data_plot['N_int_concept_1'].mean(), label='Concept' + str(concept_ids[1]), xerr= concept_data_plot['N_int_concept_1'].sem(), capsize=5)
    # bar3 = ax.barh(concept_ids[2],  concept_data_plot['N_int_concept_3'].mean(), bar_width, left=concept_data_plot['N_int_concept_1'].mean()+concept_data_plot['N_int_concept_2'].mean(), label='Concept' + str(concept_ids[2]), xerr= concept_data_plot['N_int_concept_2'].sem(), capsize=5)

    # ax.set_xlabel('Knowledge component / Concept')
    # ax.set_ylabel('Number of interactions')
    # ax.set_title('Number of interactions to learn concepts')


    # Box plot
    sns.boxplot(data = concept_data_plot_max_int_melt, x = 'Max_int_concept', y = 'Concept_id', orient = 'h').set(title='Number of interactions to learn concepts')


    plt.show()
############


def analyze_human_response_simulation(path, files, file_prefix, file_to_avoid):


    response_data = pd.DataFrame()
    const_prob_data = pd.DataFrame()

    
    for file in files:
        
        if file_prefix in file and '.csv' in file and (file_to_avoid == '' or file_to_avoid not in file):
            print('Reading file: ', file)
            sim_vars = pd.read_csv(path + '/' + file)
            condition = file.replace('.csv', '')
            condition = condition.replace(file_prefix + '_', '')
            condition = condition.replace('_set_', '')
            condition = ''.join([i for i in condition if not i.isdigit()])
            print('condition: ', condition)

            set_id = file.replace('.csv', '')
            set_id = set_id.replace(file_prefix + '_', '')
            set_id = set_id.replace(condition + '_', '')
            set_id = set_id.replace('set_', '')

            for i in range(len(sim_vars)):
                # response data
                response_data_dict = {}

                response_data_dict['set_id'] = set_id
                response_data_dict['update_id'] = sim_vars['update_id'][i]
                response_data_dict['member_id'] = sim_vars['member_id'][i]
                response_data_dict['learning_factor'] = sim_vars['learning_factor'][i]
                response_data_dict['cluster_id'] = sim_vars['cluster_id'][i]
                response_data_dict['point_probability'] = sim_vars['point_probability'][i]
                response_data_dict['response_type'] = sim_vars['response'][i]
                response_data_dict['skip_model_flag'] = sim_vars['skip_model'][i]
                response_data_dict['constraint_flag'] = sim_vars['constraint_flag'][i]
                prob_all_cnsts_combo = sim_vars['particles_learner_prob_test_history'][i]
                prob_all_cnsts_combo = prob_all_cnsts_combo.strip('[]')

                response_data_dict['particles_prob'] = np.round(float(prob_all_cnsts_combo.split(', ')[-1]), 3)
                # response_data_dict['condition'] = condition

                response_data = response_data.append(response_data_dict, ignore_index=True)


                ###########
                cnsts = sim_vars['constraints'][i]
                # print('cnsts: ', cnsts)
                chars_to_remove = ['(', ')', 'a','r','y']
                cnsts = ''.join([char for char in cnsts if char not in chars_to_remove])
                cnsts = cnsts.split('], ')
                # print('cnsts: ', cnsts)
                
                prob_initial = sim_vars['prob_initial_history'][i]
                # print('prob_initial: ', prob_initial)
                chars_to_remove = ['[', ']']
                prob_initial = ''.join([char for char in prob_initial if char not in chars_to_remove])
                prob_initial = prob_initial.split(', ')
                # print('prob_initial: ', prob_initial)

                prob_reweight = sim_vars['prob_reweight_history'][i]
                # print('prob_reweight: ', prob_reweight)
                chars_to_remove = ['[', ']']
                prob_reweight = ''.join([char for char in prob_reweight if char not in chars_to_remove])
                prob_reweight = prob_reweight.split(', ')
                # print('prob_reweight: ', prob_reweight)

                prob_resample = sim_vars['prob_resample_history'][i]
                # print('prob_resample: ', prob_resample)
                chars_to_remove = ['[', ']']
                prob_resample = ''.join([char for char in prob_resample if char not in chars_to_remove])
                prob_resample = prob_resample.split(', ')
                # print('prob_resample: ', prob_resample) 


                for cnst_id in range(len(cnsts)):
                    cnst_prob_dict = {}
                    cnst_prob_dict['set_id'] = set_id
                    cnst_prob_dict['update_id'] = sim_vars['update_id'][i]
                    cnst_prob_dict['member_id'] = sim_vars['member_id'][i]
                    cnst_prob_dict['cluster_id'] = sim_vars['cluster_id'][i]
                    cnst_prob_dict['learning_factor'] = sim_vars['learning_factor'][i]
                    cnst_prob_dict['response_type'] = sim_vars['response'][i]
                    cnst_prob_dict['skip_model_flag'] = sim_vars['skip_model'][i]
                    # cnst_prob_dict['condition'] = sim_vars['condition'][i]
                    cnst_prob_dict['constraint'] = cnsts[cnst_id]
                    cnst_prob_dict['prob_initial'] = np.round(float(prob_initial[cnst_id]), 3)
                    cnst_prob_dict['prob_reweight'] = np.round(float(prob_reweight[cnst_id]), 3)
                    cnst_prob_dict['prob_resample'] = np.round(float(prob_resample[cnst_id]), 3)

                    const_prob_data = const_prob_data.append(cnst_prob_dict, ignore_index=True)



    return response_data, const_prob_data
#############


def check_pf_particles_sampling(path, files, file_prefix):

    pos_df = pd.DataFrame()
    for file in files:
        if file_prefix in file and '.csv' in file:
            print('Reading file: ', file)
            sim_vars = pd.read_csv(path + '/' + file)

            set_id = file.replace('.csv', '')
            set_id = set_id.replace(file_prefix + '_', '')
            set_id = set_id.replace('set_', '')
            set_id, player_id = set_id.split('_')

            for i in range(len(sim_vars)):
                pos_dict = {}
                pos_dict['set_id'] = set_id
                pos_dict['player_id'] = player_id
                pos_dict['position'] = np.fromstring(sim_vars['Position'][i].strip('[]'), sep=' ')
                pos_df = pos_df.append(pos_dict, ignore_index=True)



            # pos_df['position'] = sim_vars['Position'].apply(lambda x: np.fromstring(x, sep=' '))

    
    print('pos_df: ', pos_df)
    pos_df.to_csv(path + '/pos_df.csv')
    
    pos_diff_df = pd.DataFrame()
    ## check if sampled particles are similar

    for set_id in pos_df['set_id'].unique():
        if set_id != '1':
            print('Reading set_id: ', set_id)
            pos_diff_dict = {}
            for player_id in pos_df['player_id'].unique():
                pos_df_set1 =  pos_df_data = pos_df[(pos_df['set_id'] == '1') & (pos_df['player_id'] == player_id)]
                pos_df_data = pos_df[(pos_df['set_id'] == set_id) & (pos_df['player_id'] == player_id)]
                print('pos_df_data: ', pos_df_data)
                print('pos_df_set1: ', pos_df_set1)

                print(type(pos_df_set1['position'].iloc[0]))
                pos_diff_sum = 0
                for i in range(len(pos_df_set1)):
                    pos_diff_sum += np.linalg.norm(pos_df_data['position'].iloc[i] - pos_df_set1['position'].iloc[i])
                
                print('pos_diff_sum: ', pos_diff_sum)

                pos_diff_dict['set_id'] = set_id
                pos_diff_dict['player_id'] = player_id
                pos_diff_dict['pos_diff'] = pos_diff_sum

                pos_diff_df = pos_diff_df.append(pos_diff_dict, ignore_index=True)

    print('pos_diff_df: ', pos_diff_df)
    pos_diff_df.to_csv(path + '/pos_diff_df.csv')
############


def plot_prob_ind_run(path, file):

    study_data = pd.read_pickle(path + '/' + file)
    particles_prob_learner_demo = study_data['particles_prob_learner_before_test']

    particles_prob_learner_demo_df = pd.DataFrame()

    # print('particles_prob_learner_demo: ', particles_prob_learner_demo)

    for i, row in particles_prob_learner_demo.iteritems():
        print(row)
        # row['test_id'] = i
        for i in range(len(row)):
            data_dict = copy.deepcopy(row[i])
            print('data_dict: ', data_dict)
            data_dict['test_id'] = i+1

            particles_prob_learner_demo_df = particles_prob_learner_demo_df.append(data_dict, ignore_index=True)

    particles_prob_learner_demo_df.to_csv(path + '/particles_prob_learner_demo_df.csv')

    print('particles_prob_learner_demo_df: ', particles_prob_learner_demo_df)

    f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # col_id = 0
    for col_id in range(3):  
        member_id = 'p' + str(col_id+1)
        sns.lineplot(particles_prob_learner_demo_df, x=particles_prob_learner_demo_df.index, y = member_id, hue = 'test_id', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Probability of correct test response '+ member_id)
        # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
        # col_id += 1 
    plt.show()
    ########




def analyze_individual_runs(path, file):


    # interesting interactions
    # interesting_interactions = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    

    # fixed params
    max_learning_factor = 0.9
    team_learning_rate =  np.hstack((0.05*np.ones([params.team_size, 1]), 0*np.ones([params.team_size, 1])))
    default_learning_factor_teacher = 0.8
    viz_flag = True

    study_data = pd.read_pickle(path + '/' + file)

    ### plot particle probability for learner
    if 'particles_prob_learner_demo' in study_data.columns:
        particles_prob_learner_demo = study_data['particles_prob_learner_demo']
    elif 'particles_prob_learner_demos' in study_data.columns:
        particles_prob_learner_demo = study_data['particles_prob_learner_demos']
    elif 'particles_prob_learner_before_test' in study_data.columns:
        particles_prob_learner_demo = study_data['particles_prob_learner_before_test']
    particles_prob_learner_demo_df = pd.DataFrame()

    for i, row in particles_prob_learner_demo.iteritems():
        # print(row)
        particles_prob_learner_demo_df = particles_prob_learner_demo_df.append(row, ignore_index=True)
    

    ### particle probability of teacher
    if 'particles_prob_teacher_demo' in study_data.columns:
        particles_prob_teacher_demo = study_data['particles_prob_teacher_demo']
    elif 'particles_prob_teacher_demos' in study_data.columns:
        particles_prob_teacher_demo = study_data['particles_prob_teacher_demos']
    elif 'particles_prob_teacher_before_test' in study_data.columns:
        particles_prob_teacher_demo = study_data['particles_prob_teacher_before_test']
    particles_prob_teacher_demo_df = pd.DataFrame()

    for i, row in particles_prob_teacher_demo.iteritems():
        # print(row)
        particles_prob_teacher_demo_df = particles_prob_teacher_demo_df.append(row, ignore_index=True)



    # # plot probability
    # f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))






    ### constraints
    demo_constraints = study_data['unit_constraints']
    test_constraints = study_data['test_constraints']
    test_responses_team = study_data['test_constraints_team']
    loop_count = study_data['loop_count']
    team_response_models = study_data['team_response_models']
    particles_team_teacher_actual = study_data['particles_team_teacher_after_demos']
    particles_team_learner_actual = study_data['particles_team_learner_after_demos']



    ### initialize teacher and learner particle filters
    initial_team_learning_factor = study_data['initial_team_learning_factor'].iloc[0]
    team_learning_factor = copy.deepcopy(initial_team_learning_factor)
    team_prior, teacher_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, default_learning_factor_teacher, team_prior = params.team_prior)
    learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, default_learning_factor_teacher, team_learning_factor = team_learning_factor, team_prior = params.team_prior, pf_flag='learner')

    prob_learner_after_demo = pd.DataFrame()
    prob_teacher_after_demo = pd.DataFrame()
    prob_learner_before_test = pd.DataFrame()
    prob_learner_after_test = pd.DataFrame()
    prob_teacher_before_test = pd.DataFrame()
    prob_teacher_after_test = pd.DataFrame()

    # run pf updates only for individual knowledge
    for loop_id in range(len(loop_count)):
        
        # print('loop_id: ', loop_id+1, 'interesting_interactions:', interesting_interactions)
        # if loop_id+1 in interesting_interactions:
        #     viz_flag = True
        # else:
        #     viz_flag = False


        demo_loop = demo_constraints[loop_id]
        test_loop = test_constraints[loop_id]
        test_responses_loop = test_responses_team[loop_id]
        team_models = team_response_models[loop_id]
        particles_team_teacher_actual_loop = particles_team_teacher_actual[loop_id]
        particles_team_learner_actual_loop = particles_team_learner_actual[loop_id]
        
        print('loop_id: ', loop_id)
        print('demo_loop: ', demo_loop)
        print('test_loop: ', test_loop)

        # teacher and learner update for demonstrations
        # for demo_id in range(len(demo_loop)):
        # dem_cnsts = demo_loop[demo_id]
        dem_cnsts = demo_loop
        demo_id = 0
        print('dem_cnsts: ', dem_cnsts)
        
        prob_learner_after_demo_dict = {'loop_id': loop_id+1, 'demo_id': demo_id, 'demo_constraints': dem_cnsts}
        prob_teacher_after_demo_dict = {'loop_id': loop_id+1, 'demo_id': demo_id, 'demo_constraints': dem_cnsts}
        for p_id in range(params.team_size):
            member_id = 'p' + str(p_id+1)
            
            plot_title = 'Interaction No.' + str(loop_id +1) + '. Teacher belief update after demo for member ' + member_id
            teacher_pf[member_id].update(dem_cnsts, plot_title = plot_title, viz_flag = viz_flag)
            teacher_pf[member_id].calc_particles_probability(dem_cnsts)
            
            plot_title = 'Interaction No.' + str(loop_id +1) + '. Learner belief update after demo for member ' + member_id
            learner_pf[member_id].update(dem_cnsts, learning_factor = team_learning_factor[p_id], plot_title = plot_title, viz_flag = viz_flag)
            learner_pf[member_id].calc_particles_probability(dem_cnsts)
            print('learner_pf_particles_prob_correct: ', learner_pf[member_id].particles_prob_correct)

            
            prob_learner_after_demo_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_teacher_after_demo_dict[member_id] = teacher_pf[member_id].particles_prob_correct

        # update probability dataframes
        prob_learner_after_demo = prob_learner_after_demo.append(prob_learner_after_demo_dict, ignore_index=True)
        prob_teacher_after_demo = prob_teacher_after_demo.append(prob_teacher_after_demo_dict, ignore_index=True)


        
        prob_learner_before_test_dict = {'loop_id': loop_id+1, 'test_constraints': test_loop}
        prob_teacher_before_test_dict = {'loop_id': loop_id+1, 'test_constraints': test_loop}
        prob_learner_after_test_dict = {'loop_id': loop_id+1,  'test_constraints': test_loop}
        prob_teacher_after_test_dict = {'loop_id': loop_id+1, 'test_constraints': test_loop}

        test_loop_extended = []
        for tst_id in range(len(test_loop)):
            test_loop_extended.extend(test_loop[tst_id])
        print('test_loop_extended: ', test_loop_extended, 'test_loop: ', test_loop)

        # plot sampled models
        if viz_flag:
            plot_title = 'Interaction No.' + str(loop_id +1) + '. Human models for test for member ' + member_id
            sim_helpers.plot_sampled_models(learner_pf, test_loop_extended, team_models, 1, plot_title = plot_title)
            plot_title = 'Interaction No.' + str(loop_id +1) + '. Actual learner PF after test update. Human models for test for member ' + member_id
            sim_helpers.plot_sampled_models(particles_team_learner_actual_loop, test_loop_extended, team_models, 1, plot_title = plot_title)
        
        
        # teacher and learner update for tests
        for p_id in range(params.team_size):
            member_id = 'p' + str(p_id+1)

            all_test_responses = []
            print('test_responses_loop: ', test_responses_loop)
            for test_id in range(len(test_responses_loop[p_id])):
                all_test_responses.append(test_responses_loop[p_id][test_id])
                print('test_cnsts: ', all_test_responses)



            # test_response = test_responses_team[p_id][test_id]     

            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            learner_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_learner_before_test_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_teacher_before_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct


            # update based on test responses
            print('Member: ', member_id, ' test_response: ', all_test_responses)

            plot_title = 'Interaction No.' + str(loop_id +1) + '. Teacher belief update after test for member ' + member_id
            teacher_pf[member_id].update(all_test_responses, plot_title = plot_title, viz_flag = viz_flag)  # note that test responses are ordered based on member at high level
            
            plot_title = 'Interaction No.' + str(loop_id +1) + '. Learner belief update after test for member ' + member_id
            learner_pf[member_id].update(all_test_responses, learning_factor = team_learning_factor[p_id], plot_title = plot_title, viz_flag = viz_flag)

            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            learner_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_learner_after_test_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_teacher_after_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct

            # update learning parameter
            response_type_history =[]
            for test_id in range(len(test_responses_loop[p_id])):
                test_response = test_responses_loop[p_id][test_id]
                if (test_response == test_loop_extended[test_id]).all():
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
                    response_type_history.append('correct')
                else:
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)
                    response_type_history.append('incorrect')

            var_name = 'response_type_' + member_id

            prob_learner_after_test_dict[var_name] = response_type_history
            prob_teacher_after_test_dict[var_name] = response_type_history
            prob_learner_before_test_dict[var_name] = response_type_history
            prob_teacher_before_test_dict[var_name] = response_type_history


            ## update probability dataframes
            prob_learner_before_test = prob_learner_before_test.append(prob_learner_before_test_dict, ignore_index=True)
            prob_teacher_before_test = prob_teacher_before_test.append(prob_teacher_before_test_dict, ignore_index=True)
            prob_learner_after_test = prob_learner_after_test.append(prob_learner_after_test_dict, ignore_index=True)
            prob_teacher_after_test = prob_teacher_after_test.append(prob_teacher_after_test_dict, ignore_index=True)


            ## plot actual learner PF
            if viz_flag:
                plot_title = 'Interaction No.' + str(loop_id +1) + '. Actual learner PF after test update. PF Transition for ' + member_id
                team_helpers.visualize_transition(all_test_responses, particles_team_learner_actual_loop[member_id], params.mdp_class, params.weights['val'], text = plot_title)


    
    filename = path + '/' + file.split('.')[0]

    prob_learner_after_demo.to_csv(filename + '_prob_learner_after_demo.csv')
    prob_teacher_after_demo.to_csv(filename + '_prob_teacher_after_demo.csv')
    prob_learner_before_test.to_csv(filename + '_prob_learner_before_test.csv')
    prob_teacher_before_test.to_csv(filename + '_prob_teacher_before_test.csv')
    prob_learner_after_test.to_csv(filename + '_prob_learner_after_test.csv')
    prob_teacher_after_test.to_csv(filename + '_prob_teacher_after_test.csv')



    return 1
####################################################



def run_sensitivity_analysis(path, files, file_prefix_list, runs_to_exclude_list=[], runs_to_analyze_list = [], vars_filename_prefix = ''):

    
    def convert_to_dict(sim_vars, param_varied=None, parameter_combinations=None):

        # list of params
        params_to_study = {'learning_factor_low': np.linspace(0.6, 0.8, 11), 'learning_factor_high': np.linspace(0.7, 0.9, 11), 'learning_rate': np.linspace(0.0, 0.2, 11), \
                           'max_learning_factor': np.linspace(0.85, 1.0, 11), 'default_learning_factor_teacher': np.linspace(0.7, 0.9, 11)}  # list of parameter values that were used for the sensitivity analysis

        demo_strategy = sim_vars['demo_strategy'].iloc[0]
        team_composition = sim_vars['team_composition'].iloc[0]

        study_id = int(sim_vars['study_id'].iloc[0])
        run_id = sim_vars['run_no'].iloc[0] 

        max_learning_factor = np.round(sim_vars['max_learning_factor'].iloc[0],2)
        team_learning_factor = sim_vars['initial_team_learning_factor'].iloc[0]
        # team_learning_rate = sim_vars['team_learning_rate'].iloc[0]


        learning_factor_high_learner = []
        learning_factor_low_learner = []
        for i in range(len(team_composition)):
            if team_composition[i] == 0:
                learning_factor_low_learner = np.round(team_learning_factor[i],2)
            elif team_composition[i] == 2:
                learning_factor_high_learner = np.round(team_learning_factor[i],2)

        # for i in range(len(team_learning_rate)):
        #     team_learning_rate[i] = np.round(team_learning_rate[i], 2)
        

        max_loop_count = sim_vars['loop_count'].iloc[-1]
        # bec_final = sim_vars['BEC_knowledge_level'][len(sim_vars)-1]

        sensitivity_data_dict = {}

        sensitivity_data_dict['demo_strategy'] = demo_strategy
        sensitivity_data_dict['team_composition'] = team_composition
        sensitivity_data_dict['study_id'] = study_id
        sensitivity_data_dict['run_id'] = run_id
        
        sensitivity_data_dict['max_learning_factor'] = max_learning_factor
        sensitivity_data_dict['learning_factor_high_learner'] = learning_factor_high_learner
        sensitivity_data_dict['learning_factor_low_learner'] = learning_factor_low_learner

        sim_params = sim_vars['params_conditions'].iloc[0]

        # for parameters that were missed to directly be recorded
        if parameter_combinations is not None:
            # sensitivity_data_dict['teacher_learning_factor'] = np.round(parameter_combinations[study_id-1][4], 2)
            # sensitivity_data_dict['learning_rate'] = np.round(parameter_combinations[study_id-1][2],2)
            sensitivity_data_dict['teacher_learning_factor'] = np.round(sim_params[4], 2)
            sensitivity_data_dict['learning_rate'] = np.round(sim_params[2],2)
        else:
            if param_varied == 'default_learning_factor_teacher':
                # sensitivity_data_dict['teacher_learning_factor'] = params_to_study[param_varied][study_id-1]
                sensitivity_data_dict['teacher_learning_factor'] = np.round(sim_params[4],2)
            else:
                print('teacher factor center value: ', params_to_study['default_learning_factor_teacher'][5])
                sensitivity_data_dict['teacher_learning_factor'] = params_to_study['default_learning_factor_teacher'][5] # center value!

            if param_varied == 'learning_rate':
                # sensitivity_data_dict['learning_rate'] = params_to_study[param_varied][study_id-1]
                sensitivity_data_dict['learning_rate'] = np.round(sim_params[2],2)
            else:
                print('learning rate center value: ', params_to_study['learning_rate'][5])
                sensitivity_data_dict['learning_rate'] = params_to_study['learning_rate'][5]
            
            
        if param_varied is not None:
            sensitivity_data_dict['param_varied'] = param_varied
        

        # sensitivity_data_dict['parameter_combinations'] = parameter_combinations[study_id-1]
        # sensitivity_data_dict['team_learning_rate'] = team_learning_rate

        sensitivity_data_dict['max_loop_count'] = int(max_loop_count)
        # sensitivity_data_dict['bec_final'] = bec_final



        # ##  each test has probability separately
        # for prob_dict in sim_vars[var_name][i]:
        ## joint test has probability separately
        team_particles_probability_dict = {}
        for key, val in sim_vars['particles_prob_learner_after_feedback'].iloc[-1].items():
            # print('key: ', key, ' val: ', val)
            team_particles_probability_dict[key] = float(val)

        learner_BEC_prob = []
        for p_id, player in enumerate(team_particles_probability_dict):
            particles_learner_current = sim_vars['particles_team_learner_final'][i][player]
            min_BEC_constraints = sim_vars['min_BEC_constraints'][i]
            particles_learner_current.calc_particles_probability(min_BEC_constraints)
            BEC_prob = particles_learner_current.particles_prob_correct

            learner_BEC_prob.append(BEC_prob)

        sensitivity_data_dict['average_BEC_prob'] = np.round(float(np.mean(learner_BEC_prob)),4)


        return sensitivity_data_dict

    #####################

    try: 
        with open(path + '/' + vars_filename_prefix + '_sensitivity_data.pickle', 'rb') as f:
            sensitivity_data = pickle.load(f)

    except:
        # read parameter combinations
        # with open('data/simulation/sim_experiments/sensitivity_analysis/w_feedback/param_combinations_forte.pickle', 'rb') as f:
        #         parameter_combinations = pickle.load(f)

        
        # params_pd = pd.DataFrame()
        # for i in range(len(parameter_combinations)):
        #     params_dict = {'study_id': i+1, 'learning_factor_low': parameter_combinations[i][0], 'learning_factor_high': parameter_combinations[i][1], 'learning_rate': parameter_combinations[i][2], 'max_learning_factor': parameter_combinations[i][3], 'teacher_learning_factor': parameter_combinations[i][4]}
        #     print(params_dict)
        #     params_pd = params_pd.append(params_dict, ignore_index=True)

        # params_pd.to_csv('data/simulation/sim_experiments/sensitivity_analysis/w_feedback/param_combinations_forte.csv')

        # print('parameter_combinations:', parameter_combinations)



    #     #############
        run_no = 1
        sensitivity_data_list = []
        for file in files:

            # check if file is a valid file
            for file_prefix in file_prefix_list:
                if file_prefix in file and '.pickle' in file:
                    run_file_flag = True
                    break
                else:
                    run_file_flag = False

            # check if file needs to be excluded
            for runs_to_exclude in runs_to_exclude_list:
                if runs_to_exclude in file:
                    run_file_flag = False
                    break
                    
            
            if run_file_flag:
                with open(path + '/' + file, 'rb') as f:
                    sim_vars = pickle.load(f)
                
                print('Reading file: ', file)
                print('sim_vars: ', sim_vars)

                if len(sim_vars) > 0:
                    bec_final = sim_vars['BEC_knowledge_level'][len(sim_vars)-1]
                    print('bec_final: ', bec_final)
                    # check if learning was completed
                    learning_complete = True
                    for k_type, k_val in bec_final.items():
                        if k_val[0] != 1:
                            learning_complete = False
                            run_file_flag = False
                            break
                else:
                    run_file_flag = False

                print('learning_complete: ', learning_complete)
        
            
            if run_file_flag:

                print('Running file: ', file)
                
                # check if there are multiple runs in the same file
                loop_count_var = sim_vars['loop_count']
                run_change_idx = [idx for idx in range(len(loop_count_var)-1) if loop_count_var[idx] > loop_count_var[idx+1]]

                if 'lfh' in file:
                    param_varied = 'learning_factor_high'
                elif 'lfl' in file:
                    param_varied = 'learning_factor_low'
                elif 'lr' in file:
                    param_varied = 'learning_rate'
                elif 'mlf' in file:
                    param_varied = 'max_learning_factor'
                elif 'lft' in file or 'tfl' in file:
                    param_varied = 'default_learning_factor_teacher'
                else:
                    RuntimeError('Parameter varied not found in file name')

    #             print('run_change_idx: ', run_change_idx)
                if len(run_change_idx) > 0:

                    for run_idx in range(2):
                        if run_idx == 0:
                            run_sim_vars = sim_vars.iloc[:run_change_idx[0]+1]
                        else:
                            run_sim_vars = sim_vars.iloc[run_change_idx[0]+2:]
                            
                        # reset index
                        run_sim_vars = run_sim_vars.reset_index(drop=True)


                        bec_final = run_sim_vars['BEC_knowledge_level'][len(run_sim_vars)-1]
                        # check if learning was completed
                        learning_complete = True
                        for k_type, k_val in bec_final.items():
                            if k_val[0] != 1:
                                learning_complete = False
                                break
                        
                        if learning_complete:
                            # sensitivity_data_dict = convert_to_dict(run_sim_vars, parameter_combinations)
                            sensitivity_data_dict = convert_to_dict(run_sim_vars, param_varied=param_varied)
                            sensitivity_data_list.append(sensitivity_data_dict)
                else:

                    bec_final = sim_vars['BEC_knowledge_level'][len(sim_vars)-1]
                    # check if learning was completed
                    learning_complete = True
                    for k_type, k_val in bec_final.items():
                        if k_val[0] != 1:
                            learning_complete = False
                            break
                
                    if learning_complete:
                        # sensitivity_data_dict = convert_to_dict(sim_vars, parameter_combinations)
                        sensitivity_data_dict = convert_to_dict(sim_vars, param_varied=param_varied)
                        sensitivity_data_list.append(sensitivity_data_dict)

        sensitivity_data = pd.DataFrame(sensitivity_data_list)
        sensitivity_data.to_csv(path + '/' + vars_filename_prefix + '_sensitivity_data.csv')
        with open(path + '/' + vars_filename_prefix + '_sensitivity_data.pickle', 'wb') as f:
            pickle.dump(sensitivity_data, f)
    #############
    print(sensitivity_data)

    ##############
    # plot

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    row_id, col_id = 0, 0
    for i, param_varied in enumerate(['learning_factor_low_learner', 'learning_factor_high_learner', 'learning_rate']):

        # if param_varied == 'learning_factor_low_learner':
        #     param_label = 'Naive learner $\mathregular{\beta_0}$'
        # elif param_varied == 'learning_factor_high_learner':
        #     param_label = 'Proficient learner $\mathregular{\beta_1}$'
        # elif param_varied == 'learning_rate':
        #     param_label = 'Learning rate $\mathregular{\delta\, \beta_i}'
        # elif param_varied == 'teacher_learning_factor':
        #     param_label = 'Teacher $\mathregular{\beta}$'
        
        sns.lineplot(data=sensitivity_data, x=param_varied, y='max_loop_count', ax=axs[0, col_id], errorbar='se').set(title =  param_varied)
        sns.regplot(data=sensitivity_data, x=param_varied, y='max_loop_count', ax=axs[0, col_id]).set(title =  param_varied)
        sns.regplot(data=sensitivity_data, x=param_varied, y='average_BEC_prob', ax=axs[1, col_id]).set(title=  param_varied)
        sns.lineplot(data=sensitivity_data, x=param_varied, y='average_BEC_prob', ax=axs[1, col_id], errorbar='se').set(title=  param_varied)
        col_id += 1



    plt.show()




    
    # # calculate SOBOL indices
    # problem = {
    #     'num_vars': 5,
    #     'names': ['learning_factor_low', 'learning_factor_high', 'learning_rate', 'max_learning_factor', 'default_learning_factor_teacher'],
    #     'bounds': [[0.6, 0.7], [0.75, 0.85], [0.0, 0.1], [0.85, 1.0], [0.6, 0.9]]
    # }

    # # Perform Sobol sensitivity analysis
    # Si = sobol.analyze(problem, sensitivity_data['max_loop_count'], print_to_console=True)

    # # Print the results
    # print(Si)

    # Using .dtypes attribute
    # column_data_types = sensitivity_data.dtypes
    # print("Data Types of Columns (using .dtypes attribute):\n", column_data_types)


    # sensitivity_data_short = sensitivity_data[['max_loop_count', 'max_learning_factor', 'learning_factor_high_learner', 'teacher_learning_factor', 'learning_rate']]


    # # Compute partial rank correlation coefficients
    # partial_corr_results = {}
    # var_list = ['max_learning_factor', 'learning_factor_high_learner', 'teacher_learning_factor', 'learning_rate']
    # for var in var_list:
    #     covars = [var2 for var2 in var_list if var2 != var]
                
    #     partial_corr_results[var] = partial_corr(data=sensitivity_data_short, x='max_loop_count', y=var, covar=covars, method='pearson')

    # # Print the results
    # print("Partial Rank Correlation Coefficients:")
    # print(partial_corr_results)

    # fig, ax = plt.subplots(1, len(var_list), figsize=(10, 6))

    # for i, var in enumerate(var_list):
    #     ax[i].scatter(sensitivity_data_short[var], sensitivity_data_short['max_loop_count'])
    #     ax[i].set_xlabel(var)
    #     ax[i].set_ylabel('N_interactions')
    #     ax[i].set_title('Partial Rank Correlation Coefficient')

    # plt.show()

    # # Calculate correlation coefficients
    # correlation_matrix = sensitivity_data_short.corr()

    # # Plot heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Heatmap between Input Parameters and Output')
    # plt.show()



############################


def debug_calc_prob_mass_correct_side(constraints, particles, team_learning_factor):

    prob_mass_correct_side_constraints = {}
    for i in range(params.team_size):
        member_id = 'p' + str(i+1)
        lf_var = member_id + '_lf'
        particles[member_id].calc_particles_probability(constraints)
        prob_mass_correct_side_constraints[lf_var] = team_learning_factor[i]
        prob_mass_correct_side_constraints[member_id] = particles[member_id].particles_prob_correct

    return prob_mass_correct_side_constraints


    

def simulate_individual_runs_w_feedback(params, path, file, run_id, feedback_flag = True, review_flag = True, viz_flag = False, vars_filename_prefix = '', args=None):
    ### read trial data and simulate particle filter updates ####

    # debug params
    interesting_interactions = []

    # sim params
    learner_update_type = 'no_noise'

    # load trial data
    trial_data = pd.read_pickle(path + '/' + file)
    
    # trial conditions
    demo_strategy = trial_data['demo_strategy'].iloc[0]

    if args is None:
        team_composition = trial_data['team_composition'].iloc[0]

        # trial params
        max_learning_factor = trial_data['max_learning_factor'].iloc[0]
        initial_team_learning_factor = trial_data['initial_team_learning_factor'].iloc[0]
        team_learning_factor_delta = np.hstack((0.05*np.ones([params.team_size, 1]), 0*np.ones([params.team_size, 1])))
        print('initial_team_learning_factor: ', initial_team_learning_factor)

    else:
        team_composition, initial_team_learning_factor, team_learning_factor_delta = args
        max_learning_factor = params.max_learning_factor

    teacher_learning_factor = np.ones([params.team_size, 1])*params.default_learning_factor_teacher


    team_learning_factor = copy.deepcopy(initial_team_learning_factor)
    
    
    # simulated filename
    vars_filename_prefix = vars_filename_prefix + '_' + file.split('.')[0] + '_' + str(run_id) 


    # trial variables
    demo_constraints = trial_data['unit_constraints']
    min_demo_constraints = trial_data['min_KC_constraints']
    test_constraints = trial_data['test_constraints']
    test_responses_team = trial_data['test_constraints_team']
    loop_count = trial_data['loop_count']
    team_response_models = trial_data['team_response_models']
    kc_id_list = trial_data['knowledge_comp_id']
    # team_learning_factor = trial_data['team_learning_factor']

    # actual particles and probabilities (for comparison)
    # teacher_pf_actual_after_demo = trial_data['particles_team_teacher_after_demos']
    # learner_pf_actual_after_demo = trial_data['particles_team_learner_after_demos']
    # teacher_pf_actual_after_test = trial_data['particles_team_teacher_final']
    # learner_pf_actual_after_test = trial_data['particles_team_learner_final']
    # prob_pf_actual_teacher_before_demo = trial_data['particles_prob_teacher_before_demo']
    # prob_pf_actual_learner_before_demo = trial_data['particles_prob_learner_before_demo']
    # prob_pf_actual_teacher_before_test = trial_data['particles_prob_teacher_before_test']
    # prob_pf_actual_learner_before_test_read_data = trial_data['particles_prob_learner_before_test']

    # initialize (simulated) teacher and learner particle filters
    team_prior, teacher_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, teacher_learning_factor=teacher_learning_factor, team_prior = params.team_prior, vars_filename=vars_filename_prefix, model_type = params.teacher_update_model_type)
    learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = team_learning_factor, team_prior = params.team_prior, pf_flag='learner', vars_filename=vars_filename_prefix, model_type = learner_update_type)

    # initialize dataframes to save probability data
    prob_pf = pd.DataFrame()

    prev_kc_id = 1
    # simulate teaching-learning interaction process
    for loop_id in range(len(loop_count)):
        print(colored('loop_id: ' + str(loop_id), 'red'))

        
        # if len(interesting_interactions) > 0:
        #     if loop_id+1 in interesting_interactions:
        #         viz_flag = True
        #     else:
        #         viz_flag = False

        # loop data
        demo_loop = demo_constraints.iloc[loop_id]
        min_demo_loop = min_demo_constraints.iloc[loop_id]
        test_loop = test_constraints.iloc[loop_id]
        test_responses_loop = test_responses_team.iloc[loop_id]
        team_models_loop = team_response_models.iloc[loop_id]
        current_kc_id = kc_id_list.iloc[loop_id]

        # check and reset team learning factor
        if loop_id > 0 and current_kc_id > prev_kc_id:
            team_learning_factor = copy.deepcopy(initial_team_learning_factor)
            print('team_learning_factor reset to initial_team_learning_factor for new KC: ', current_kc_id, '. TLF: ', team_learning_factor)

        # learner_pf_actual_after_demo_loop = learner_pf_actual_after_demo.iloc[loop_id]
        # prob_pf_actual_learner_before_test_loop = prob_pf_actual_learner_before_test_read_data.iloc[loop_id]
        # team_learning_factor_loop = team_learning_factor.iloc[loop_id]

        # initialize dicts to save probability data
        ## current PF (w test response and wo feedback)
        prob_pf_teacher_before_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, teacher_pf, [params.default_learning_factor_teacher]*params.team_size)
        prob_pf_teacher_before_demo_dict['prob_type'] = 'teacher_before_demo'
        prob_pf_teacher_before_demo_dict['loop_id'] = loop_id+1
        prob_pf_teacher_before_demo_dict['constraints'] = min_demo_loop
        prob_pf_teacher_before_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_teacher_before_demo_dict['team_composition'] = team_composition

        prob_pf_learner_before_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, learner_pf, team_learning_factor)
        prob_pf_learner_before_demo_dict['prob_type'] = 'learner_before_demo'
        prob_pf_learner_before_demo_dict['loop_id'] = loop_id+1
        prob_pf_learner_before_demo_dict['constraints'] = min_demo_loop
        prob_pf_learner_before_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_learner_before_demo_dict['team_composition'] = team_composition

        ## learner PF without 

        # update demo dataframes
        prob_pf = prob_pf.append(prob_pf_teacher_before_demo_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_before_demo_dict, ignore_index=True)

        # test dict
        prob_pf_teacher_before_test_dict = {'loop_id': loop_id+1,  'prob_type': 'teacher_before_test', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_learner_before_test_dict = {'loop_id': loop_id+1,  'prob_type': 'learner_before_test', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_teacher_after_test_dict = {'loop_id': loop_id+1,  'prob_type': 'teacher_after_test', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_teacher_after_feedback_dict = {'loop_id': loop_id+1,  'prob_type': 'teacher_after_feedback', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_learner_after_feedback_dict = {'loop_id': loop_id+1,  'prob_type': 'learner_after_feedback', 'demo_strategy': demo_strategy, 'team_composition': team_composition}

        # test constraints
        test_loop_extended = []
        for test_id in range(len(test_loop)):
            test_loop_extended.extend(test_loop[test_id])
        print('test_loop_extended: ', test_loop_extended)


        # update teacher and learner particle filters for demonstrations
        for p_id in range(params.team_size):
            member_id = 'p' + str(p_id+1)

            ### update teacher and learner particle filters for demonstrations
            print('Teacher update after demo for member: ', member_id)
            teacher_pf[member_id].update(demo_loop, teacher_learning_factor[p_id], plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Teacher belief after demo for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = params.teacher_update_model_type)
            # teacher_pf[member_id].calc_particles_probability(demo_loop)
            # prob_pf_teacher_after_demo_dict[member_id] = teacher_pf[member_id].particles_prob_correct

            print('Learner update after demo for member: ', member_id)
            learner_pf[member_id].update(demo_loop, team_learning_factor[p_id], plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Learner belief after demo for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)
            # learner_pf[member_id].calc_particles_probability(demo_loop)
            # prob_pf_learner_after_demo_dict[member_id] = learner_pf[member_id].particles_prob_correct

        # update prob after demos
        prob_pf_teacher_after_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, teacher_pf, [params.default_learning_factor_teacher]*params.team_size)
        prob_pf_teacher_after_demo_dict['loop_id'] = loop_id+1
        prob_pf_teacher_after_demo_dict['constraints'] = min_demo_loop
        prob_pf_teacher_after_demo_dict['prob_type'] = 'teacher_after_demo'
        prob_pf_teacher_after_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_teacher_after_demo_dict['team_composition'] = team_composition
        

        prob_pf_learner_after_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, learner_pf, team_learning_factor)
        prob_pf_learner_after_demo_dict['loop_id'] = loop_id+1
        prob_pf_learner_after_demo_dict['constraints'] = min_demo_loop
        prob_pf_learner_after_demo_dict['prob_type'] = 'learner_after_demo'
        prob_pf_learner_after_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_learner_after_demo_dict['team_composition'] = team_composition

        # update demo dataframes
        prob_pf = prob_pf.append(prob_pf_teacher_after_demo_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_after_demo_dict, ignore_index=True)

        # plot sampled models
        print('team_models_loop: ', team_models_loop)
        if viz_flag:
            plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Human models sampled for test'
            sim_helpers.plot_sampled_models(learner_pf, test_loop_extended, team_models_loop, 1, plot_title = plot_title, vars_filename=vars_filename_prefix)
            # plot_title = 'Actual interaction No.' + str(loop_id +1) + '. Human models sampled for test'
            # sim_helpers.plot_sampled_models(learner_pf_actual_after_demo_loop, test_loop_extended, team_models_loop, 1, plot_title = plot_title, vars_filename=vars_filename_prefix)
            
            # if loop_id == 3:
            #     break
        print('test_responses_loop: ', test_responses_loop)



        # ### update teacher and learner particle filters for tests
            
        # # Method 2: update particle filter (after all test responses)
        N_tests = len(test_loop)
        
        
        for p_id in range(params.team_size):
            member_id = 'p' + str(p_id+1)
            lf_var = member_id + '_lf'
            
            if N_tests == 1:
                all_test_responses = test_responses_loop[p_id]
            else:
                all_test_responses = []
                for test_id in range(N_tests):
                    all_test_responses.append(test_responses_loop[p_id][test_id])
            print('all_test_responses: ', all_test_responses)


            # probabilities before test
            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_teacher_before_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct
            prob_pf_teacher_before_test_dict['constraints'] = test_loop_extended
            prob_pf_teacher_before_test_dict[lf_var] = params.default_learning_factor_teacher

            learner_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_learner_before_test_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_pf_learner_before_test_dict['constraints'] = test_loop_extended
            prob_pf_learner_before_test_dict[lf_var] = team_learning_factor[p_id]


            # update particle filter for test response (only for teacher)
            print('Teacher update after tests for member: ', member_id)
            teacher_pf[member_id].update(all_test_responses, teacher_learning_factor[p_id], model_type = params.teacher_update_model_type,  plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Teacher belief after test for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix)

            # probabibilites after test
            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_teacher_after_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct
            prob_pf_teacher_after_test_dict['constraints'] = test_loop_extended
            prob_pf_teacher_after_test_dict['test_response'] = all_test_responses
            prob_pf_teacher_after_test_dict[lf_var] = params.default_learning_factor_teacher


            ## update particle filter for feedback
            teacher_feedback_lf = []
            learner_feedback_lf = []

            # 
            # for test_id in range(N_tests):
                # if (all_test_responses[test_id] == test_loop_extended[test_id]).all():
                #     teacher_feedback_lf.append(min(1.1*params.default_learning_factor_teacher, max_learning_factor))
                #     learner_feedback_lf.append(min(1.05*team_learning_factor[p_id], max_learning_factor))
                # else:
                #     teacher_feedback_lf.append(min(1.05*params.default_learning_factor_teacher, max_learning_factor))
                #     learner_feedback_lf.append(min(1.05*team_learning_factor[p_id], max_learning_factor))
            
            all_tests_correct_flag = True
            for test_id in range(N_tests):
                if not (all_test_responses[test_id] == test_loop_extended[test_id]).all():
                    all_tests_correct_flag = False
                    break
            
            # update learning factor before feedback
            if feedback_flag:
                if all_tests_correct_flag:
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_factor_delta[p_id, 0], max_learning_factor)
                else:
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_factor_delta[p_id, 1], max_learning_factor)

            if review_flag:
                print('Teacher update after feedback for member: ', member_id)
                teacher_pf[member_id].update(test_loop_extended, teacher_learning_factor[p_id], model_type = params.teacher_update_model_type,  plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Teacher belief after corrective feedback for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix)

                print('Learner update after feedback for member: ', member_id)
                learner_pf[member_id].update(test_loop_extended, team_learning_factor[p_id], model_type = learner_update_type, plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Learner belief after corrective feedback for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix)

            # # update learning factor after feedback
            # if all_tests_correct_flag:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
            # else:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)

            # probabilities after feedback
            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_teacher_after_feedback_dict[member_id] = teacher_pf[member_id].particles_prob_correct
            prob_pf_teacher_after_feedback_dict['constraints'] = test_loop_extended
            prob_pf_teacher_after_feedback_dict[lf_var] = teacher_feedback_lf


            learner_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_learner_after_feedback_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_pf_learner_after_feedback_dict['constraints'] = test_loop_extended
            prob_pf_learner_after_feedback_dict[lf_var] = learner_feedback_lf
        
        #########

            # response type and update team learning factor
            print('all_test_responses: ', all_test_responses, 'test_loop_extended: ', test_loop_extended)
            prob_pf_learner_before_test_dict['N_tests'] = N_tests
            prob_pf_teacher_before_test_dict['N_tests'] = N_tests
            prob_pf_teacher_after_test_dict['N_tests'] = N_tests
            
            tests_correct_flag = True
            for test_id in range(N_tests):
                
                if (all_test_responses[test_id] == test_loop_extended[test_id]).all():
                    # team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
                    var_name = 'response_type_' + member_id
                    if test_id == 0:
                        prob_pf_learner_before_test_dict[var_name] = ['correct']
                        prob_pf_teacher_before_test_dict[var_name] = ['correct']
                        prob_pf_teacher_after_test_dict[var_name] = ['correct']
                        prob_pf_teacher_after_feedback_dict[var_name] = ['correct']
                        prob_pf_learner_after_feedback_dict[var_name] = ['correct']
                    else:
                        prob_pf_learner_before_test_dict[var_name].append('correct')
                        prob_pf_teacher_before_test_dict[var_name].append('correct')
                        prob_pf_teacher_after_test_dict[var_name].append('correct')
                        prob_pf_teacher_after_feedback_dict[var_name].append('correct')
                        prob_pf_learner_after_feedback_dict[var_name].append('correct')
                else:
                    # team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)
                    tests_correct_flag = False
                    var_name = 'response_type_' + member_id
                    if test_id == 0:
                        prob_pf_learner_before_test_dict[var_name] = ['incorrect']
                        prob_pf_teacher_before_test_dict[var_name] = ['incorrect']
                        prob_pf_teacher_after_test_dict[var_name] = ['incorrect']
                        prob_pf_teacher_after_feedback_dict[var_name] = ['incorrect']
                        prob_pf_learner_after_feedback_dict[var_name] = ['incorrect']
                    else:
                        prob_pf_learner_before_test_dict[var_name].append('incorrect')
                        prob_pf_teacher_before_test_dict[var_name].append('incorrect')
                        prob_pf_teacher_after_test_dict[var_name].append('incorrect')
                        prob_pf_teacher_after_feedback_dict[var_name].append('incorrect')
                        prob_pf_learner_after_feedback_dict[var_name].append('incorrect')

            # # update learning factor when all tests are correct
            # if tests_correct_flag:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
            # else:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)

        # update test dataframes
        prob_pf = prob_pf.append(prob_pf_teacher_before_test_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_before_test_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_teacher_after_test_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_teacher_after_feedback_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_after_feedback_dict, ignore_index=True)
        
        # update KC
        prev_kc_id = current_kc_id
        

    #################
    # save dataframes final
        
    prob_pf.to_csv(path + '/' + vars_filename_prefix + '_prob_pf.csv')
    with open(path + '/' + vars_filename_prefix + '_prob_pf.pickle', 'wb') as f:
        pickle.dump(prob_pf, f)
    ####################

    # plot simulated probability data
    # f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # # Method 1
    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(prob_pf_learner_before_test, x=prob_pf_learner_before_test.index, y = member_id, hue = 'test_id', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Simulated Prob. correct test response member'+ member_id)

    # # Method 2
    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(prob_pf_learner_before_test, x=prob_pf_learner_before_test.index, y = member_id, ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Simulated Prob. correct test response member'+ member_id)
    # if params.save_plots_flag:
    #     plt.savefig(path + '/' + vars_filename_prefix + '_simulated_probability.png')

    # # plot actual probability data
    # f2, ax2 = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # # for col_id in range(3):  
    # #     member_id = 'p' + str(col_id+1)
    # #     sns.lineplot(prob_pf_actual_learner_before_test, x=prob_pf_actual_learner_before_test.index, y = member_id, hue = 'test_id', ax=ax2[col_id], errorbar=('se', 1), err_style="band").set(title='Actual Prob. correct test response member'+ member_id)

    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(prob_pf_actual_learner_before_test, x=prob_pf_actual_learner_before_test.index, y = member_id, ax=ax2[col_id], errorbar=('se', 1), err_style="band").set(title='Actual Prob. correct test response member'+ member_id)

    # if params.save_plots_flag:
    #     plt.savefig(path + '/' + vars_filename_prefix + '_actual_probability.png')

    # plt.show()
    


def plot_pf_updates(path, file_prefix):
    # plots probbability


    filelist = os.listdir(path)
    save_filename = path + '/' + file_prefix + '_full_compiled_data_uf_test.csv'
    noise_cond_to_check = 'no_noise'
    

    prob_data = pd.DataFrame()
    run_id = 1
    for filename in filelist:
        # if file_prefix in filename and '.csv' in filename:
        # if file_prefix in filename and '.pickle' in filename and 'uf_test' in filename:
        if file_prefix in filename and '.pickle' in filename:
            if file_prefix + '.pickle' != filename:
                
                # only plot sample conditions
                # if not '51' in filename:
                #     continue


                filename_wo_prefix = filename.split(file_prefix)[0]
                

                if 'uf_test' in filename_wo_prefix:
                    continue

                print('filename_wo_prefix: ', filename_wo_prefix)
                print('Reading file: ', filename)
                # trial_data = pd.read_csv(path + '/' + filename)
                with open(path + '/' + filename, 'rb') as f:
                    trial_data = pickle.load(f)

                trial_data['run_id'] = run_id*np.ones(len(trial_data))
                
                if 'learner_no_noise' in filename_wo_prefix:
                    trial_data['learner_noise_cond'] = ['no_noise']*len(trial_data)
                else:
                    trial_data['learner_noise_cond'] = ['noise']*len(trial_data)

                if 'w_feedback' in filename_wo_prefix:
                    if 'reverse' in filename_wo_prefix:
                        trial_data['feedback_cond'] = ['w_feedback_incorrect_understands_better']*len(trial_data)
                    elif 'normal' in filename_wo_prefix:
                        trial_data['feedback_cond'] = ['w_feeback_correct_understands_better']*len(trial_data)
                    else:
                        # skip
                        print('Skipping trial data: ', filename)
                        continue
                    # trial_data['feedback_cond'] = ['w_feedback']*len(trial_data)
                elif 'no_review' in filename_wo_prefix:
                    trial_data['feedback_cond'] = ['no_feedback_no_review']*len(trial_data)
                else:
                    trial_data['feedback_cond'] = ['no_feedback_w_review']*len(trial_data)

                    

               

                if 'team_composition' not in trial_data.columns:
                    trial_data['team_composition'] = ['[0, 0, 0]']*len(trial_data)
                else:
                    trial_data['team_composition'] = trial_data['team_composition'].astype(str)

                prob_data = prob_data.append(trial_data, ignore_index=True)


                run_id += 1

    prob_data.to_csv(save_filename)

    print(prob_data)

    # plot data

    # plot_data_no_feedback = prob_data[(prob_data['prob_type']=='learner_before_test') & (prob_data['noise_cond']=='noise') & (prob_data['feedback_cond']=='no_feedback')]
    # plot_data_feedback = prob_data[(prob_data['prob_type']=='learner_before_test') & (prob_data['noise_cond']=='noise') & (prob_data['feedback_cond']=='w_feedback')]

    # plot_data_no_feedback = prob_data[(prob_data['learner_noise_cond']==noise_cond_to_check) & (prob_data['feedback_cond']=='no_feedback')]
    # plot_data_feedback = prob_data[ (prob_data['learner_noise_cond']==noise_cond_to_check) & (prob_data['feedback_cond']=='w_feedback')]

    # plot_data_no_feedback = prob_data[(prob_data['learner_noise_cond']==noise_cond_to_check) & (prob_data['feedback_cond']=='no_feedback') & (prob_data['feedback_type']=='incorrect_understands_better')]
    # plot_data_feedback = prob_data[ (prob_data['learner_noise_cond']==noise_cond_to_check) & (prob_data['feedback_cond']=='w_feedback') & (prob_data['feedback_type']=='incorrect_understands_better')]

    plot_data = prob_data[(prob_data['learner_noise_cond']==noise_cond_to_check)]


    # prob_type_list = ['learner_before_demo', 'learner_after_demo', 'learner_before_test', 'learner_after_test', 'learner_after_feedback']
    # prob_type_list = ['teacher_before_demo', 'teacher_after_demo', 'teacher_before_test', 'teacher_after_test', 'teacher_after_feedback']
    
    prob_type_list = ['learner_before_test']
    # prob_type_list = ['teacher_after_feedback', 'learner_after_feedback']

    # prob_type_list = ['learner_before_test']

    plot_data_no_feedback_prob_type = pd.DataFrame()
    plot_data_feedback_prob_type = pd.DataFrame()
    plot_data_prob_type = pd.DataFrame()
    for prob_type in prob_type_list:
        # plot_data_no_feedback_prob_type = plot_data_no_feedback_prob_type.append(plot_data_no_feedback[plot_data_no_feedback['prob_type']==prob_type], ignore_index=True)
        # plot_data_feedback_prob_type = plot_data_feedback_prob_type.append(plot_data_feedback[plot_data_feedback['prob_type']==prob_type], ignore_index=True)
        plot_data_prob_type = plot_data_prob_type.append(plot_data[plot_data['prob_type']==prob_type], ignore_index=True)


    # for plot_id in range(len(plot_data_no_feedback_prob_type)):
    #     for p_id in range(params.team_size):
    #         member_id = 'p' + str(p_id+1)
    #         lf_var = member_id + '_lf'
    #         if type(plot_data_no_feedback_prob_type[lf_var].iloc[plot_id]) == list:
    #             plot_data_no_feedback_prob_type[lf_var].iloc[plot_id] = plot_data_no_feedback_prob_type[lf_var].iloc[plot_id][0]

    # for plot_id in range(len(plot_data_feedback_prob_type)):
    #     for p_id in range(params.team_size):
    #         member_id = 'p' + str(p_id+1)
    #         lf_var = member_id + '_lf'
    #         if type(plot_data_feedback_prob_type[lf_var].iloc[plot_id]) == list:
    #             plot_data_feedback_prob_type[lf_var].iloc[plot_id] = plot_data_feedback_prob_type[lf_var].iloc[plot_id][0]



    # save files
    plot_data_feedback_prob_type.to_csv(path + '/' + file_prefix + '_plot_data_w_feedback_prob_type_' + noise_cond_to_check + '.csv')
    plot_data_no_feedback_prob_type.to_csv(path + '/' + file_prefix + '_plot_data_no_feedback_prob_type_' + noise_cond_to_check + '.csv')

    plot_data_prob_type.to_csv(path + '/' + file_prefix + '_plot_data.csv')


    # prob_type_order = ['teacher_before_demo', 'learner_before_demo', 'teacher_after_demo', 'learner_after_demo', 'teacher_before_test', 'learner_before_test', 'teacher_after_test', 'teacher_after_feedback', 'learner_after_feedback']
    feedback_cond_order = ['no_feedback_no_review', 'no_feedback_w_review', 'w_feedback_incorrect_understands_better', 'w_feeback_correct_understands_better']

    ## plot teacher vs learner probabilities
    # teacher plots
    # ft, axt = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # for col_id in range(3):
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(plot_data_prob_type, x='loop_id', y = member_id, hue = 'team_composition', ax=axt[col_id], errorbar=('se', 1), err_style="band").set(title='Prob. correct test response before test - team composition')
    #     axt[col_id].axvline(x=4, color='k', linestyle='--')
    #     axt[col_id].axvline(x=7, color='k', linestyle='--')
    #     axt[col_id].set_ylim([0, 1])
    #     axt[col_id].set_xlim([0, 10])

    
    # fl, axl = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # for col_id in range(3):
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(plot_data_prob_type, x='loop_id', y = member_id, hue = 'feedback_type', ax=axl[col_id], errorbar=('se', 1), err_style="band").set(title='Prob. correct test response before test - Feedback type')
    #     axl[col_id].axvline(x=4, color='k', linestyle='--')
    #     axl[col_id].axvline(x=7, color='k', linestyle='--')
    #     axl[col_id].set_ylim([0, 1])
    #     axl[col_id].set_xlim([0, 10])

    fl2, axl2 = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize=(15,10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # axl3 = np.array(axl2).reshape(-1)
    for row_id in range(3):
        member_id = 'p' + str(row_id+1)
        sns.lineplot(plot_data_prob_type, x='loop_id', y = member_id, hue = 'feedback_cond', hue_order=feedback_cond_order, ax=axl2[row_id, 0], errorbar=('se', 1), err_style="band", legend=False).set(title='Prob. correct test response before test - Feedback condition Member '+ member_id)
        axl2[row_id, 0].axvline(x=4, color='k', linestyle='--')
        axl2[row_id, 0].axvline(x=7, color='k', linestyle='--')
        axl2[row_id, 0].set_ylim([0, 1])
        axl2[row_id, 0].set_xlim([0, 10])

        # # plot uf in same plot
        # axl3[col_id] = axl2[col_id].twinx()
        # axl3[col_id].set_ylim([0.5, 1])
        # sns.lineplot(plot_data_prob_type, x='loop_id', y = member_id+'_lf', ax=axl3[col_id], errorbar=('se', 1), err_style="band", linestyle = '--', color = 'r').set(title='Member '+ member_id)

        # plot uf in adjacent plot
        sns.lineplot(plot_data_prob_type, x='loop_id', y = member_id+'_lf', hue = 'feedback_cond', hue_order=feedback_cond_order, ax=axl2[row_id,1], errorbar=('se', 1), err_style="band", linestyle = '--', color = 'r', legend=False).set(title='Understanding factor. Member '+ member_id)
        axl2[row_id, 1].axvline(x=4, color='k', linestyle='--')
        axl2[row_id, 1].axvline(x=7, color='k', linestyle='--')
        axl2[row_id, 1].set_ylim([0.5, 1])

    
    
    ## understanding factor vs probability

    # fl2, axl2 = plt.subplots(nrows=1, sharex=True, sharey=False, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # # axl3 = np.array(axl2).reshape(-1)
    # for row_id in range(1):
    #     member_id = 'p' + str(row_id+1)
    #     sns.lineplot(plot_data_prob_type, x=member_id+'_lf', y = member_id, hue = 'loop_id',  ax=axl2, errorbar=('se', 1), err_style="band", legend=False).set(title='UF vs. Prob. correct test response before test - Feedback condition Member '+ member_id)
    #     axl2.axvline(x=4, color='k', linestyle='--')
    #     axl2.axvline(x=7, color='k', linestyle='--')
    #     axl2.set_ylim([0, 1])
    #     axl2.set_xlim([0, 1])

        # # plot uf in adjacent plot
        # sns.lineplot(plot_data_prob_type, x='loop_id', y = member_id+'_lf', hue = 'feedback_cond', hue_order=feedback_cond_order, ax=axl2[row_id,1], errorbar=('se', 1), err_style="band", linestyle = '--', color = 'r', legend=False).set(title='Understanding factor. Member '+ member_id)
        # axl2[row_id, 1].axvline(x=4, color='k', linestyle='--')
        # axl2[row_id, 1].axvline(x=7, color='k', linestyle='--')
        # axl2[row_id, 1].set_ylim([0.5, 1])


    # fl2.suptitle('Prob. correct test response before test - Feedback condition')

    # # plot probabilities vs loop id
    # f1, ax1= plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15,10))
    # ax3 = np.array(ax1).reshape(-1)
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(plot_data_no_feedback_prob_type, x='loop_id', y = member_id, ax=ax1[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)
    #     # sns.lineplot(plot_data_no_feedback_prob_type, x='loop_id', y = member_id, hue = 'prob_type', ax=ax1[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)
    #     ax1[col_id].axvline(x=4, color='k', linestyle='--')
    #     ax1[col_id].axvline(x=7, color='k', linestyle='--')
        
    #     ax3[col_id] = ax1[col_id].twinx()
    #     sns.lineplot(plot_data_no_feedback_prob_type, x='loop_id', y = member_id+'_lf', ax=ax3[col_id], errorbar=('se', 1), err_style="band", color = 'r').set(title='Member'+ member_id)
    #     # ax3[col_id].set_ylim(ax3[col_id].get_ylim()[::-1])

    # f1.suptitle('Prob. correct test response, no noise, no feedback')
    


    # f2, ax2 = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15,10))
    # ax4 = np.array(ax2).reshape(-1)
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(plot_data_feedback_prob_type, x='loop_id', y = member_id, ax=ax2[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)
    #     # sns.lineplot(plot_data_feedback_prob_type, x='loop_id', y = member_id, hue = 'prob_type', ax=ax2[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)
    #     ax2[col_id].axvline(x=4, color='k', linestyle='--')
    #     ax2[col_id].axvline(x=7, color='k', linestyle='--')

    #     ax4[col_id] = ax2[col_id].twinx()
    #     sns.lineplot(plot_data_feedback_prob_type, x='loop_id', y = member_id+'_lf', ax=ax4[col_id], errorbar=('se', 1), err_style="band", color='r').set(title='Member'+ member_id)
    #     # ax4[col_id].set_ylim(ax4[col_id].get_ylim()[::-1])

    # f2.suptitle('Prob. correct test response, no noise, with feedback')

    # plt.show()



    # # plot learning factor vs probability
    # plot_data_no_feedback_prob_type_loops = plot_data_no_feedback_prob_type[plot_data_no_feedback_prob_type['loop_id']<=5]
    # plot_data_feedback_prob_type_loops = plot_data_feedback_prob_type[plot_data_feedback_prob_type['loop_id']<=5]

    # print(plot_data_no_feedback_prob_type_loops)
    # print(plot_data_feedback_prob_type_loops)

    # f5, ax5 = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     lf_member = member_id + '_lf'
    #     plot_data = plot_data_feedback_prob_type_loops[plot_data_feedback_prob_type_loops['prob_type']=='learner_before_test']
    #     sns.lineplot(plot_data, x=lf_member, y = member_id, ax=ax5[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)
    #     sns.lineplot(plot_data, x=lf_member, y = member_id, ax=ax5[col_id], errorbar=('se', 1), err_style="band", color='r').set(title='Member'+ member_id)

    #     # sns.lineplot(plot_data_no_feedback_prob_type_loops, x=lf_member, y = member_id, ax=ax5[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)
    #     # for loop_id in [1, 2, 3]:
    #     #     plot_data_loop = plot_data_no_feedback_prob_type_loops[plot_data_no_feedback_prob_type_loops['loop_id']==loop_id]
    #     #     print(plot_data_loop)
    #     #     sns.lineplot(plot_data_loop, x=lf_member, y = member_id, ax=ax5[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)
        
    # f5.suptitle('Learning factor vs. Prob. correct test response before test')


    # f6, ax6 = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     lf_member = member_id + '_lf'
    #     for loop_id in [1, 2, 3]:
    #         plot_data_loop = plot_data_feedback_prob_type_loops[plot_data_feedback_prob_type_loops['loop_id']==loop_id]
    #         print(plot_data_loop)
    #         sns.lineplot(plot_data_loop, x=lf_member, y = member_id, ax=ax6[col_id], errorbar=('se', 1), err_style="band").set(title='Member'+ member_id)

    # f6.suptitle('Learning factor vs. Prob. correct test response, no noise, no feedback')


    plt.show()
######################################
    

def plot_prob_data(path, file):

    with open(path + '/' + file, 'rb') as f:
        prob_data = pickle.load(f)

    print(prob_data)


    # reformat probability data
    prob_data_reformatted = pd.DataFrame()
    unique_runs = prob_data['run_id'].unique()

    for run_id in unique_runs:
        run_data = prob_data[prob_data['run_id'] == run_id]
        run_data['loop_id'] = np.arange(1, len(run_data)+1)
        prob_data_reformatted = prob_data_reformatted.append(run_data, ignore_index=True)


    # plot teacher and learner probability of correct response for every interaction
    f, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15,10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)



    for p in range(params.team_size):
        member_id = 'p' + str(p+1)
        # plot_data = 
        sns.lineplot(prob_data, x = 'loop_id', y = member_id, hue = 'prob_type', ax=ax[0, p], errorbar=('se', 1), err_style="band").set(title=member_id)





####################################

def read_prob_data(path, file):

    filelist = os.listdir(path)

    all_prob_data_learner_before_test = pd.DataFrame()

    run_id = 1
    for filename in filelist:
        if file in filename and 'learner_before_test' in filename:
            print('filename: ', filename)
            prob_data = pd.read_csv(path + '/' + filename)
            prob_data['run_id'] = run_id*np.ones(len(prob_data))
            if 'learn_noise' in filename:
                prob_data['noise_cond'] = ['noise']*len(prob_data)
            else:
                prob_data['noise_cond'] = ['no_noise']*len(prob_data)
            
            if 'm2' in filename:
                prob_data['test_update'] = ['all_tests']*len(prob_data)
            else:
                prob_data['test_update'] = ['individual_tests']*len(prob_data)


            run_id += 1

            all_prob_data_learner_before_test = pd.concat([all_prob_data_learner_before_test, prob_data], ignore_index=True)

    
    # Plot knowledge level for without noise
    row_id = 0
    f, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(15,10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for p in range(params.team_size):
        member_id = 'p' + str(p+1)
        plot_data = all_prob_data_learner_before_test[(all_prob_data_learner_before_test['noise_cond']=='no_noise') & (all_prob_data_learner_before_test['test_update']=='all_tests')]
        
        # plot_data[member_id] = plot_data[member_id] + np.random.normal(0.02, 0.04, len(plot_data))

        sns.lineplot(plot_data, x = 'loop_id', y = member_id, ax=ax[row_id], errorbar=('se', 1), err_style="band").set(title=member_id)

        # sns.lineplot(plot_data, x = 'loop_id', y = member_id, hue = 'test_id', ax=ax[row_id], errorbar=('se', 1), err_style="band").set(title=member_id)

        row_id += 1 
    f.suptitle('Prob. correct response for all tests, no noise, no corrective feedback')

    
    print(all_prob_data_learner_before_test)
    all_prob_data_learner_before_test.to_csv(path + '/' + file + '_prob_data_learner_before_test.csv')


    prob_filename = '_prob_vars_no_noise_all_tests'

    # For all tests
    plot_data.groupby(['loop_id']).agg({'p1': 'std', 'p2': 'std', 'p3': 'std'}).to_csv(path + '/' + file + prob_filename + '.csv')

    # For individual tests
    # plot_data.groupby(['loop_id', 'test_id']).agg({'p1': 'std', 'p2': 'std', 'p3': 'std'}).to_csv(path + '/' + file + prob_filename + '.csv')
    
    # plot_data.groupby(['loop_id', 'test_id']).agg({'p1': 'std', 'p2': 'std', 'p3': 'std'}).plot()
    plt.show()

    

################################
    
def read_demo_gen_log(demo_file_name, new_flag=False):

    # Initialize the lists to store the data
    lengths = []
    constraint_spaces = []
    sample_models = []
    max_info_gains = []
    N_clusters = []
    cluster_weights = []
    samp_models_cluster_idxs = []
    


    # Variable to collect sample models temporarily
    temp_models = []

    # Function to clean model lines
    def clean_model_line(line):
        return line.strip().replace('\n', '').replace('[', '').replace(']', '').split()

    # # Read the file
    if not new_flag:
        with open(demo_file_name, 'r') as file:
            for line in file:
                if line.startswith('Length of summary:'):
                    # Save previously collected models before resetting
                    if temp_models:
                        sample_models.append(temp_models)
                        temp_models = []  # Reset for the next entry
                    length = int(line.split(': ')[1].strip())
                    lengths.append(length)
                elif line.startswith('constraint_space_to_sample_human_models:'):
                    constraint_space = line.split(': ')[1].strip()
                    constraint_spaces.append(constraint_space)
                elif line.startswith('sample_human_models:'):
                    # print('line: ', line)
                    short_line = line.split(': ')[1].strip()
                    # print('short_line: ', short_line)
                    temp_models.append(clean_model_line(short_line))
                elif line.startswith(' [['):  # Assuming model lines start with '[['
                    temp_models.append(clean_model_line(line))
                elif line.startswith('Max infogain:'):  # End of current entry
                    max_info_gain = float(line.split(': ')[1].strip())
                    max_info_gains.append(max_info_gain)
                elif line.startswith('N clusters:'):
                    N_cluster = float(line.split(': ')[1].strip())
                    N_clusters.append(N_cluster)
                elif line.startswith('Cluster weights:'):
                    Cluster_weight = line.split(': ')[1].strip()
                    cluster_weights.append(Cluster_weight)
                elif line.startswith('sampled_models_cluster_idxs:'):
                    cluster_idx = line.split(': ')[1].strip()
                    samp_models_cluster_idxs.append(cluster_idx)

                    
        # Add the last set of models collected if not already added
        if temp_models:
            sample_models.append(temp_models)
        ##################
    else:
        # Read the file
        with open(demo_file_name, 'r') as file:
            for line in file:
                if line.startswith('Length of summary:'):
                    length = int(line.split(': ')[1].strip())
                    lengths.append(length)
                elif line.startswith('constraint_space_to_sample_human_models:'):
                    constraint_space = line.split(': ')[1].strip()
                    constraint_spaces.append(constraint_space)
                elif line.startswith('sample_human_models:'):
                    array_string = line.split(': ')[1].strip()
                    temp_models = array_string.replace('array', '')
                    # print('array_string: ', temp_models)

                    array_strings = temp_models.split("]), ([")
                    array_strings[0] = array_strings[0].replace("[([", "")
                    array_strings[-1] = array_strings[-1].replace("])]", "")
                    # print('array_strings: ', array_strings)

                    # Convert array strings to NumPy arrays
                    sample_arrays = [np.array(eval(arr)) for arr in array_strings]

                    # print('sample_arrays: ', sample_arrays)
                    sample_models.append(sample_arrays)
                elif line.startswith('Max infogain:'):  # End of current entry
                    max_info_gain = float(line.split(': ')[1].strip())
                    max_info_gains.append(max_info_gain)
                elif line.startswith('N clusters:'):
                    N_cluster = float(line.split(': ')[1].strip())
                    N_clusters.append(N_cluster)
                elif line.startswith('Cluster weights:'):
                    Cluster_weight = line.split(': ')[1].strip()
                    cluster_weights.append(Cluster_weight)
                elif line.startswith('sampled_models_cluster_idxs:'):
                    cluster_idx = line.split(': ')[1].strip()
                    samp_models_cluster_idxs.append(cluster_idx)


    ##############################


    max_info_gain_updated = []
    # interaction_number = []  # does not collect it correctly; just compare summary lengths
    max_info_id = 0
    int_id = 1
    for i in range(len(lengths)-1):
        # interaction_number.append(int_id)

        if lengths[i] == lengths[i+1]:
            max_info_gain_updated.append(0)
            # int_id += 1  # update interaction number
        else:
            max_info_gain_updated.append(max_info_gains[max_info_id])
            max_info_id += 1

    # interaction_number.append(int_id)
    max_info_gain_updated.append(0) # for the final summary


    # # clean interaction number! (not complete)
    # int_no_to_adjust = []
    # for length in lengths:
    #     int_no_ids = [i for i, x in enumerate(lengths) if x == length]
    #     uniq_int = interaction_number[int_no_ids].unique()
    #     if len(uniq_int) > 2:
    #         int_data = []
    #         for uniq_i in uniq_int:
    #             int_data.append([uniq_i, len(interaction_number[uniq_i])])

    #         int_no_to_adjust.append()
    # print('lengths: ', len(lengths))
    # print('constraint_spaces: ', len(constraint_spaces))
    # print('sample_models: ', len(sample_models))
    # print('max_info_gains: ', len(max_info_gain_updated))

    # print('lengths: ', lengths)
    # print('constraint_spaces: ', constraint_spaces)
    # print('sample_models: ', sample_models)
    # print('max_info_gains: ', max_info_gain_updated)



    # Create a DataFrame
    df = pd.DataFrame({
        'length_of_summary': lengths,
        'constraint_space_to_sample_human_models': constraint_spaces,
        'sample_human_models': sample_models,
        'max_infogain': max_info_gain_updated,
        'N_clusters': N_clusters,
        'cluster_weights': cluster_weights,
        'sampled_models_cluster_idxs': samp_models_cluster_idxs
    })


    return df

##########################

def plot_pf_dist(path, file, counterfactuals_plot_flag = False):

    with open(path + '/' + file, 'rb') as f:
        pf_data = pickle.load(f)

    if counterfactuals_plot_flag:
        folder = file.strip('.pickle')
        demo_file_name = path + '/ind_sim_trials/' + folder + '/demo_gen_log.txt'
        counterfactuals = read_demo_gen_log(demo_file_name, new_flag=True)
        counterfactuals.to_csv(path + '/ind_sim_trials/' + folder + '_counterfactuals.csv')
            
    for i in range(len(counterfactuals)):
        counterfactuals['sample_human_models'].iloc[i] = [np.array(x).astype(float) for x in counterfactuals['sample_human_models'].iloc[i]]
    
    # print(counterfactuals)

    # print(type(counterfactuals['sample_human_models'].iloc[0]))
    # print(counterfactuals['sample_human_models'].iloc[0])

    # print(pf_data)

    # plot data
    BEC_area_constraints_list = []
    for int_id in range(len(pf_data)):
        learner_pf_after_demo = pf_data['particles_team_learner_after_demos'].iloc[int_id]
        learner_pf_after_test = pf_data['particles_team_learner_final'].iloc[int_id]
        teacher_pf_after_demo = pf_data['particles_team_teacher_after_demos'].iloc[int_id]
        teacher_pf_after_feedback = pf_data['particles_team_teacher_final'].iloc[int_id]

        test_response_models = pf_data['team_response_models'].iloc[int_id]
        min_KC_constraints = pf_data['min_KC_constraints'].iloc[int_id]
        min_BEC_constraints = pf_data['min_BEC_constraints'].iloc[int_id]
        test_responses = pf_data['test_constraints_team'].iloc[int_id]
        team_learning_factor = pf_data['team_learning_factor'].iloc[int_id]
        knowledge_id = pf_data['knowledge_id'].iloc[int_id]

        unit_constraints = pf_data['unit_constraints'].iloc[int_id]
        BEC_area_constraints = BEC_helpers.calc_solid_angles(unit_constraints)
        BEC_area_constraints_list.append(BEC_area_constraints)

        # find sampled counterfactuals
        summary_count = pf_data['summary_count'].iloc[int_id]
        counterfactual_models_list = counterfactuals[counterfactuals['length_of_summary']==summary_count]['sample_human_models']
        counterfactual_models = counterfactual_models_list.iloc[0]
        
        # find the summary counts for this interaction
        indices_to_check = counterfactuals.index[(counterfactuals.index <= counterfactual_models_list.index[0]) & (counterfactuals.index > counterfactuals.index[0])]
        indices_to_check = indices_to_check[::-1]
        max_info_gain = []
        for idx in indices_to_check:
            if counterfactuals['length_of_summary'].iloc[idx] != counterfactuals['length_of_summary'].iloc[idx-1]:
                max_info_gain.append(np.round(counterfactuals['max_infogain'].iloc[idx-1],2))
            else:
                break

        # print('indices_to_check:', indices_to_check, 'max_info_gain:', max_info_gain)
        # max_info_gain_list = counterfactuals[counterfactuals['length_of_summary']==summary_count]['max_infogain']
        # max_info_gain = max_info_gain_list.iloc[0]
        
        

        # print('summary_count: ', summary_count, 'counterfactual_models: ', counterfactual_models)
        # print('sampled_models: ', test_response_models)

        # f2, ax2 = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(15,10))
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)

        f2 = plt.figure(figsize=(15,10))
        ax2_0 = f2.add_subplot(2, 5, 1, projection='3d')
        ax2_list = [ax2_0]
        
        for i in range(1, 10):
            # Create each subplot with shared axes
            ax = f2.add_subplot(2, 5, i + 1, projection='3d', sharex=ax2_0, sharey=ax2_0, sharez=ax2_0)
            # Append the subplot to the list
            ax2_list.append(ax)
        ax2 = np.array(ax2_list)

        # print(len(ax2))

        for p in range(params.team_size):
            member_id = 'p' + str(p+1)
            teacher_pf_after_demo[member_id].plot(fig=f2, ax=ax2[p])
            teacher_pf_after_demo[member_id].calc_particles_probability(min_KC_constraints)
            # print('teacher_pf_prob_after_demo: ', teacher_pf_after_demo[member_id].particles_prob_correct)

            # print('p: ', p)
            # N_models = len(test_response_models[member_id])
            # for model_id in range(N_models):
                # print('model: ', test_response_models[member_id][model_id])
                # if N_models ==1 :
                #     print(sampled_models[member_id][model_id])
                #     ax2[p].scatter(sampled_models[member_id][model_id][0], sampled_models[member_id][model_id][1], sampled_models[member_id][model_id][2],  color='r', s=50)
                # else:
                    # test response model
                    # ax2[p].scatter(test_response_models[member_id][model_id][0][0], test_response_models[member_id][model_id][0][1], test_response_models[member_id][model_id][0][2],  color='r', s=50)

                
            # counterfactual model for next interaction! (just the first)
            if knowledge_id == member_id:
                for counterfactual_model in counterfactual_models:
                    ax2[5+p].scatter(counterfactual_model[0], counterfactual_model[1], counterfactual_model[2],  color='r', marker='+', s=20)
                ax2[5+p].text(-1, 1, 1, 'ig: ' + str(max_info_gain), color='k')


            ax2[p].set_title('Teacher after demo ' + member_id)
            teacher_pf_after_feedback[member_id].plot(fig=f2, ax=ax2[5+p])
            ax2[5+p].set_title('Teacher after feedback ' + member_id)

            teacher_pf_after_feedback[member_id].calc_particles_probability(min_KC_constraints)
            print('teacher_pf_after_feedback: ', teacher_pf_after_feedback[member_id].particles_prob_correct)

            label_axes(ax2[p], params.mdp_class, params.weights['val'])
            label_axes(ax2[5+p], params.mdp_class, params.weights['val'])


        # common and joint after demo
        teacher_pf_after_demo['common_knowledge'].plot(fig=f2, ax=ax2[3])
        ax2[3].set_title('Teacher after demo common knowledge')

        teacher_pf_after_demo['joint_knowledge'].plot(fig=f2, ax=ax2[4])
        ax2[4].set_title('Teacher after demo joint knowledge')


        # common and joint after feedback
        
        teacher_pf_after_feedback['common_knowledge'].plot(fig=f2, ax=ax2[8])
        ax2[8].set_title('Teacher after feedback common knowledge')
        if knowledge_id == 'common_knowledge':
            for counterfactual_model in counterfactual_models:
                ax2[8].scatter(counterfactual_model[0], counterfactual_model[1], counterfactual_model[2],  color='r', marker='+', s=20)
            ax2[8].text(-1, 1, 1, 'ig: ' + str(max_info_gain), color='k')

        
        teacher_pf_after_feedback['joint_knowledge'].plot(fig=f2, ax=ax2[9])
        ax2[9].set_title('Teacher after feedback joint knowledge')
        if knowledge_id == 'joint_knowledge':
            for counterfactual_model in counterfactual_models:
                ax2[9].scatter(counterfactual_model[0], counterfactual_model[1], counterfactual_model[2],  color='r', marker='+', s=20)
            ax2[9].text(-1, 1, 1, 'ig: ' + str(max_info_gain), color='k')
    
        label_axes(ax2[3], params.mdp_class, params.weights['val'])
        label_axes(ax2[4], params.mdp_class, params.weights['val'])
        label_axes(ax2[8], params.mdp_class, params.weights['val'])
        label_axes(ax2[9], params.mdp_class, params.weights['val'])


        teacher_pf_after_demo['common_knowledge'].calc_particles_probability(min_KC_constraints)
        print('teacher_pf_after_demo: ', teacher_pf_after_demo['common_knowledge'].particles_prob_correct)

        teacher_pf_after_feedback['common_knowledge'].calc_particles_probability(min_KC_constraints)
        print('teacher_pf_after_feedback: ', teacher_pf_after_feedback['common_knowledge'].particles_prob_correct)

        teacher_pf_after_demo['common_knowledge'].calc_particles_probability(min_BEC_constraints)
        print('teacher_pf_BEC_after_demo: ', teacher_pf_after_demo['common_knowledge'].particles_prob_correct)

        teacher_pf_after_feedback['common_knowledge'].calc_particles_probability(min_BEC_constraints)
        print('teacher_pf_BEC_after_feedback: ', teacher_pf_after_feedback['common_knowledge'].particles_prob_correct)

        f2.suptitle('Teacher Particle Filter Distribution int no: ' + str(int_id+1))

        plt.savefig(path + '/pf_dist_teacher_' + file + str(int_id+1) + '.png')

        #####################################


        # f = plt.figure(figsize=(15,10))
        # ax1_0 = f.add_subplot(2, 3, 1, projection='3d')
        # ax1_list = [ax1_0]
        
        # for i in range(1, 6):
        #     # Create each subplot with shared axes
        #     ax = f.add_subplot(2, 3, i + 1, projection='3d', sharex=ax1_0, sharey=ax1_0, sharez=ax1_0)
        #     # Append the subplot to the list
        #     ax1_list.append(ax)
        # ax1 = np.array(ax1_list)
        
        # for p in range(params.team_size):
        #     member_id = 'p' + str(p+1)
        #     learner_pf_after_demo[member_id].plot(fig=f, ax=ax1[p])
        #     utils_teams.visualize_planes_team(min_KC_constraints, fig=f, ax= ax1[p], alpha=0.25)
        #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_KC_constraints)
        #     poly = Polyhedron.Polyhedron(ieqs=ieqs)
        #     BEC_viz.visualize_spherical_polygon(poly, fig=f, ax=ax1[p], plot_ref_sphere=False, alpha=0.25)


            
        #     print('p: ', p)
        #     N_models = len(test_response_models[member_id])
        #     for model_id in range(N_models):
        #         print('model: ', test_response_models[member_id][model_id])
        #         # if N_models ==1 :
        #         #     ax1[p].scatter(test_response_models[member_id][model_id][0], test_response_models[member_id][model_id][1], test_response_models[member_id][model_id][2],  color='r', s=50)
        #         # else:
        #         ax1[p].scatter(test_response_models[member_id][model_id][0][0], test_response_models[member_id][model_id][0][1], test_response_models[member_id][model_id][0][2],  color='r', s=50)
            
        #     ax1[p].set_title('Learner after demo ' + member_id)
        #     learner_pf_after_demo[member_id].plot(fig=f2, ax=ax1[3+p])
        #     utils_teams.visualize_planes_team(test_responses[p], fig=f, ax= ax1[3+p], alpha=0.25)
        #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(test_responses[p])
        #     poly = Polyhedron.Polyhedron(ieqs=ieqs)
        #     BEC_viz.visualize_spherical_polygon(poly, fig=f, ax=ax1[3+p], plot_ref_sphere=False, alpha=0.25)
        #     ax1[3+p].set_title('Learner after feedback ' + member_id)

        #     learner_pf_after_demo[member_id].calc_particles_probability(min_KC_constraints)
        #     print('learner_pf_after_demo: ', learner_pf_after_demo[member_id].particles_prob_correct)
        #     learner_after_demo_text = 'p_KC:' + str(np.round(learner_pf_after_demo[member_id].particles_prob_correct,3))

        #     learner_pf_after_test[member_id].calc_particles_probability(min_KC_constraints)
        #     print('learner_pf_after_feedback: ', learner_pf_after_test[member_id].particles_prob_correct)
        #     learner_after_test_text = 'p_KC:' + str(np.round(learner_pf_after_test[member_id].particles_prob_correct,3))

        #     learner_pf_after_demo[member_id].calc_particles_probability(min_BEC_constraints)
        #     print('learner_pf_BEC_after_demo: ', learner_pf_after_demo[member_id].particles_prob_correct)
        #     learner_after_demo_text = learner_after_demo_text + ', ' + 'p_BEC:' + str(np.round(learner_pf_after_demo[member_id].particles_prob_correct,4))

        #     learner_pf_after_test[member_id].calc_particles_probability(min_BEC_constraints)
        #     print('learner_pf_BEC_after_feedback: ', learner_pf_after_test[member_id].particles_prob_correct)
        #     learner_after_test_text = learner_after_test_text + ', ' + 'p_BEC:' + str(np.round(learner_pf_after_test[member_id].particles_prob_correct,4))

        #     ax1[p].text(-1, -1, 1, learner_after_demo_text, color='black')
        #     ax1[3+p].text(-1, -1, 1, learner_after_test_text, color='black')

        #     ax1[p].text(-1,-1,-1, 'LF:' + str(np.round(team_learning_factor[p],2)), color='black')
        #     ax1[3+p].text(-1,-1,-1, 'LF:' + str(np.round(team_learning_factor[p],2)), color='black')

        # f.suptitle('Learner Particle Filter Distribution int no: ' + str(int_id))

        # plt.savefig(path + '/pf_dist_learner_' + file + str(int_id) + '.png')

        plt.show()


############################################

def combine_baseline_data(path):

    with open(path + '/analysis_baseline_trial_data.pickle', 'rb') as f:
        trial_data = pickle.load(f)

    with open(path + '/analysis_baseline_interaction_data.pickle', 'rb') as f:
        interaction_data = pickle.load(f)

    with open(path + '/analysis_baseline_particles_prob.pickle', 'rb') as f:
        prob_data = pickle.load(f)

    with open(path + '/analysis_baseline_concept_data.pickle', 'rb') as f:
        concept_data = pickle.load(f)


    N = len(trial_data)
    print(trial_data)

    N_teams = np.floor(N/3).astype(int)
    print(N_teams)


    team_composition = [[0, 0, 0], [0, 0, 2], [0, 2, 2], [2, 2, 2]]

    team_composition_list = [team_composition[team_id%4] for team_id in range(N_teams)]
    print('team_composition_list: ', team_composition_list)

    run_no_list_low = []
    run_no_list_high = []

    for ind_id in range(len(trial_data)):
        if trial_data['team_composition'].iloc[ind_id] == str([0]):
            run_no_list_low.append(trial_data['run_no'].iloc[ind_id])
        else:
            run_no_list_high.append(trial_data['run_no'].iloc[ind_id])


    print(run_no_list_low)
    print(run_no_list_high)

    trial_data_baseline_teams = pd.DataFrame()
    interaction_data_baseline_teams = pd.DataFrame()
    particles_data_baseline_teams = pd.DataFrame()
    concept_data_baseline_teams = pd.DataFrame()

    team_id = 0
    compile_inds = True
    while (compile_inds):
        team_composition = team_composition_list[team_id]
        print('team_composition: ', team_composition)
        run_id_team_list = []
        for ind_id in range(len(team_composition)):
            ind_learner = team_composition[ind_id]
            if ind_learner == 0:
                print('Sampling low learner')
                run_id_team = random.sample(run_no_list_low, 1)[0]
                run_no_list_low.remove(run_id_team)
            else:
                print('Sampling high learner')
                run_id_team = random.sample(run_no_list_high, 1)[0]
                run_no_list_high.remove(run_id_team)

            if len(run_no_list_low) == 0 or len(run_no_list_high) == 0:
                compile_inds = False
                print('run_no_list_low: ', run_no_list_low)
                print('run_no_list_high: ', run_no_list_high)
                break
            
            run_id_team_list.append(run_id_team)
            if ind_id == 0:
                trial_data_temp_team = trial_data[trial_data['run_no'] == run_id_team].copy()
            else:
                # print(type(trial_data_temp_team.loc['max_loop_count']), trial_data.loc[trial_data['run_no'] == run_id_team]['max_loop_count'].astype(float))
                trial_data_temp_team['max_loop_count'] = trial_data_temp_team['max_loop_count'].astype(float) + trial_data[trial_data['run_no'] == run_id_team]['max_loop_count'].astype(float).values[0]
                trial_data_temp_team['average_team_knowledge'] = trial_data_temp_team['average_team_knowledge'] + trial_data[trial_data['run_no'] == run_id_team]['average_team_knowledge'].values[0]

        trial_data_temp_team['average_team_knowledge'] = trial_data_temp_team['average_team_knowledge']/len(team_composition)

        trial_data_temp_team['team_composition'] = str(team_composition)
        trial_data_temp_team['run_nos'] = str(run_id_team_list)
        trial_data_baseline_teams = pd.concat([trial_data_baseline_teams, trial_data_temp_team], ignore_index=True)
        # print('trial_data_baseline_teams: ', trial_data_baseline_teams)

        team_id += 1



    with open(path + '/analysis_baseline_trial_data_teams.pickle', 'wb') as f:
        pickle.dump(trial_data_baseline_teams, f)

    trial_data_baseline_teams.to_csv(path + '/analysis_baseline_trial_data_teams.csv')

##########################################


def anova(path, vars_filename_prefix):

    # with open('models/augmented_taxi2/analysis_trial_data_combined.pickle', 'rb') as f:
    #     trial_data = pickle.load(f)

    # trial_data_wo_baseline = trial_data[trial_data['demo_strategy'] != 'baseline'].reset_index(drop=True)
    # sns.boxplot(x='demo_strategy', y='max_loop_count', data=trial_data_wo_baseline)

    # ###############  independent variables  ###########################
    # with open(path + '/' + vars_filename_prefix+ '_trial_data_subset_N15_checked.pickle', 'rb') as f:
    #     trial_data_wo_baseline = pickle.load(f)

    with open(path + '/' + vars_filename_prefix+ '_6_trial_data_N15_checked.pickle', 'rb') as f:
        trial_data_wo_baseline = pickle.load(f)


    # with open('models/augmented_taxi2/sim_study_2_dup_wo_rest_lc_flag_concept_data.pickle', 'rb') as f:
    #     concept_data_total = pickle.load(f)


    # unique_runs = trial_data_wo_baseline['analysis_run_id'].unique()

    # unique_runs = [x for x in unique_runs if x not in [7]]

    # index = concept_data_total['analysis_run_id'].isin(unique_runs).index

    # concept_data = concept_data_total.iloc[index]
    # print('Length of concept data: ', len(concept_data), 'Length of concept data total: ', len(concept_data_total))

    # concept_data['delta_kc_team_avg_knowledge'] = np.zeros(len(concept_data))

    # for id in range(len(concept_data)):

    #     kc_id = concept_data['kc_id'].iloc[id]
    #     run_id = concept_data['analysis_run_id'].iloc[id]

    #     print(concept_data[(concept_data['analysis_run_id'] == run_id) & (concept_data['kc_id']==kc_id-1)])
    #     if kc_id !=1:
    #         concept_data['delta_kc_team_avg_knowledge'].iloc[id] = concept_data['avg_BEC_prob_learner_after_feedback'].iloc[id] - concept_data[(concept_data['analysis_run_id'] == run_id) & (concept_data['kc_id']==kc_id-1)]['avg_BEC_prob_learner_after_feedback'].iloc[0] # just the first one; sometimes trials get appended with another trial data

    #     else:
    #         concept_data['delta_kc_team_avg_knowledge'].iloc[id] = concept_data['avg_BEC_prob_learner_after_feedback'].iloc[id]


    # concept_data.to_csv('models/augmented_taxi2/updated_concept_data.csv')
    
    # outliers = [103, 192, 316]

    # trial_data_wo_baseline = trial_data_wo_baseline[~trial_data_wo_baseline['analysis_run_id'].isin(outliers)]

    # print(len(trial_data_wo_baseline))

    # sns.boxplot(x='demo_strategy', y='max_loop_count', data=trial_data_wo_baseline)
    # plt.show()
    ##################################################
    
    # fig, axs = plt.subplots(1, 2, figsize=(15,10))

    # sns.boxplot(x='demo_strategy', y='max_loop_count', ax=axs[0], data=trial_data_wo_baseline).set_title('Number of interactions')
    # sns.boxplot(x='demo_strategy', y='average_team_knowledge', ax=axs[1], data=trial_data_wo_baseline).set_title('Average team knowledge')
    # axs[0].set_ylabel('Number of interactions')
    # axs[1].set_ylabel('Average team knowledge')
    # axs[0].set_xlabel('Demo strategy')
    # axs[1].set_xlabel('Demo strategy')

    # fig, axs = plt.subplots(1, 2, figsize=(15,10))

    # sns.boxplot(x='team_composition', y='max_loop_count', ax=axs[0], data=trial_data_wo_baseline).set_title('Number of interactions')
    # sns.boxplot(x='team_composition', y='average_team_knowledge', ax=axs[1], data=trial_data_wo_baseline).set_title('Average team knowledge')
    # axs[0].set_ylabel('Number of interactions')
    # axs[1].set_ylabel('Average team knowledge')
    # axs[0].set_xlabel('Team composition')
    # axs[1].set_xlabel('Team composition')

    # plt.show()

    # outliers = [14, 229] # identified from the boxplot; analysis_run_id
    # trial_data_wo_baseline = trial_data_wo_baseline[~trial_data_wo_baseline['analysis_run_id'].isin(outliers)]


    # print(colored('N_interactions Two-way ANOVA', 'blue'))
    # # Perform two-way ANOVA
    # model = ols('max_loop_count ~ C(demo_strategy) * C(team_composition)', data=trial_data_wo_baseline).fit()
    # anova_results = sm.stats.anova_lm(model, typ=2)

    # print(anova_results)

    # # Tukey
    # print(colored('N_interactions Tukey for demo strategy', 'blue'))
    # tukey_results_demo = pairwise_tukeyhsd(endog=trial_data_wo_baseline['max_loop_count'], groups=trial_data_wo_baseline['demo_strategy'], alpha=0.05)
    # print(tukey_results_demo)

    # print(colored('N_interactions Tukey for team composition', 'blue'))
    # tukey_results_team = pairwise_tukeyhsd(endog=trial_data_wo_baseline['max_loop_count'], groups=trial_data_wo_baseline['team_composition'], alpha=0.05)
    # print(tukey_results_team)

    # # # d= trial_data.describe(include = [np.number])
    # # # print(d)
    # d2 = trial_data_wo_baseline.describe(include = [np.number])
    # print(d2)

    # # Perform two-way ANOVA
    # print(colored('Average team knowledge Two-way ANOVA', 'blue'))
    # model = ols('average_team_knowledge ~ C(demo_strategy) * C(team_composition)', data=trial_data_wo_baseline).fit()
    # anova_results = sm.stats.anova_lm(model, typ=2)

    # print(anova_results)

    # # Tukey
    # print(colored('Average team knowledge Tukey for demo strategy', 'blue'))
    # tukey_results_demo = pairwise_tukeyhsd(endog=trial_data_wo_baseline['average_team_knowledge'], groups=trial_data_wo_baseline['demo_strategy'], alpha=0.05)
    # print(tukey_results_demo)

    # print(colored('Average team knowledge Tukey for team composition', 'blue'))
    # tukey_results_team = pairwise_tukeyhsd(endog=trial_data_wo_baseline['average_team_knowledge'], groups=trial_data_wo_baseline['team_composition'], alpha=0.05)
    # print(tukey_results_team)

    # # Interaction 

    # ind = trial_data_wo_baseline['demo_strategy']=='individual_knowledge_low'
    # trial_data_wo_baseline.loc[ind, 'demo_strategy'] = '1.individual_knowledge_low'

    # ind = trial_data_wo_baseline['demo_strategy']=='individual_knowledge_high'
    # trial_data_wo_baseline.loc[ind, 'demo_strategy'] = '2.individual_knowledge_high'

    # ind = trial_data_wo_baseline['demo_strategy']=='common_knowledge'
    # trial_data_wo_baseline.loc[ind, 'demo_strategy'] = '3.common_knowledge'

    # ind = trial_data_wo_baseline['demo_strategy']=='joint_knowledge'
    # trial_data_wo_baseline.loc[ind, 'demo_strategy'] = '4.joint_knowledge'

    # trial_data_wo_baseline['demo_strategy'] = trial_data_wo_baseline['demo_strategy'].astype('str')
    # trial_data_wo_baseline['team_composition'] = trial_data_wo_baseline['team_composition'].astype('str')

    # trial_data_wo_baseline['Interaction'] = trial_data_wo_baseline['demo_strategy'].astype(str) + '-' + trial_data_wo_baseline['team_composition'].astype(str)


    # # print(colored('Average team knowledge and Team composition interaction Tukey for team composition', 'blue'))
    # # tukey_results_team = pairwise_tukeyhsd(endog=trial_data_wo_baseline['average_team_knowledge'], groups=trial_data_wo_baseline['Interaction'], alpha=0.05)
    # # print(tukey_results_team)

    # # # separate ANOVAs

    # # for level in trial_data_wo_baseline['team_composition'].unique():
    # #     subset = trial_data_wo_baseline[trial_data_wo_baseline['team_composition'] == level]
    # #     model = ols('average_team_knowledge ~ demo_strategy', data=subset).fit()
    # #     anova_table = anova_lm(model)
        
    # #     print(f"ANOVA for Level {level}:")
    # #     print(anova_table)

    # #     p_values = anova_table['PR(>F)'].values
    # #     adjusted_p_values = p_values * len(p_values)  # Bonferroni correction

        
    # #     print('adjusted_p_values:', adjusted_p_values)

    # #     print(colored(f'Tukey for Level {level}', 'blue'))
    # #     tukey_results_team = pairwise_tukeyhsd(endog=subset['average_team_knowledge'], groups=subset['demo_strategy'], alpha=0.05)
    # #     print(tukey_results_team)

    
    # # for level in trial_data_wo_baseline['team_composition'].unique():
    # #     subset = trial_data_wo_baseline[trial_data_wo_baseline['team_composition'] == level]
    # #     model = ols('max_loop_count ~ demo_strategy', data=subset).fit()
    # #     anova_table = anova_lm(model)
        
    # #     print(f"ANOVA, max_loop_count, for Level {level}:")
    # #     print(anova_table)

    # #     p_values = anova_table['PR(>F)'].values
    # #     adjusted_p_values = p_values * len(p_values)  # Bonferroni correction

        
    # #     print('adjusted_p_values:', adjusted_p_values)

    # #     print(colored(f'Tukey for Level {level}', 'blue'))
    # #     tukey_results_team = pairwise_tukeyhsd(endog=subset['max_loop_count'], groups=subset['demo_strategy'], alpha=0.05)
    # #     print(tukey_results_team)


    # # print(type(trial_data_wo_baseline['demo_strategy'].iloc[0]))
    # # print(type(trial_data_wo_baseline['team_composition'].iloc[0]))

    # trial_data_wo_baseline = trial_data_wo_baseline.reset_index(drop=True)

    # colors = sns.color_palette("colorblind", 7).as_hex()

    # colors = [str(color) for color in colors]

    # fig = interaction_plot(x=trial_data_wo_baseline['demo_strategy'], trace=trial_data_wo_baseline['team_composition'], response=trial_data_wo_baseline['max_loop_count'],
    #                        colors=[colors[0], colors[1], colors[2], colors[3]], markers=['D', '^', 'o', 's'], ms=10, linewidth = 10)
    
    # ax1 = plt.gca()
    # ax1.set_title('No. of interactions vs. Demo Strategy and Team composition', fontsize=28)
    # ax1.set_xlabel('Demo Strategy')
    # ax1.set_ylabel('No. of interactions')
    # ax1.legend_.remove()

    # # plot error bars
    # # Calculate standard errors for each group
    # groups = trial_data_wo_baseline.groupby(['demo_strategy', 'team_composition']).agg({'max_loop_count':['mean', 'sem']})
    # # groups = trial_data_wo_baseline.groupby(['demo_strategy', 'team_composition'])
    # # means = groups.mean()['max_loop_count']
    # # std_errors = groups.sem()['max_loop_count']  # Change to .std() for standard deviation, if preferred

    # print(groups)
    # # Add error bars
    # for (demo_strat, team_comp), row_data in groups.iterrows():
    #     # error = std_errors(demo_strat, team_comp)
    #     # mean = means(demo_strat, team_comp)

    #     error = row_data['max_loop_count']['sem']
    #     mean = row_data['max_loop_count']['mean']

    #     if demo_strat == '1.individual_knowledge_low':
    #         x_val = 0
    #     elif demo_strat == '2.individual_knowledge_high':
    #         x_val = 1
    #     elif demo_strat == '3.common_knowledge':
    #         x_val = 2
    #     else:
    #         x_val = 3

    #     if team_comp == '[0, 0, 0]':
    #         ax1.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[0], capsize=8)
    #     elif team_comp == '[0, 0, 2]':
    #         ax1.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[1], capsize=8)
    #     elif team_comp == '[0, 2, 2]':
    #         ax1.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[2], capsize=8)
    #     else:
    #         ax1.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[3], capsize=8)
    #     # ax1.errorbar(x_val, mean, yerr=error, fmt='none', color='black', capsize=5)




    # fig2 = interaction_plot(x=trial_data_wo_baseline['demo_strategy'], trace=trial_data_wo_baseline['team_composition'], response=trial_data_wo_baseline['average_team_knowledge'],
    #                        colors=[colors[0], colors[1], colors[2], colors[3]], markers=['D', '^', 'o', 's'], ms=10,  linewidth = 10)
    
    # ax2 = plt.gca()
    # ax2.set_title('Average team knowledge vs. Demo Strategy and Team composition', fontsize=28)
    # ax2.set_xlabel('Demo Strategy')
    # ax2.set_ylabel('Average team knowledge, $\mathregular{p_{BEC}}$')
    # ax2.legend_.remove()


    # # plot error bars
    # # Calculate standard errors for each group
    # groups = trial_data_wo_baseline.groupby(['demo_strategy', 'team_composition']).agg({'average_team_knowledge':['mean', 'sem']})
    # # groups = trial_data_wo_baseline.groupby(['demo_strategy', 'team_composition'])
    # # means = groups.mean()['max_loop_count']
    # # std_errors = groups.sem()['max_loop_count']  # Change to .std() for standard deviation, if preferred

    # print(groups)
    # # Add error bars
    # for (demo_strat, team_comp), row_data in groups.iterrows():
    #     # error = std_errors(demo_strat, team_comp)
    #     # mean = means(demo_strat, team_comp)

    #     error = row_data['average_team_knowledge']['sem']
    #     mean = row_data['average_team_knowledge']['mean']

    #     if demo_strat == '1.individual_knowledge_low':
    #         x_val = 0
    #     elif demo_strat == '2.individual_knowledge_high':
    #         x_val = 1
    #     elif demo_strat == '3.common_knowledge':
    #         x_val = 2
    #     else:
    #         x_val = 3

    #     if team_comp == '[0, 0, 0]':
    #         ax2.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[0], capsize=8)
    #     elif team_comp == '[0, 0, 2]':
    #         ax2.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[1], capsize=8)
    #     elif team_comp == '[0, 2, 2]':
    #         ax2.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[2], capsize=8)
    #     else:
    #         ax2.errorbar(x_val, mean, yerr=error, fmt='none', color=colors[3], capsize=8)
    #     # ax1.errorbar(x_val, mean, yerr=error, fmt='none', color='black', capsize=5)

    # plt.show()

    # pvals = []
    # ## BonFerroni
    # reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

    ##############################################

    # kc_id_list = concept_data['kc_id'].unique()

    # for kc_id in kc_id_list:
    #     kc_data = concept_data[concept_data['kc_id'] == kc_id]
    #     kc_data['team_composition'] = kc_data['team_composition'].astype('str')

    #     print(colored('N_interactions Two-way ANOVA for KC: ' + str(kc_id), 'blue'))
    #     # Perform two-way ANOVA
    #     model = ols('N_interactions_learn_concept ~ C(demo_strategy) * C(team_composition)', data=kc_data).fit()
    #     anova_results = sm.stats.anova_lm(model, typ=2)
    #     print(anova_results)

    #     # ## Tukey
    #     print(colored('N_interactions Tukey for demo strategy for KC: ' + str(kc_id), 'blue'))
    #     tukey_results_demo = pairwise_tukeyhsd(endog=kc_data['N_interactions_learn_concept'], groups=kc_data['demo_strategy'], alpha=0.05)
    #     print(tukey_results_demo)

    #     print(colored('N_interactions Tukey for team composition for KC: ' + str(kc_id), 'blue'))
    #     tukey_results_team = pairwise_tukeyhsd(endog=kc_data['N_interactions_learn_concept'], groups=kc_data['team_composition'], alpha=0.05)
    #     print(tukey_results_team)

    #     # Perform two-way ANOVA
    #     print(colored(' Average team knowledge Two-way ANOVA for KC: ' + str(kc_id), 'blue'))
    #     model = ols('avg_BEC_prob_learner_after_feedback ~ C(demo_strategy) * C(team_composition)', data=kc_data).fit()
    #     anova_results = sm.stats.anova_lm(model, typ=2)
    #     print(anova_results)

    #     # ## Tukey
    #     print(colored(' Average team knowledge Tukey for demo strategy for KC: ' + str(kc_id), 'blue'))
    #     tukey_results_demo = pairwise_tukeyhsd(endog=kc_data['avg_BEC_prob_learner_after_feedback'], groups=kc_data['demo_strategy'], alpha=0.05)
    #     print(tukey_results_demo)

    #     print(colored(' Average team knowledge Tukey for team composition for KC: ' + str(kc_id), 'blue'))
    #     tukey_results_team = pairwise_tukeyhsd(endog=kc_data['avg_BEC_prob_learner_after_feedback'], groups=kc_data['team_composition'], alpha=0.05)
    #     print(tukey_results_team)



    # ######################################################

    # #########  Dependent variables ########################


    # with open(path + '/' + vars_filename_prefix+ '_demo_strategies_subset.pickle', 'rb') as f:
    #     demo_constraints_data = pickle.load(f)

    with open(path + '/' + vars_filename_prefix+ '_demo_strategies_subset_N15_checked.pickle', 'rb') as f:
        demo_constraints_data = pickle.load(f)
    
    # print(colored('Two-way ANOVA: Mean distance between sampled counterfactuals vs. Demo Strategy and Team composition', 'blue'))
    # # Perform two-way ANOVA
    # model = ols('pairwise_dist_mean ~ C(demo_strategy) * C(team_composition)', data=demo_constraints_data).fit()
    # anova_results = sm.stats.anova_lm(model, typ=2)
    # print(anova_results)

    # print(colored('Two-way ANOVA: Std. dev distance between sampled counterfactuals vs. Demo Strategy and Team composition', 'blue'))
    # # Perform two-way ANOVA
    # model = ols('pairwise_dist_std ~ C(demo_strategy) * C(team_composition)', data=demo_constraints_data).fit()
    # anova_results = sm.stats.anova_lm(model, typ=2)
    # print(anova_results)

    # print(colored('Two-way ANOVA: Running BEC constraints area vs. Demo Strategy and Team composition', 'blue'))
    # # Perform two-way ANOVA
    # model = ols('running_BEC_constraints_area ~ C(demo_strategy) * C(team_composition)', data=demo_constraints_data).fit()
    # anova_results = sm.stats.anova_lm(model, typ=2)
    # print(anova_results)
        
    print(colored('Two-way ANOVA: Scale BEC constraints area vs. Demo Strategy and Team composition', 'blue'))
    # Perform two-way ANOVA
    model = ols('scale_reduce_BEC_constraints_area ~ C(demo_strategy) * C(team_composition)', data=demo_constraints_data).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
    print(anova_results)

    ## Tukey
    print(colored('scale_reduce_BEC_constraints_area Tukey for demo strategy', 'blue'))
    tukey_results_demo = pairwise_tukeyhsd(endog=demo_constraints_data['scale_reduce_BEC_constraints_area'], groups=demo_constraints_data['demo_strategy'], alpha=0.05)
    print(tukey_results_demo)
    
    # ## Tukey
    # print(colored('Running BEC constraints area Tukey for demo strategy', 'blue'))
    # tukey_results_demo = pairwise_tukeyhsd(endog=demo_constraints_data['running_BEC_constraints_area'], groups=demo_constraints_data['demo_strategy'], alpha=0.05)
    # print(tukey_results_demo)

    
    # print(colored('Two-way ANOVA: KC_constraints area vs. Demo Strategy and Team composition', 'blue'))
    # # Perform two-way ANOVA
    # model = ols('KC_constraints_area ~ C(demo_strategy) * C(team_composition)', data=demo_constraints_data).fit()
    # anova_results = sm.stats.anova_lm(model, typ=2)
    # print(anova_results)
    
    # ## Tukey
    # print(colored('Running BEC constraints area Tukey for demo strategy', 'blue'))
    # tukey_results_demo = pairwise_tukeyhsd(endog=demo_constraints_data['KC_constraints_area'], groups=demo_constraints_data['demo_strategy'], alpha=0.05)
    # print(tukey_results_demo)
        
    # colors = sns.color_palette("colorblind", 7).as_hex()

    # colors = [str(color) for color in colors]


    # # fig = interaction_plot(x=demo_constraints_data['demo_strategy'], trace=demo_constraints_data['kc_id'], response=demo_constraints_data['KC_constraints_area'],
    # #                 colors=[colors[0], colors[1], colors[2]], markers=['D', '^', 'o'], ms=10)

    # # fig2  = interaction_plot(x=demo_constraints_data['demo_strategy'], trace=demo_constraints_data['kc_id'], response=demo_constraints_data['running_BEC_constraints_area'],
    # #                 colors=[colors[0], colors[1], colors[2]], markers=['D', '^', 'o'], ms=10)

    # # plt.show()

    # order_list = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']
    # demo_labels = ['', '', 'Ind. (high)', 'Common']


    # f2, ax_2 = plt.subplots(ncols=1)
    # sns.barplot(data = demo_constraints_data, x = 'demo_strategy', y = 'KC_constraints_area', order=order_list, ax=ax_2, errorbar=('se',1)).set(title='Demo constraints area vs. Demo Strategy')
    # # sns.barplot(data = demo_constraints_data, x = 'team_composition', y = 'KC_constraints_area', ax=ax_2[1], errorbar=('se',1)).set(title='KC constraints area vs. Team composition')
    # ax_2.set_xticklabels(demo_labels)
    # plt.show()


    # kc_id_list = demo_constraints_data['kc_id'].unique()

    # for kc_id in kc_id_list:
    #     kc_data = demo_constraints_data[demo_constraints_data['kc_id'] == kc_id]


    #     print(colored('Demo constraints area Two-way ANOVA for kc id: ' + str(kc_id), 'blue'))
    #     # Perform two-way ANOVA
    #     model = ols('running_BEC_constraints_area ~ C(demo_strategy) * C(team_composition)', data=kc_data).fit()
    #     anova_results = sm.stats.anova_lm(model, typ=2)

    #     print(anova_results)

    #     ## Tukey
    #     print(colored('Demo constraints area Tukey for demo strategy for kc id: ' + str(kc_id), 'blue'))
    #     tukey_results_demo = pairwise_tukeyhsd(endog=kc_data['running_BEC_constraints_area'], groups=kc_data['demo_strategy'], alpha=0.05)
    #     print(tukey_results_demo)

        # print(colored('Demo constraints area for team composition for kc id: ' + str(kc_id), 'blue'))
        # tukey_results_team = pairwise_tukeyhsd(endog=kc_data['running_BEC_constraints_area'], groups=kc_data['team_composition'], alpha=0.05)
        # print(tukey_results_team)
        

        # ind = demo_constraints_data['demo_strategy']=='individual_knowledge_low'
        # demo_constraints_data.loc[ind, 'demo_strategy'] = '1. individual_knowledge_low'

        # ind = demo_constraints_data['demo_strategy']=='individual_knowledge_high'
        # demo_constraints_data.loc[ind, 'demo_strategy'] = '2. individual_knowledge_high'

        # ind = demo_constraints_data['demo_strategy']=='common_knowledge'
        # demo_constraints_data.loc[ind, 'demo_strategy'] = '3. common_knowledge'

        # ind = demo_constraints_data['demo_strategy']=='joint_knowledge'
        # demo_constraints_data.loc[ind, 'demo_strategy'] = '4. joint_knowledge'

        # fig = interaction_plot(x=demo_constraints_data['demo_strategy'], trace=demo_constraints_data['kc_id'], response=demo_constraints_data['KC_constraints_area'],
        #                     colors=['red', 'blue', 'green'], markers=['D', '^', 'o'], ms=10)

        # plt.show()

####################################

def find_corrupt_file(path, files, file_prefix_list, runs_to_exclude_list = []):


    # runs_to_exclude_list = []
    excluded_exp_runs = [runs_to_exclude_list]
    file_read_error_flag = True
    exp_id = 0

    files_to_exclude_list = []
    
    while file_read_error_flag:
    
        try:
            print(colored('Experiment number: ' + str(exp_id+1) + '. Excluding runs: ' + str(excluded_exp_runs[exp_id]), 'blue'))
            # print(files)
            
            for file in files:
                # check if file is a valid pickle file
                
                for file_prefix in file_prefix_list:
                    if file_prefix in file and '.pickle' in file and '.png' not in file:
                        run_file_flag = True
                        break
                    else:
                        run_file_flag = False


                # check if file needs to be excluded
                for runs_to_exclude in excluded_exp_runs[exp_id]:
                    if runs_to_exclude in file:
                        run_file_flag = False
                        break
                

                if run_file_flag:
                    with open(path + '/' + file, 'rb') as f:
                        sim_vars = pickle.load(f)
                    print('Opening file: ', file)

                    run_id = int(file.split('_')[-1].split('.')[0])
                    if exp_id == 0:
                        runs_to_exclude_list.append(file)

                    ###########################
                    # if sim_vars['demo_strategy'].iloc[0] == 'joint_knowledge':
                    #     files_to_exclude_list.append(file)


                        ###########################

            file_read_error_flag = False
        except:
            print(colored('Error reading file: ' + file, 'red'))
            print('Runs to check: ', runs_to_exclude_list)
            excluded_exp_runs.append([runs_to_exclude_list[exp_id]])
            exp_id += 1
            file_read_error_flag = True


    ####################

    # # remove joint knowledge files
    # for file in files_to_exclude_list:
    #     file_prefix = file.split('run')[0]
    #     print('Removing file: ', file)
    #     print('File prefix: ', file_prefix)

    #     print('Removing file: ', file)
    #     os.remove(path + '/' + file)
    #     csv_file = file.split('.')[0] + '.csv'
    #     print('Removing file: ', csv_file)
    #     os.remove(path + '/' + csv_file)

    #     for prefix in ['update_data_', 'prob_data_']:
    #         file_remove = prefix + file_prefix + 'run' + file.split('run')[1].split('.')[0]
    #         print('Removing file: ', file_remove)
    #         os.remove(path + '/' + file_remove + '.csv')


    
            

######################################
            
def analyze_incomplete_runs(path, file):

    with open(path + '/' + file, 'rb') as f:
        incomplete_runs = pickle.load(f)


    run_data_list = []


    for id, run in incomplete_runs.iterrows():
        run_file = run['file_name']

        if run['sim_status'] == 'No new demos' or run['sim_status'] == 'Teaching complete':
            with open(path + '/' + run_file, 'rb') as f:
                sim_vars = pickle.load(f)

                study_id = sim_vars['study_id'].iloc[0]
                run_id = sim_vars['run_no'].iloc[0]
                team_composition = sim_vars['team_composition'].iloc[0]
                demo_strategy = sim_vars['demo_strategy'].iloc[0]

                for i, int_data in sim_vars.iterrows():
                    run_data_dict = {'study_id': study_id, 'run_id': run_id, 'team_composition': team_composition, 'demo_strategy': demo_strategy, 'sim_status': run['sim_status'], \
                                     'int_id': int_data['loop_count']}
                    
                    run_data_dict['kc_id'] = int_data['knowledge_comp_id']
                    run_data_dict['min_KC_constraints'] = int_data['min_KC_constraints']

                    BEC_knowledge_team = 0
                    for member in int_data['BEC_knowledge_level']:
                        run_data_dict['BEC_knowledge_level_' + member] = int_data['BEC_knowledge_level'][member]
                        if 'p' in member:
                            # print(int_data['BEC_knowledge_level'][member])
                            mem_know = [float(x) for x in int_data['BEC_knowledge_level'][member]]
                            # print(mem_know)
                            BEC_knowledge_team += np.sum(mem_know)

                    run_data_dict['BEC_knowledge_team'] = BEC_knowledge_team/3
                
                    run_data_list.append(run_data_dict)

    run_data_df = pd.DataFrame(run_data_list)
    run_data_df.to_csv(path + '/incomplete_runs_analyzed.csv')
    with open(path + '/incomplete_runs_analyzed.pickle', 'wb') as f:
        pickle.dump(run_data_df, f)


#######################################
        
def analyze_demo_strategies(path, files, file_prefix_list, runs_to_exclude_list = [], vars_filename_prefix = ''):

    # with open(path + '/' + vars_filename_prefix + '_trial_data_subset.pickle', 'rb') as f:
    #     trial_data = pickle.load(f)

    # run_list = trial_data['analysis_run_id'].unique()
    # trial_data['BEC_know_learn_goal'] = np.zeros(len(trial_data), dtype=bool)

    # for id, run in trial_data.iterrows():

    #     file = run['file_name']

    #     with open(path + '/' + file, 'rb') as f:
    #         sim_vars = pickle.load(f)
        
    #     bec_final = sim_vars['BEC_knowledge_level'].iloc[-1]

    #     # check if learning was completed
    #     learning_complete = True
    #     for k_type, k_val in bec_final.items():
    #         if k_val[0] != 1:
    #             learning_complete = False
    #             break

    #     if learning_complete:
    #         trial_data['BEC_know_learn_goal'].loc[id] = True

    # trial_data.to_csv(path + '/' + vars_filename_prefix + 'trial_data.csv')
    # with open(path + '/' + vars_filename_prefix + 'trial_data.pickle', 'wb') as f:
    #     pickle.dump(trial_data, f)


        



    try:
        with open(path + '/' + vars_filename_prefix + '_demo_strategies_subset_N15_checked2.pickle', 'rb') as f:
            run_data_df_subset = pickle.load(f)
        print('Opening file: ', vars_filename_prefix + '_demo_strategies_subset_N15_checked.pickle')

    except:
        run_data_list = []
        run_data_learning_incomplete_list = []
        analysis_run_id = 1

        with open(path + '/' + vars_filename_prefix + '_trial_data_subset_N15_checked.pickle', 'rb') as f:
            trial_data = pickle.load(f)

        unique_runs = trial_data['analysis_run_id'].unique()


        for file in files:
            # check if file is a valid pickle file
            
            for file_prefix in file_prefix_list:
                if file_prefix in file and '.pickle' in file and '.png' not in file and 'counterfactual' not in file:
                    run_file_flag = True
                    break
                else:
                    run_file_flag = False


            # check if file needs to be excluded
            for runs_to_exclude in runs_to_exclude_list:
                if runs_to_exclude in file:
                    run_file_flag = False
                    break
            

            if run_file_flag:
                with open(path + '/' + file, 'rb') as f:
                    sim_vars = pickle.load(f)
                print('Opening file: ', file)

                
                study_id = sim_vars['study_id'][0]
                sim_run_id = sim_vars['run_no'][0]

                # check if learning was completed based on final BEC knowledge
                # learning_complete = True
                # for k_type, k_val in bec_final.items():
                #     if k_val[0] != 1:
                #         learning_complete = False
                #         break

                # check if learning was completed based on sim status flag
                if sim_vars['sim_status'].iloc[-1] == 'Teaching complete':
                    learning_complete = True
                else:
                    learning_complete = False

                # read counterfactuals sampled models data
                folder = file.strip('.pickle')
                demo_file_name = path + '/ind_sim_trials/' + folder + '/demo_gen_log.txt'

                try:
                    # counterfactuals = pd.read_csv(path + '/' + folder + '_counterfactuals.csv')
                    with open(path + '/ind_sim_trials/' + folder + '_counterfactuals.pickle', 'rb') as f:
                        counterfactuals = pickle.load(f)
                    print('Already compiled counterfactuals data. Reading from file. ')
                except:
                    if 'missing' in file:
                        counterfactuals = read_demo_gen_log(demo_file_name, new_flag=True)
                    else:
                        counterfactuals = read_demo_gen_log(demo_file_name, new_flag=True)
                 #######################################
                        
                # for i in range(len(counterfactuals)):
                #     counterfactuals['sample_human_models'].iloc[i] = [np.array(x).astype(float) for x in counterfactuals['sample_human_models'].iloc[i]]

                # print('counterfactuals: ', counterfactuals['sample_human_models'])
                # print('type: ', type(counterfactuals['sample_human_models'].iloc[0]))

                ############## compile data
                study_id = sim_vars['study_id'].iloc[0]
                run_id = sim_vars['run_no'].iloc[0]
                team_composition = str(sim_vars['team_composition'].iloc[0])
                demo_strategy = str(sim_vars['demo_strategy'].iloc[0])
                sim_status = str(sim_vars['sim_status'].iloc[-1])
                min_BEC_constraints = sim_vars['min_BEC_constraints'].iloc[-1]
                running_BEC_constraints = params.prior

                # filter extra sims in same file
                kc_id_list = sim_vars['knowledge_comp_id']
                run_start_id = []
                for id in range(len(kc_id_list)-1):
                    if kc_id_list[id+1] - kc_id_list[id] < 0:
                        print(colored('Duplicate trials found for file: ' + file + '. Run id: ' + str(sim_run_id), 'red' ))
                        run_start_id = [id+1]
                        break
                
                if len(run_start_id) != 0:
                    sim_vars = sim_vars.iloc[run_start_id[0]:].reset_index(drop=True)
                ##################
                    
                for i, int_data in sim_vars.iterrows():
                    # print('int_data: ', int_data)
                    run_data_dict = {'study_id': study_id, 'run_id': run_id, 'analysis_run_id': analysis_run_id, 'team_composition': team_composition, 'demo_strategy': demo_strategy, 'sim_status': sim_status, \
                                        'int_id': int_data['loop_count']}
                    
                    run_data_dict['kc_id'] = int_data['knowledge_comp_id']
                    run_data_dict['min_KC_constraints'] = int_data['min_KC_constraints']

                    running_BEC_constraints.extend(int_data['min_KC_constraints'])

                    run_data_dict['KC_constraints_area'] = BEC_helpers.calc_solid_angles([int_data['min_KC_constraints']])[0]
                    run_data_dict['BEC_constraints_area'] = BEC_helpers.calc_solid_angles([int_data['min_BEC_constraints']])[0]
                    running_BEC_constraints = BEC_helpers.remove_redundant_constraints(running_BEC_constraints, params.weights['val'], params.step_cost_flag)
                    run_data_dict['running_BEC_constraints'] = copy.deepcopy(running_BEC_constraints)
                    run_data_dict['running_BEC_constraints_area'] = BEC_helpers.calc_solid_angles([running_BEC_constraints])[0]

                    if i > 0:
                        run_data_dict['delta_BEC_constraints_area'] = running_BEC_constraints_area_prev - run_data_dict['running_BEC_constraints_area']
                        run_data_dict['scale_reduce_BEC_constraints_area'] = running_BEC_constraints_area_prev/run_data_dict['running_BEC_constraints_area'] # area reduction (information gain) # prior BEC area (step cost is negative)
                    else:
                        run_data_dict['delta_BEC_constraints_area'] = 2*np.pi - run_data_dict['running_BEC_constraints_area'] # area reduction (information gain) # prior BEC area (step cost is negative)
                        run_data_dict['scale_reduce_BEC_constraints_area'] = 2*np.pi/run_data_dict['running_BEC_constraints_area'] # area reduction (information gain) # prior BEC area (step cost is negative)
                    
                    
                    BEC_knowledge_team = 0
                    for member in int_data['BEC_knowledge_level']:
                        run_data_dict['BEC_knowledge_level_' + member] = int_data['BEC_knowledge_level'][member]
                        if 'p' in member:
                            # print(int_data['BEC_knowledge_level'][member])
                            mem_know = [float(x) for x in int_data['BEC_knowledge_level'][member]]
                            # print(mem_know)
                            BEC_knowledge_team += np.sum(mem_know)

                    run_data_dict['BEC_knowledge_team'] = BEC_knowledge_team/3

                    # find sampled counterfactuals
                    summary_count = int_data['summary_count']
                    counterfactual_models_list = counterfactuals[counterfactuals['length_of_summary']==summary_count]['sample_human_models'].reset_index(drop=True)
                    # print(counterfactuals)
                    # print(summary_count)
                    # print(counterfactual_models_list)
                    counterfactual_models = counterfactual_models_list.iloc[0]
                    counterfactual_models = [np.array(x).astype(float) for x in counterfactual_models]
                    # print(counterfactual_models)
                    # print('type: ', type(counterfactual_models[0]))

                    # calculate how close the sampled points are
                    counterfactual_models_latllong = cg.cart2latlong(counterfactual_models)
                    pairwise = metrics.pairwise.haversine_distances(counterfactual_models_latllong)
                    # print(pairwise)
                    run_data_dict['pairwise_dist_mean'] = np.mean(pairwise)
                    run_data_dict['pairwise_dist_std'] = np.std(pairwise)
                    run_data_dict['sampled_models_cluster_idxs'] = counterfactuals[counterfactuals['length_of_summary']==summary_count]['sampled_models_cluster_idxs'].iloc[0]
                    run_data_dict['cluster_weights'] = counterfactuals[counterfactuals['length_of_summary']==summary_count]['cluster_weights'].iloc[0]

                    # update end of loop vars
                    running_BEC_constraints_area_prev = run_data_dict['running_BEC_constraints_area']

                    # print(run_data_dict)
                    if learning_complete:
                        run_data_list.append(run_data_dict)
                    else:
                        run_data_learning_incomplete_list.append(run_data_dict)
                    

                    with open(path + '/ind_sim_trials/' + folder + '_counterfactuals.pickle', 'wb') as f:
                        pickle.dump(counterfactuals, f)
                    counterfactuals.to_csv(path + '/ind_sim_trials/' + folder + '_counterfactuals.csv')

                ####################
                if learning_complete:
                    analysis_run_id += 1
        
        run_data_df = pd.DataFrame(run_data_list)
        run_data_incomplete_df = pd.DataFrame(run_data_learning_incomplete_list)

        with open(path + '/' + vars_filename_prefix + '_demo_strategies_N15_checked.pickle', 'wb') as f:
            pickle.dump(run_data_df, f)

        run_data_df.to_csv(path + '/' + vars_filename_prefix + '_demo_strategies_N15_checked.csv')


        with open(path + '/' + vars_filename_prefix + '_demo_strategies_learning_incomplete.pickle', 'wb') as f:
            pickle.dump(run_data_incomplete_df, f)
        
        run_data_incomplete_df.to_csv(path + '/' + vars_filename_prefix + '_demo_strategies_learning_incomplete.csv')

        index_to_choose = run_data_df['analysis_run_id'].isin(unique_runs).index
        run_data_df_subset = run_data_df.iloc[index_to_choose]

        with open(path + '/' + vars_filename_prefix + '_demo_strategies_subset_N15_checked.pickle', 'wb') as f:
            pickle.dump(run_data_df_subset, f)

        run_data_df_subset.to_csv(path + '/' + vars_filename_prefix + '_demo_strategies_subset_N15_checked.csv')


    # #####################################

    # # How are the sampled models and constraints different?

    # # sampled points

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # sns.barplot(data=run_data_df_subset, x='demo_strategy', y='pairwise_dist_mean', order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'], ax=ax[0])
    # sns.barplot(data=run_data_df_subset, x='demo_strategy', y='pairwise_dist_std', order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'], ax=ax[1])

    # fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    # sns.barplot(data=run_data_df_subset, x='team_composition', y='pairwise_dist_mean',  ax=ax2[0])
    # sns.barplot(data=run_data_df_subset, x='team_composition', y='pairwise_dist_std', ax=ax2[1])

    # # fig3, ax3 = plt.subplots(1, 2, figsize=(10, 5))
    # # sns.barplot(data=run_data_df_subset, x='demo_strategy', y='pairwise_dist_mean', hue='team_composition', order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'], ax=ax3[0])
    # # sns.barplot(data=run_data_df_subset, x='demo_strategy', y='pairwise_dist_std', hue = 'team_composition', order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'], ax=ax3[1])

    # plt.show()

    # ###########################

    # # # Unique constraints
    # unique_kc_lookup = pd.read_csv(path + '/unique_KC_constraints_lookup.csv')
    # unique_kc_lookup['min_KC_constraints_str'] = unique_kc_lookup['min_KC_constraints'].astype(str)

    # run_data_df_subset['min_KC_constraints_str'] = run_data_df_subset['min_KC_constraints'].astype(str)
    
    # # #######################
    # unique_kc_cons = run_data_df_subset['min_KC_constraints_str'].unique()
    
    # unique_kc_area = []
    # unique_kc_id = []
    # for i in range(len(unique_kc_cons)):
    #     index = run_data_df_subset[run_data_df_subset['min_KC_constraints_str'] == unique_kc_cons[i]].index[0]
    #     min_const = BEC_helpers.remove_redundant_constraints(run_data_df_subset['min_KC_constraints'].iloc[index], params.weights['val'], params.step_cost_flag)
    #     unique_kc_area.append(np.round(BEC_helpers.calc_solid_angles([min_const])[0],3))
    #     unique_kc_id.append(run_data_df_subset['kc_id'].iloc[index])

    # unique_kc_cons_df = pd.DataFrame(unique_kc_cons, columns=['min_KC_constraints'])
    # unique_kc_cons_df['KC_constraints_area'] = unique_kc_area
    # unique_kc_cons_df['kc_id'] = unique_kc_id


    # unique_kc_cons_df.to_csv(path + '/' + vars_filename_prefix + '_unique_kc_cons.csv')
    # # # ######################
    
    ############## plots ####################
        
    ## Information content of constraints
    
    # # Histogram of contraints area ratio for each demo strategy
    
    # fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    # fig2, ax2 = plt.subplots(2, 2, figsize=(10, 5))

    # row_id, col_id = 0, 0
    # for demo in ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']:
    #     demo_data_non_repeat = run_data_df_subset[(run_data_df_subset['demo_strategy'] == demo) & (run_data_df_subset['scale_reduce_BEC_constraints_area']!=1)]
    #     demo_data = run_data_df_subset[(run_data_df_subset['demo_strategy'] == demo)]
    #     sns.histplot(demo_data_non_repeat['scale_reduce_BEC_constraints_area'], ax=ax[row_id, col_id], label=demo, kde=True).set(title='Constraints area change ratio for demo ' + str(demo), xlabel='Area reduction (information gain)', ylabel='Frequency')

    #     sns.lineplot(data=demo_data, x='int_id', y='scale_reduce_BEC_constraints_area', ax=ax2[row_id, col_id], label=demo).set(title='Constraints area change ratio for demo ' + str(demo), xlabel='Interaction id', ylabel='Area reduction (information gain)')

    #     if row_id == 1:
    #         row_id = 0
    #         col_id += 1
    #     else:
    #         row_id += 1
    
    # plt.show()
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.barplot(data=run_data_df_subset, x='demo_strategy', y='scale_reduce_BEC_constraints_area', order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'], ax=ax)

    plt.show()

    # sns.barplot(data=run_data_df_subset, x='demo_strategy', y='KC_constraints_area', order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'])
    # axn = plt.gca()
    # axn.set_title('Demo constraints area vs. Demo Strategy')
    # axn.set_xlabel('Demo Strategy')
    # axn.set_ylabel('Demo constraints area (1/information content)')
    
    # plt.show()


    # # # print('N runs: ', len(run_data_df['analysis_run_id'].unique()))

    # #kc wise plotes
    # kc_id_list = run_data_df_subset['kc_id'].unique()
    # outlier = [] # analysis_run_id

    # for kc_id in kc_id_list:

    #     kc_run_data = run_data_df_subset[run_data_df_subset['kc_id'] == kc_id]


    #     print(kc_id)
    #     print(kc_run_data['min_KC_constraints_str'].unique())

    #     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #     sns.barplot(data=kc_run_data, x='demo_strategy', y='KC_constraints_area', ax=ax)

    #     fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    #     sns.barplot(data=kc_run_data, x='demo_strategy', hue='team_composition', y='KC_constraints_area', ax=ax2)
    
    #     plt.show()


    #     # unique constraints for each demo
    #     demo_list = kc_run_data['demo_strategy'].unique()
    #     for demo in demo_list:
    #         unique_kc_cons = kc_run_data[kc_run_data['demo_strategy']==demo]['min_KC_constraints_str'].unique()
    #         print('demo: ', demo, 'kc_id: ', kc_id, 'N unique_kc_cons: ', len(unique_kc_cons), 'unique_kc_cons: ', unique_kc_cons)

    #     ## knowledge for each KC for each demo
        # fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))

        # runs = kc_run_data['analysis_run_id'].unique()
        # KC_end_data_list = []
        # for run in runs:
        #     if run not in outlier:
        #         run_data = kc_run_data[kc_run_data['analysis_run_id'] == run].iloc[-1].to_dict()

        #         KC_end_data_list.append(run_data)

        # KC_end_data_df = pd.DataFrame(KC_end_data_list)


        # sns.barplot(data=KC_end_data_df, x='demo_strategy', y='running_BEC_constraints_area', order = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge'], ax=ax3)

        # plt.show()


    ################
    # fig4, ax4 = plt.subplots(1, 1, figsize=(10, 5))
    # # Count repetitions
    # unique_cnst_id = np.zeros(len(run_data_df_subset))
    # for i in range(len(run_data_df_subset)):
    #     unique_cnst_id[i] = unique_kc_lookup[unique_kc_lookup['min_KC_constraints_str'] == run_data_df_subset['min_KC_constraints_str'].iloc[i]]['unique_constraint_id'].iloc[0]
    # run_data_df_subset['unique_constraint_id'] = unique_cnst_id

    # kc_id_list = run_data_df_subset['kc_id'].unique()
    # for kc_id in kc_id_list:
    #     kc_run_data = run_data_df_subset[run_data_df_subset['kc_id'] == kc_id]
    #     # print(kc_run_data['unique_constraint_id'].unique())
    #     repetitions = kc_run_data.groupby(['team_composition', 'demo_strategy', 'unique_constraint_id']).size().reset_index(name='Count')

    #     # Plot stacked bar chart using seaborn with hue
    #     sns.barplot(y='Count', x='unique_constraint_id', data=repetitions, ax=ax4)
    #     sns.barplot(y='Count', x='unique_constraint_id', hue='team_composition', data=repetitions,  ax=ax4)
    #     sns.barplot(y='Count', x='unique_constraint_id', hue='demo_strategy',  data=repetitions, ax=ax4[1])
        
        
    #     fig5, ax5 = plt.subplots(1, 1, figsize=(10, 5))
    #     sns.barplot(y='Count', x='demo_strategy', hue='team_composition', data=repetitions, ax=ax5)

    # plt.show()

    ########################

    # # Constraints area
    # fig6, ax6 = plt.subplots(1, 2, figsize=(10, 5))
    # sns.barplot(data=run_data_df, x='demo_strategy', y='KC_constraints_area', hue='kc_id', ax=ax6[0])
    # sns.barplot(data=run_data_df, x='team_composition', y='KC_constraints_area', hue='kc_id', ax=ax6[1])
    
    # plt.show()


###################################
    
def split_multiple_runs(path, files, file_prefix_list, runs_to_exclude_list = []):
    

    for file in files:
        # check if file is a valid pickle file
        
        for file_prefix in file_prefix_list:
            if file_prefix in file and '.pickle' in file and '.png' not in file:
                run_file_flag = True
                break
            else:
                run_file_flag = False


        # check if file needs to be excluded
        for runs_to_exclude in runs_to_exclude_list:
            if runs_to_exclude in file:
                run_file_flag = False
                break
        

        if run_file_flag:
            with open(path + '/' + file, 'rb') as f:
                sim_vars = pickle.load(f)
            print('Opening file: ', file)

            # check if there are multiple runs in the same file
            loop_count_var = sim_vars['loop_count']
            run_change_idx = [idx for idx in range(len(loop_count_var)-1) if loop_count_var[idx] > loop_count_var[idx+1]]

            print('Run change idx: ', run_change_idx)
            if len(run_change_idx) > 0:

                for run_idx in range(len(run_change_idx)+1):
                    if run_idx == 0:
                        run_sim_vars = sim_vars.iloc[:run_change_idx[run_idx]+1]
                    elif run_idx == len(run_change_idx):
                        run_sim_vars = sim_vars.iloc[run_change_idx[run_idx-1]+1:]
                    else:
                        run_sim_vars = sim_vars.iloc[run_change_idx[run_idx]+1:run_change_idx[run_idx+1]+1]
                        
                    # reset index
                    run_sim_vars = run_sim_vars.reset_index(drop=True)

                    print('Saving new file: ', path + '/' + file.split('.pickle')[0] + '_run_' + str(run_idx) + '.pickle')

                    with open(path + '/' + file.split('.pickle')[0] + '_run_' + str(run_idx) + '.pickle', 'wb') as f:
                        pickle.dump(run_sim_vars, f)
                    
                    run_sim_vars.to_csv(path + '/' + file.split('.pickle')[0] + '_run_' + str(run_idx) + '.csv')

####################################
if __name__ == "__main__":

    # process team knowledge data
    path = 'models/augmented_taxi2'
    # # path = 'data/simulation/sim_experiments/w_feedback'
    files = os.listdir(path)

    # all_file_prefix_list = ['debug_data_debug_trials_01_22_no_noise_w_feedback_study_1']
    # all_runs_to_exclude_list = [[3, 12, 24, 7], [1,4,6,8], [], [1,3, 11, 12, 16, 18], [17, 21, 35], [], [], [], \
    #                             [], [], [], [], [], [], []]
    all_runs_to_exclude_list = []

    # sets_to_consider = [14]
    # file_prefix_list = [all_file_prefix_list[i] for i in sets_to_consider]
    # runs_to_exclude_list = [all_runs_to_exclude_list[i] for i in sets_to_consider]

    # file_prefix_list = ['03_04_sim_study_test_demo_strategies', '03_04_sim_study_forte_wo_test_sampling_restriction', '03_08_missing_sim_study_2_dup_no_restrictions_study_1']
    # file_prefix_list = [file_prefix_list[1]]
    # file_prefix_list = ['03_04_sim_study_forte_wo_test_sampling_restriction']
    # file_prefix_list = ['03_07_sim_study_no_dup_tests_N8_noise_study_1']
    # file_prefix_list = ['03_18_sim_study_no_dup_N12_sample_cluster_weight_low_noise_no_sampling']
    # file_prefix_list = ['03_08_sim_study_no_dup_N12_sample_cluster_weights_study_1']

    file_prefix_list = ['03_08_sim_study_no_dup_N6_sample_cluster_weights_study_1_run_22']
    
    runs_to_exclude_list = ['unfinished'] 
    # runs_to_exclude_list = ['low_noise']
    
   
    # runs_to_exclude_ids = []
    # runs_to_exclude_list = []
    # for id in runs_to_exclude_ids:
    #     runs_to_exclude_list.append(file_prefix_list[0] + '_run_' + str(id))
    
    
    
    # trials_01_01_w_updated_noise_57.csv - outlier, N = 48 trials

    # vars_filename_prefix = 'sim_study_no_dup_tests_N8_noise'
    # vars_filename_prefix = 'sim_study_2_dup_wo_rest_lc_flag'
    # vars_filename_prefix = 'sim_study_no_dup_N12_sample_cluster_weights'
    # vars_filename_prefix = 'sim_study_no_dup_N6'
    # vars_filename_prefix = 'sim_study_no_dup_N12_sample_cluster_weights_study_1' 
    vars_filename_prefix = 'sim_study_no_dup_N6_sample_cluster_weights'
    # vars_filename_prefix = 'sim_study_no_dup_N6_sample_cluster_weights_low_noise'
    # vars_filename_prefix = 'sim_study_no_dup_N12_sample_cluster_weight_low_noise_no_sampling'

    # print(file_prefix_list)
    # print(runs_to_exclude_list)
    
    # find_corrupt_file(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list)

    # run_analysis_script(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list, vars_filename_prefix = vars_filename_prefix)
    # sim_study_data_analysis(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list, vars_filename_prefix = vars_filename_prefix)
    # sim_study_data_analysis(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list, vars_filename_prefix = vars_filename_prefix, max_trials=6)
    # sim_study_data_analysis(path, files, file_prefix_list, data_condition='baseline', runs_to_exclude_list = runs_to_exclude_list, vars_filename_prefix = vars_filename_prefix)


    # split_multiple_runs(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list)


    # combine_baseline_data(path)
    # sim_study_plots(path, vars_filename_prefix)
    # anova(path, vars_filename_prefix)
    
    analyze_demo_strategies(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list, vars_filename_prefix = vars_filename_prefix)

    # analyze_incomplete_runs(path, 'forte_study_no_samp_rest_learning_incomplete_runs.pickle')

    ##################################################
    # path = 'models/augmented_taxi2/'

    # filename = '03_04_sim_study_test_demo_strategies_study_1_run_5.pickle'

    # plot_pf_dist(path, filename, counterfactuals_plot_flag=True)

    ###########################################

    # # # # # ## Sensitivity Analysis
    # path = 'models/augmented_taxi2/'
    # files = os.listdir(path)

    # file_prefix_list = ['03_17_sensitivity_tc2_jk']
    # # file_prefix_list = ['03_04_sensitivity_tc2_jk_lfl_study_1_run_5']

    # runs_excluse_list = ['fixed', '03_04_sensitivity_tc2_jk_lfl_study_1_run_5', '03_04_sensitivity_tc2_jk_lfl_study_1_run_4', '03_04_sensitivity_tc2_jk_lfl_study_2_run_4', \
    #                      '03_04_sensitivity_tc2_jk_lfl_study_4_run_4']
    # run_sensitivity_analysis(path, files, file_prefix_list, runs_to_exclude_list=runs_excluse_list, runs_to_analyze_list = [], vars_filename_prefix = vars_filename_prefix)




    ##############################
    # ## Analyze response sampling tests
    
    # path = 'data/simulation/sim_experiments'
    # files = os.listdir(path)
    # file_prefix_list = ['debug_trial_12_29_noise_particles']
    # file_to_avoid = ''
    # response_data, const_prob_data = analyze_human_response_simulation(path, files, file_prefix_list, file_to_avoid)
    # response_data['learning_factor'] = np.round(response_data['learning_factor'].astype(float), 3)
    # const_prob_data['learning_factor'] = np.round(const_prob_data['learning_factor'].astype(float), 3)


    # print('const_prob_data: ', const_prob_data)

    # ## process response data
    # # cols = ['condition', 'update_id', 'learning_factor', 'skip_model_flag', 'response_type', 'cluster_id', 'particles_prob']
    # cols = ['set_id', 'update_id', 'learning_factor', 'skip_model_flag', 'response_type', 'cluster_id', 'particles_prob']
    # response_data_selected = response_data[cols]

    # # print('response_data_selected: ', response_data_selected)

    # response_data_valid = response_data_selected[response_data_selected['skip_model_flag'] == False]
    # # print('response_data_valid: ', response_data_valid)

    # response_data_correct = response_data_selected[response_data_selected['response_type'] == 'correct']
    # # print('response_data_correct: ', response_data_correct)

    # # response_prob = response_data_correct.groupby(['condition', 'update_id', 'learning_factor', 'particles_prob']).agg({'cluster_id':'count'})                                                                                                 
    # # response_prob_den = response_data_valid.groupby(['condition', 'update_id', 'learning_factor','particles_prob']).agg({'cluster_id':'count'})

    # response_prob_num = response_data_correct.groupby(['set_id', 'learning_factor', 'update_id']).agg({'cluster_id':'count', 'particles_prob': 'mean'})                                                                                                 
    # response_prob = response_data_valid.groupby(['set_id', 'learning_factor', 'update_id']).agg({'cluster_id':'count', 'particles_prob': 'mean'})

    # response_prob['probability'] = response_prob_num['cluster_id']/response_prob['cluster_id']
    # response_prob['probability'] = response_prob['probability'].fillna(0)
    # response_prob.to_csv(path + '/response_prob.csv')

    # # response_prob.rename(columns={'cluster_id': 'correct_test_probability'}, inplace=True)


    # print('response_prob: ', response_prob)
    # # print('response_prob_num: ', response_prob_num)
    # # print('response_prob_den: ', response_prob_den)
    # ########

    # # process constraint prob data
    # cols = ['set_id', 'update_id', 'learning_factor', 'skip_model_flag', 'response_type', 'constraint', 'cluster_id', 'prob_initial', 'prob_reweight', 'prob_resample']
    # cnst_data_selected = const_prob_data[cols]
    # cnst_data_selected_valid = cnst_data_selected[cnst_data_selected['skip_model_flag'] == False]    
    # cnst_data_selected_correct = cnst_data_selected[cnst_data_selected['response_type'] == 'correct']

    # # cnst_prob = cnst_data_selected_correct.groupby(['set_id', 'learning_factor',  'update_id', 'constraint']).agg({'cluster_id':'count', 'prob_initial': 'mean', 'prob_reweight': 'mean', 'prob_resample': 'mean'})                                                                                                 
    # # cnst_prob_den = cnst_data_selected_valid.groupby(['set_id', 'learning_factor',  'update_id', 'constraint']).agg({'cluster_id':'count'})

    # # cnst_prob['probability'] = cnst_prob['cluster_id']/cnst_prob_den['cluster_id']
    # # # print('cnst_prob: ', cnst_prob.columns)
    # # cnst_prob['update_prob_delta'] = cnst_prob['prob_reweight'] - cnst_prob['prob_initial']


    # # # ###### plot response data
    # # f3, ax3 = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,6))
    # # plt.subplots_adjust(wspace=0.1, hspace=0.1)        
    # # sns.lineplot(response_prob, x = 'learning_factor', y = 'particles_prob', ax=ax3[0], legend=True).set(title='Learning factor vs. particles_probability')
    # # sns.lineplot(response_prob, x = 'learning_factor', y = 'particles_prob', hue = 'update_id', ax=ax3[1], legend=True).set(title='particles_probability vs. probability of correct response based on number of PF updates/interactions')
 
    # # f, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,6))
    # # plt.subplots_adjust(wspace=0.1, hspace=0.1)        
    # # sns.lineplot(response_prob, x = 'learning_factor', y = 'probability', ax=ax[0], legend=True).set(title='Learning factor vs. probability of correct response')
    # # sns.lineplot(response_prob, x = 'learning_factor', y = 'probability', hue = 'update_id', ax=ax[1], legend=True).set(title='Learning factor vs. probability of correct response based on number of PF updates/interactions')
    
    # f2, ax2 = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)        
    # sns.regplot(response_prob, x = 'particles_prob', y = 'probability', ax=ax2).set(title='particles_probability vs. probability of correct response')
    # # sns.lmplot(response_prob, x = 'particles_prob', y = 'probability', col='update_id').set(title='particles_probability vs. probability of correct response based on number of PF updates/interactions')
    # # sns.scatterplot(response_prob, x = 'particles_prob', y = 'probability', hue='update_id').set(title='particles_probability vs. probability of correct response based on number of PF updates/interactions')

    # # # sns.regplot(data=response_prob[response_prob['update_id']==0], x = 'particles_prob', y = 'probability', ax=ax2[1]).set(title='particles_probability vs. probability of correct response based on number of PF updates/interactions')
    # # # sns.regplot(data=response_prob[response_prob['update_id']==1], x = 'particles_prob', y = 'probability', ax=ax2[1])
    # # # sns.regplot(data=response_prob[response_prob['update_id']==2], x = 'particles_prob', y = 'probability', ax=ax2[1])
    # # # sns.regplot(data=response_prob[response_prob['update_id']==3], x = 'particles_prob', y = 'probability', ax=ax2[1])
    # # # sns.regplot(data=response_prob[response_prob['update_id']==4], x = 'particles_prob', y = 'probability', ax=ax2[1])

    # plt.show()

    # ##################

    ###### plot constraint data
    # f4, ax4 = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)        
    # sns.regplot(cnst_prob, x = 'prob_initial', y = 'prob_reweight', ax=ax4[0]).set(title='prob_initial vs. prob_reweight')
    # sns.regplot(data=cnst_prob[cnst_prob['update_id']==0], x = 'prob_initial', y = 'prob_reweight', ax=ax4[1]).set(title='prob_initial vs. prob_reweight based on number of PF updates/interactions')
    # sns.regplot(data=cnst_prob[cnst_prob['update_id']==1], x = 'prob_initial', y = 'prob_reweight', ax=ax4[1])
    # sns.regplot(data=cnst_prob[cnst_prob['update_id']==2], x = 'prob_initial', y = 'prob_reweight', ax=ax4[1])
    # sns.regplot(data=cnst_prob[cnst_prob['update_id']==3], x = 'prob_initial', y = 'prob_reweight', ax=ax4[1])
    # sns.regplot(data=cnst_prob[cnst_prob['update_id']==4], x = 'prob_initial', y = 'prob_reweight', ax=ax4[1])

    # f5, ax5 = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(10,6))
    # sns.histplot(cnst_prob, x = 'update_prob_delta', hue = 'update_id', ax=ax5, legend=True).set(title='Change in probability of constraint after reweighting')
    # sns.histplot(cnst_prob, x = 'update_prob_delta', ax=ax5, legend=True).set(title='Change in probability of constraint after reweighting')
    # sns.scatterplot(cnst_prob, x = 'learning_factor', y = 'update_prob_delta', ax=ax5, legend=True).set(title='Change in probability of constraint after reweighting')

    # f2, ax2 = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)        
    # sns.lineplot(response_prob, x = 'particles_prob', y = 'probability', ax=ax2[0], legend=True).set(title='particles_probability vs. probability of correct response for various sampling conditions')
    # sns.lineplot(response_prob, x = 'particles_prob', y = 'probability', hue = 'update_id', ax=ax2[1], legend=True).set(title='particles_probability vs. probability of correct response based on number of PF updates/interactions')
    
    # f, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)        
    # sns.lineplot(response_prob, x = 'learning_factor', y = 'probability', ax=ax[0], legend=True).set(title='Learning factor vs. probability of correct response for various sampling conditions')
    # sns.lineplot(response_prob, x = 'learning_factor', y = 'probability', hue = 'update_id', ax=ax[1], legend=True).set(title='Learning factor vs. probability of correct response based on number of PF updates/interactions')
  
    # plt.show()

    # # plot concept interaction data
    # team_long_data = pd.read_csv('models/augmented_taxi2/team_knowledge_level_long.csv')
    # sns.boxplot(team_long_data, x='loop_count'y='knowledge_comp_id')

    ##################

    # path = 'data/simulation/sampling_tests'
    # files = os.listdir(path)
    # file_prefix = 'particles_positions'
    # check_pf_particles_sampling(path, files, file_prefix)


    ##############################

    # ## Analyze individual runs

    # # path = 'data/simulation/sim_experiments/new_data'
    # path = 'models/augmented_taxi2'
    # file = ''

    # # analyze_individual_runs(path, 'trials_01_09_regular_study_1_run_100.pickle')

    # # plot_prob_ind_run(path, 'trials_01_09_regular_study_1_run_100.pickle')

    # simulate_individual_runs(path, 'debug_trials_01_09_no_noise_study_1_run_8.pickle')

    # ###########################
    # # ## Analyze particle filter update process
    # path = 'models/augmented_taxi2'
    # file_prefix = 'debug_trials_01_23_no_noise_w_feedback_study_1_run_70'
    # # file_prefix = 'debug_trials_01_09_no_noise_study_1_run_8'

    # plot_pf_updates(path, file_prefix)





    ##################
    x = 1




    