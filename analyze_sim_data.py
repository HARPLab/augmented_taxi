# Analyze simulation data

import pandas as pd
import ast
import json
import teams.teams_helpers as teams_helpers
import params_team as params
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
from termcolor import colored
import warnings
import textwrap

from ast import literal_eval
import copy

import teams.teams_helpers as team_helpers
import params_team as params

warnings.simplefilter(action='ignore', category=FutureWarning)


def str_to_dict(string, var_type = None, splitter = ', '):
    # remove the curly braces from the string
    # print('String before: ', string)
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


def run_analysis_script(path, files, file_prefix_list, runs_to_exclude_list=[], runs_to_analyze_list = []):
    

    team_unit_knowledge_level = pd.DataFrame()
    team_BEC_knowledge_level_expected = pd.DataFrame()
    team_BEC_knowledge_level = pd.DataFrame()
    team_knowledge = pd.DataFrame()
    learning_rate = pd.DataFrame()
    likelihood_correct_response = pd.DataFrame()
    particles_prob = pd.DataFrame()
    learning_incomplete_runs = pd.DataFrame()

    # intiialize unique simulation run id
    run_id = 0

    for file in files:

        # check if file is a valid csv
        for file_prefix in file_prefix_list:
            if file_prefix in file and '.csv' in file:
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

            sim_vars = pd.read_csv(path + '/' + file)

            # check if learning was completed
            bec_final = sim_vars['BEC_knowledge_level'][len(sim_vars)-1]
            BEC_team_knowledge_final= str_to_dict(bec_final, var_type = 'float')
            learning_complete = True
            for k_type, k_val in BEC_team_knowledge_final.items():
                if k_val != 1:
                    learning_complete = False
                    break

            
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
                    if type(tk) == str and type(bec_k) == str and type(bec_k_e) == str and type(tkc) == str:
                        
                        # Unit knowledge level
                        unit_knowledge_dict = str_to_dict(tk, var_type = 'float')
                        unit_knowledge_dict['run_no'] = run_id
                        unit_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        unit_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        unit_knowledge_dict['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        unit_knowledge_dict['file_name'] = file
                        team_unit_knowledge_level = team_unit_knowledge_level.append(unit_knowledge_dict, ignore_index=True)

                        # BEC knowledge level 
                        BEC_team_knowledge_dict = str_to_dict(bec_k, var_type = 'float')
                        BEC_team_knowledge_dict['run_no'] = run_id
                        BEC_team_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        BEC_team_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        BEC_team_knowledge_dict['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        BEC_team_knowledge_dict['team_composition'] = sim_vars['team_composition'][i]
                        BEC_team_knowledge_dict['file_name'] = file
                        team_BEC_knowledge_level = team_BEC_knowledge_level.append(BEC_team_knowledge_dict, ignore_index=True)
                    
                        # expected BEC knowledge level
                        BEC_team_knowledge_dict_expected = str_to_dict(bec_k_e, var_type = 'float')
                        BEC_team_knowledge_dict_expected['run_no'] = run_id
                        BEC_team_knowledge_dict_expected['loop_count'] = int(sim_vars['loop_count'][i])
                        BEC_team_knowledge_dict_expected['demo_strategy'] = sim_vars['demo_strategy'][i]
                        BEC_team_knowledge_dict_expected['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        BEC_team_knowledge_dict_expected['file_name'] = file
                        team_BEC_knowledge_level_expected = team_BEC_knowledge_level_expected.append(BEC_team_knowledge_dict_expected, ignore_index=True)

                        # # knowledge mix condition
                        # learning_rate_dict = str_to_dict(ilv, var_type = 'array', splitter='),')
                        # learning_rate_dict['run_no'] = run_id
                        # learning_rate_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        # learning_rate_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        # learning_rate = learning_rate.append(learning_rate_dict, ignore_index=True)
                        # # print('learning_rate_dict: ', learning_rate_dict)

                        # team knowledge constraints
                        team_knowledge_dict = str_to_dict(tkc, splitter = ', \'')
                        team_knowledge_dict['run_no'] = run_id
                        team_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        team_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        team_knowledge_dict['file_name'] = file
                        team_knowledge = team_knowledge.append(team_knowledge_dict, ignore_index=True)

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


                        # particles probability
                        team_particles_probability_dict = str_to_dict(sim_vars['particles_prob_learner_demo'][i], var_type=float)
                        lf = sim_vars['team_learning_factor'][i]
                        lf = lf.strip('[]')
                        lf = lf.split(' ')
                        lf = [i for i in lf if i != '']
                        # print('lcr: ', lcr)
                        lf_array = np.array(list(lf), dtype=float)
                        # print('lf_array: ', lf_array)

                        kc_variables = sim_vars['variable_filter'][i]
                        if i==0:
                            current_kc_variable = kc_variables
                        if current_kc_variable != kc_variables:
                            update_id = 1
                            current_kc_variable = kc_variables

                        if kc_variables == '[[0. 1. 0.]]':
                            kc_id = 1
                        elif kc_variables == '[[1. 0. 0.]]':
                            kc_id = 2
                        elif kc_variables == '[[0. 0. 0.]]':
                            kc_id = 3
                        else:
                            print(colored('Unregognized variable filter: ' + kc_variables, 'red'))
                        
                        for p_id, player in enumerate(team_particles_probability_dict):
                            particles_probability_dict = {}
                            particles_probability_dict['run_no'] = run_id
                            particles_probability_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                            particles_probability_dict['team_composition'] = sim_vars['team_composition'][i]
                            particles_probability_dict['loop_count'] = int(sim_vars['loop_count'][i])
                            particles_probability_dict['update_id'] = update_id
                            particles_probability_dict['kc_id'] = kc_variables
                            particles_probability_dict['player_id'] = player
                            particles_probability_dict['learning_factor'] = lf_array[p_id]
                            particles_probability_dict['particles_prob'] = float(team_particles_probability_dict[player])
                            particles_probability_dict['file_name'] = file

                            particles_prob = particles_prob.append(particles_probability_dict, ignore_index=True)

                        
                        update_id += 1

                    else:
                        print(colored('Some non-string variables...','red'))


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
                print(colored('Leanring incomplete for file: ' + file + '. Run id: ' + str(run_id), 'red' ))
                learning_incomplete_runs_dict = {}
                learning_incomplete_runs_dict['run_no'] = run_id
                learning_incomplete_runs_dict['file_name'] = file
                learning_incomplete_runs_dict['team_composition'] = sim_vars['team_composition'][0]
                learning_incomplete_runs_dict['demo_strategy'] = sim_vars['demo_strategy'][0]
                learning_incomplete_runs_dict['max_loop_count'] = sim_vars['loop_count'].iloc[-1]
                learning_incomplete_runs = learning_incomplete_runs.append(learning_incomplete_runs_dict, ignore_index=True)
            
            ######
            run_id += 1


    # process team knowledge data
    BEC_knowledge_level = []
    BEC_knowledge_level_expected = []
    knowledge_type = []
    ind_knowledge_type = []
    normalized_loop_count = []
    lcr_var = []
    team_mix_var = []
    team_knowledge_level_min = team_BEC_knowledge_level.copy(deep=True)
    print('team_knowledge_level_min columns: ', team_knowledge_level_min.columns)
    team_knowledge_level_min = team_knowledge_level_min.drop(['p1', 'p2', 'p3', 'common_knowledge', 'joint_knowledge'], axis=1)


    # Long format data
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

    
    team_knowledge_level_long.to_csv(path + '/team_knowledge_level_long.csv')
    team_knowledge_level_long.describe(include='all').to_csv(path + '/descriptives.csv')
    # team_knowledge_level_long = pd.read_csv('models/augmented_taxi2/team_knowledge_level_long.csv')

    
    # ############## concept-wise interaction count
    # concept_ids = team_BEC_knowledge_level['knowledge_comp_id'].unique()
    # unique_ids = team_knowledge_level_long['run_no'].unique()
    # interaction_count = pd.DataFrame()

    # print(team_BEC_knowledge_level)

    # for id in unique_ids:
    #     interaction_count_dict = {}
    #     for c_id in range(len(concept_ids)):
    #         interaction_count_dict['run_no'] = id
    #         interaction_count_dict['demo_strategy'] = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['demo_strategy'].iloc[0]
    #         interaction_count_dict['team_composition'] = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['team_composition'].iloc[0]
            
    #         print('A: ', team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['loop_count'])
    #         print('run_no: ', id, ' c_id: ', c_id, ' concept_ids[c_id+1]: ', concept_ids[c_id])
            
    #         if concept_ids[c_id] <= team_BEC_knowledge_level['knowledge_comp_id'].iloc[-1]:
    #             max_loop_id = team_BEC_knowledge_level[(team_BEC_knowledge_level['run_no'] == id) & (team_BEC_knowledge_level['knowledge_comp_id'] == concept_ids[c_id])]['loop_count'].iloc[-1]
    #             max_loop_count = max_loop_id - team_BEC_knowledge_level[(team_BEC_knowledge_level['run_no'] == id) & (team_BEC_knowledge_level['knowledge_comp_id'] == concept_ids[c_id])]['loop_count'].iloc[0] + 1
            
    #         # else:
    #         #     max_loop_id = []
    #         #     max_loop_count = 0

    #         interaction_count_dict['Int_end_id_concept_'+str(concept_ids[c_id])] = max_loop_id
    #         interaction_count_dict['N_int_concept_'+str(concept_ids[c_id])] = max_loop_count    

    #     interaction_count = interaction_count.append(interaction_count_dict, ignore_index=True)

    # print('interaction_count: ', interaction_count)
    # interaction_count.to_csv('models/augmented_taxi2/interaction_count.csv')
    # ######################################################################
    

    ## run-wise data
    unique_ids = team_knowledge_level_long['run_no'].unique()
    run_data = pd.DataFrame()
    for id in unique_ids:
        run_data_dict = {}
        run_data_dict['run_no'] = id
        run_data_dict['demo_strategy'] = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['demo_strategy'].iloc[0]
        run_data_dict['max_loop_count'] = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['loop_count'].iloc[-1]
        run_data_dict['team_composition'] = team_knowledge_level_long[team_knowledge_level_long['run_no'] == id]['team_composition'].iloc[0]
        run_data_dict['file_name'] = team_knowledge_level_long[team_knowledge_level_long['run_no'] == id]['file_name'].iloc[0]
        
        run_data = run_data.append(run_data_dict, ignore_index=True)

    # run_data.to_csv(path + '/run_data.csv')

    print(colored('Number of runs processed: ' + str(len(run_data)), 'red'))

    # run_data = pd.read_csv('models/augmented_taxi2/run_data.csv')

    print('Incomplete runs: ', learning_incomplete_runs)
    learning_incomplete_runs.to_csv(path + '/learning_incomplete_runs.csv')


    ## normalize loop count

    # for id in unique_ids:
    #     idx = team_knowledge_level_long[(team_knowledge_level_long['run_no'] == id)].index
    #     max_loop_count = np.max(team_knowledge_level_long.loc[idx, 'loop_count'])
    #     team_knowledge_level_long.loc[idx, 'normalized_loop_count'] = team_knowledge_level_long.loc[idx, 'loop_count']/max_loop_count


    # print(team_knowledge_level_long)
    # print(knowledge_type)

    ##############################################   Plots    ##########################################################



    # for know_id in ['p1', 'p2', 'p3', 'common_knowledge', 'joint_knowledge']:
    #     sns.lineplot(data = team_BEC_knowledge_level, x = 'loop_count', y = know_id, hue = 'demo_strategy').set(title='Knowledge level for ' + know_id)
    #     plt.show()
    #     # plt.savefig('models/augmented_taxi2/BEC_knowledge_level_' + know_id + '.png')
    #     # plt.close()
    
    
    # f, ax = plt.subplots(nrows=2,ncols=3, sharex=True, sharey=True, figsize=(10,6))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # row_id = 0
    # col_id = 0
    # for team_mix_cond in [[0, 1, 1]]:
    #     col_id = 0
    #     for know_id in ['individual', 'common_knowledge', 'joint_knowledge']:    
    #         plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) & (team_knowledge_level_long['team_mix']==str(team_mix_cond))]
    #         # plot_data.to_csv('models/augmented_taxi2/plot_data_' + know_id + '_' + str(team_mix_cond) + '.csv')
    #         print('plot_data: ', type(plot_data))
    #         sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
    #         col_id += 1 
    #     row_id += 1


    ###########
    
    ## Choose conditions to plot
    know_list_full = ['individual', 'common_knowledge', 'joint_knowledge']
    team_mix_full = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    demo_list = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']
    

    know_list = know_list_full[0:]
    team_mix = team_mix_full[0:]

    # Plot knowledge level for each combination of team composition and knowledge type
    col_id = 0
    for team_mix_cond in team_mix:
        col_id = 0
        f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for know_id in know_list:    
            plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) & (team_knowledge_level_long['team_composition']==str(team_mix_cond))]
            
            # plot_data.to_csv('models/augmented_taxi2/plot_data_' + know_id + '_' + str(team_mix_cond) + '.csv')
            # print('Plotting  ', 'row_id: ', row_id, ' col_id: ', col_id, ' know_id: ', know_id, ' team_mix_cond: ', team_mix_cond)
            sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge']).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
            # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))

            col_id += 1 
        
    # plt.show()
            
    # plt.savefig('models/augmented_taxi2/BEC_knowledge_level_' + know_id + '.png')
    # plt.close()
    ########
        
    ## plot knowledge level for all conditions
    f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    col_id = 0
    for know_id in know_list:  
        plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) ]
        sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for a team mix: ' + str(team_mix_cond))
        # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
        col_id += 1 
    # plt.show()
    ########

    ## plot knowledge level for all team composition
    for team_mix_cond in team_mix:
        f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        col_id = 0
        for know_id in know_list:  
            plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) &(team_knowledge_level_long['team_composition']==str(team_mix_cond))]
            sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for a team mix: ' + str(team_mix_cond))
            # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
            col_id += 1 
        # plt.show()

    # plot knowledge for demo strategy
    for demo_id in demo_list:
        f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10,6))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        col_id = 0
        for know_id in know_list: 
            plot_data = team_knowledge_level_long[(team_knowledge_level_long['knowledge_type']==know_id) &(team_knowledge_level_long['demo_strategy']==str(demo_id))]
            sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for demo strategy: ' + str(demo_id))
            # sns.lineplot(plot_data, x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[row_id, col_id], errorbar=('se', 1), err_style="band", hue_order = ['individual_knowledge_low','individual_knowledge_high','common_knowledge','joint_knowledge'], legend=False).set(title='Knowledge level for ' + know_id + ' with a team mix: ' + str(team_mix_cond))
            col_id += 1 
        # plt.show()
    
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
    # plot interaction count for concepts

    # f, ax_sbc = plt.subplots(ncols=1)
    # sns.barplot(data = interaction_count, x = 'knowledge_comp_id', y = 'max_loop_count', hue = 'demo_strategy', ax=ax_sbc, errorbar=('se',1)).set(title='Interaction count for concepts')



    # plot interaction count overall
    f, ax_c = plt.subplots(ncols=1)
    sns.barplot(data = run_data, x = 'demo_strategy', y = 'max_loop_count', hue = 'team_composition', ax=ax_c, errorbar=('se',1)).set(title='Max number of interactions')


    f2, ax_2 = plt.subplots(ncols=2)
    sns.barplot(data = run_data, x = 'demo_strategy', y = 'max_loop_count', ax=ax_2[0], errorbar=('se',1)).set(title='Max number of interactions vs. Demo Strategy')
    sns.barplot(data = run_data, x = 'team_composition', y = 'max_loop_count', ax=ax_2[1], errorbar=('se',1)).set(title='Max number of interactions vs. Team composition')
    #############

    ### plot probability of particles in the correct side of test
    # simulation run conditions
    dem_strategy_list = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']
    kc_id_list = ['[[0. 1. 0.]]', '[[1. 0. 0.]]', '[[0. 0. 0.]]']
    team_mix_list = ['[0, 0, 0]', '[0, 0, 2]', '[0, 2, 2]', '[2, 2, 2]']

    print('particles_prob: ', particles_prob)
    particles_prob.to_csv(path + '/particles_prob.csv')
    

    # ## plotting seprately for each team condition and demo strategy
    # for team_composition in team_mix_list:
    #     for dem_strategy in dem_strategy_list:

    #         run_data = particles_prob[(particles_prob['team_composition']==str(team_composition)) & (particles_prob['demo_strategy']==dem_strategy)]

    #         if len(run_data) > 0:
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
        
    #             plt.show()
    ########


    # ## plotting for all experiment conditions

    f3, ax3 = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10,6))
    plt.subplots_adjust(wspace=0.1, hspace=0.5)  

    kc_no = 0
    for kc_id in kc_id_list:

        plot_data = particles_prob[particles_prob['kc_id']==kc_id]
        if len(plot_data) > 0:
            print('plot_data: ', plot_data)
            plot_title = 'Learning factor vs. particles_probability for kc ' +  kc_id 
            wrapped_title = "\n".join(textwrap.wrap(plot_title, 40))
            sns.lineplot(plot_data, x = 'learning_factor', y = 'particles_prob', ax=ax3[0, kc_no], legend=True).set(title=wrapped_title)
            plot_title = 'Updatewise Learning factor vs. particles_probability for kc ' +  kc_id
            wrapped_title = "\n".join(textwrap.wrap(plot_title, 40))
            sns.lineplot(plot_data, x = 'learning_factor', y = 'particles_prob', hue = 'update_id', ax=ax3[1, kc_no], legend=True).set(title=wrapped_title)
            kc_no += 1

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


def analyze_individual_runs(path, file):

    # fixed params
    max_learning_factor = 1
    team_learning_rate =  np.hstack((0.05*np.ones([params.team_size, 1]), 0*np.ones([params.team_size, 1])))


    study_data = pd.read_pickle(path + '/' + file)

    ### plot particle probability for learner
    particles_prob_learner_demo = study_data['particles_prob_learner_demo']
    particles_prob_learner_demo_df = pd.DataFrame()

    for i, row in particles_prob_learner_demo.iteritems():
        # print(row)
        particles_prob_learner_demo_df = particles_prob_learner_demo_df.append(row, ignore_index=True)
    

    ### particle probability of teacher
    particles_prob_teacher_demo = study_data['particles_prob_teacher_demo']
    particles_prob_teacher_demo_df = pd.DataFrame()

    for i, row in particles_prob_teacher_demo.iteritems():
        # print(row)
        particles_prob_teacher_demo_df = particles_prob_teacher_demo_df.append(row, ignore_index=True)


    ### constraints
    demo_constraints = study_data['unit_constraints']
    test_constraints = study_data['test_constraints']
    test_responses_team = study_data['test_constraints_team']
    loop_count = study_data['loop_count']


    ### initialize teacher and learner particle filters
    initial_team_learning_factor = study_data['initial_team_learning_factor'].iloc[0]
    team_learning_factor = copy.deepcopy(initial_team_learning_factor)
    team_prior, teacher_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior)
    learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = team_learning_factor, team_prior = params.team_prior, pf_flag='learner')

    prob_learner_after_demo = pd.DataFrame()
    prob_teacher_after_demo = pd.DataFrame()
    prob_learner_before_test = pd.DataFrame()
    prob_learner_after_test = pd.DataFrame()
    prob_teacher_before_test = pd.DataFrame()
    prob_teacher_after_test = pd.DataFrame()

    # run pf updates only for individual knowledge
    for loop_id in range(len(loop_count)):
        demo_loop = demo_constraints[loop_id]
        test_loop = test_constraints[loop_id]
        test_responses_loop = test_responses_team[loop_id]
        
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
            teacher_pf[member_id].update(dem_cnsts)
            learner_pf[member_id].update(dem_cnsts, learning_factor = team_learning_factor[p_id])

            teacher_pf[member_id].calc_particles_probability(dem_cnsts)
            learner_pf[member_id].calc_particles_probability(dem_cnsts)
            prob_learner_after_demo_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_teacher_after_demo_dict[member_id] = teacher_pf[member_id].particles_prob_correct

        # update probability dataframes
        prob_learner_after_demo = prob_learner_after_demo.append(prob_learner_after_demo_dict, ignore_index=True)
        prob_teacher_after_demo = prob_teacher_after_demo.append(prob_teacher_after_demo_dict, ignore_index=True)


        # teacher and learner update for tests
        for test_id in range(len(test_loop)):
            test_cnsts = test_loop[test_id]
            print('test_cnsts: ', test_cnsts)

            prob_learner_before_test_dict = {'loop_id': loop_id+1, 'test_id': test_id, 'test_constraints': test_cnsts}
            prob_teacher_before_test_dict = {'loop_id': loop_id+1, 'test_id': test_id, 'test_constraints': test_cnsts}
            prob_learner_after_test_dict = {'loop_id': loop_id+1, 'test_id': test_id, 'test_constraints': test_cnsts}
            prob_teacher_after_test_dict = {'loop_id': loop_id+1, 'test_id': test_id, 'test_constraints': test_cnsts}

            for p_id in range(params.team_size):
                member_id = 'p' + str(p_id+1)

                test_response = test_responses_team[p_id][test_id]

                teacher_pf[member_id].calc_particles_probability(test_cnsts)
                learner_pf[member_id].calc_particles_probability(test_cnsts)
                prob_learner_before_test_dict[member_id] = learner_pf[member_id].particles_prob_correct
                prob_teacher_before_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct

                # update learning parameter
                if (test_response == test_cnsts[0]).all():
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
                    response_type = 'correct'
                else:
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)
                    response_type = 'incorrect'

                # update based on test responses
                print('Member: ', member_id, ' test_response: ', test_response)

                teacher_pf[member_id].update(test_response)  # note that test responses are ordered based on member at high level
                learner_pf[member_id].update(test_response, learning_factor = team_learning_factor[p_id])

                teacher_pf[member_id].calc_particles_probability(test_response)
                learner_pf[member_id].calc_particles_probability(test_response)
                prob_learner_after_test_dict[member_id] = learner_pf[member_id].particles_prob_correct
                prob_teacher_after_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct

                var_name = 'response_type_' + member_id

                prob_learner_after_test_dict[var_name] = response_type
                prob_teacher_after_test_dict[var_name] = response_type
                prob_learner_before_test_dict[var_name] = response_type
                prob_teacher_before_test_dict[var_name] = response_type


            ## update probability dataframes
            prob_learner_before_test = prob_learner_before_test.append(prob_learner_before_test_dict, ignore_index=True)
            prob_teacher_before_test = prob_teacher_before_test.append(prob_teacher_before_test_dict, ignore_index=True)
            prob_learner_after_test = prob_learner_after_test.append(prob_learner_after_test_dict, ignore_index=True)
            prob_teacher_after_test = prob_teacher_after_test.append(prob_teacher_after_test_dict, ignore_index=True)

    
    filename = path + '/' + file.split('.')[0]

    prob_learner_after_demo.to_csv(filename + '_prob_learner_after_demo.csv')
    prob_teacher_after_demo.to_csv(filename + '_prob_teacher_after_demo.csv')
    prob_learner_before_test.to_csv(filename + '_prob_learner_before_test.csv')
    prob_teacher_before_test.to_csv(filename + '_prob_teacher_before_test.csv')
    prob_learner_after_test.to_csv(filename + '_prob_learner_after_test.csv')
    prob_teacher_after_test.to_csv(filename + '_prob_teacher_after_test.csv')



    return 1
####################################################


if __name__ == "__main__":

    # # process team knowledge data
    # path = 'data/simulation/sim_experiments/new_data'
    # files = os.listdir(path)

    # # all_file_prefix_list = ['trials_12_17']
    # # all_runs_to_exclude_list = [[3, 12, 24, 7], [1,4,6,8], [], [1,3, 11, 12, 16, 18], [17, 21, 35], [], [], [], \
    # #                             [], [], [], [], [], [], []]
    # # all_runs_to_exclude_list = []

    # # sets_to_consider = [14]
    # # file_prefix_list = [all_file_prefix_list[i] for i in sets_to_consider]
    # # runs_to_exclude_list = [all_runs_to_exclude_list[i] for i in sets_to_consider]

    # file_prefix_list = ['trials_12_29_w_updated', 'trials_12_30_w_updated', 'trials_12_31_w_updated', 'trials_01_01_w_updated', 'trials_01_02_w_updated']
    # runs_to_exclude_list = ['unfinished', 'trials_01_01_w_updated_noise_57'] 
    # # trials_01_01_w_updated_noise_57.csv - outlier, N = 48 trials


    # print(file_prefix_list)
    # print(runs_to_exclude_list)
    
    # runs_to_analyze_list = []

    # run_analysis_script(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list, runs_to_analyze_list = runs_to_analyze_list)

    # # analyze_run_data(path, path + '/run_data.csv')




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

    #### Analyze individual runs

    path = 'data/simulation/sim_experiments/new_data'
    file = ''

    analyze_individual_runs(path, 'trials_12_29_w_updated_noise_8.pickle')








    ##################
    x = 1




    