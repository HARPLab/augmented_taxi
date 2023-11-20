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
     # print(files)

     
    team_unit_knowledge_level = pd.DataFrame()
    team_BEC_knowledge_level_expected = pd.DataFrame()
    team_BEC_knowledge_level = pd.DataFrame()
    team_knowledge = pd.DataFrame()
    learning_rate = pd.DataFrame()
    likelihood_correct_response = pd.DataFrame()

    for file in files:

        prefix_id = 0
        for file_prefix in file_prefix_list:
            if file_prefix in file and '.csv' in file:
                run_file_flag = True
                break
            else:
                run_file_flag = False
                prefix_id += 1

        if run_file_flag:

            sim_vars = pd.read_csv(path + '/' + file)

            if (len(runs_to_analyze_list) > 0 and len(runs_to_analyze_list[prefix_id]) > 0 and sim_vars['run_no'][0] in runs_to_analyze_list ) or \
                (len(runs_to_analyze_list) == 0 and ( len(runs_to_exclude_list) == 0 or len(runs_to_exclude_list[prefix_id]) == 0 or sim_vars['run_no'][0] not in runs_to_exclude_list[prefix_id]) ):

            # if sim_vars['run_no'][0] not in runs_to_exclude[0]:

                print(colored('Reading file: ' + file,'red' ))  
            
                for i in range(len(sim_vars)):

                    # unit knowledge level
                    tk = sim_vars['unit_knowledge_level'][i]
                    bec_k = sim_vars['BEC_knowledge_level'][i]
                    bec_k_e = sim_vars['BEC_knowledge_level_expected'][i]
                    tkc = sim_vars['team_knowledge'][i]
                    ilv = sim_vars['initial_likelihood_vars'][i]
                    lcr = sim_vars['likelihood_correct_response'][i]


                    if type(tk) == str and type(bec_k) == str and type(bec_k_e) == str and type(tkc) == str and type(ilv) == str and type(lcr) == str:
                        run_id = file_prefix_list[prefix_id] + '_' + str(sim_vars['run_no'][i])
                        print(colored('Parsing run id: ' + run_id, 'blue'))
                        
                        unit_knowledge_dict = str_to_dict(tk, var_type = 'float')
                        unit_knowledge_dict['run_no'] = run_id
                        unit_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        unit_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        unit_knowledge_dict['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        team_unit_knowledge_level = team_unit_knowledge_level.append(unit_knowledge_dict, ignore_index=True)

                        # BEC knowledge level 
                        BEC_team_knowledge_dict = str_to_dict(bec_k, var_type = 'float')
                        BEC_team_knowledge_dict['run_no'] = run_id
                        BEC_team_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        BEC_team_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        BEC_team_knowledge_dict['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        team_BEC_knowledge_level = team_BEC_knowledge_level.append(BEC_team_knowledge_dict, ignore_index=True)
                    
                        # expected BEC knowledge level
                        BEC_team_knowledge_dict_expected = str_to_dict(bec_k_e, var_type = 'float')
                        BEC_team_knowledge_dict_expected['run_no'] = run_id
                        BEC_team_knowledge_dict_expected['loop_count'] = int(sim_vars['loop_count'][i])
                        BEC_team_knowledge_dict_expected['demo_strategy'] = sim_vars['demo_strategy'][i]
                        BEC_team_knowledge_dict_expected['knowledge_comp_id'] = int(sim_vars['knowledge_comp_id'][i])
                        team_BEC_knowledge_level_expected = team_BEC_knowledge_level_expected.append(BEC_team_knowledge_dict_expected, ignore_index=True)


                        # knowledge mix condition
                        learning_rate_dict = str_to_dict(ilv, var_type = 'array', splitter='),')
                        learning_rate_dict['run_no'] = run_id
                        learning_rate_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        learning_rate_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        learning_rate = learning_rate.append(learning_rate_dict, ignore_index=True)
                        # print('learning_rate_dict: ', learning_rate_dict)

                        # team knowledge constraints
                        team_knowledge_dict = str_to_dict(tkc, splitter = ', \'')
                        team_knowledge_dict['run_no'] = run_id
                        team_knowledge_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        team_knowledge_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        team_knowledge = team_knowledge.append(team_knowledge_dict, ignore_index=True)

        

                        # likelihood response
                        team_likelihood_correct_response_dict = {}
                        # print('lcr before: ', lcr)
                        lcr = lcr.strip('[]')
                        lcr = lcr.split(' ')
                        lcr = [i for i in lcr if i != '']
                        # print('lcr: ', lcr)
                        lcr_array = np.array(list(lcr), dtype=float)
                        # print('lcr_array: ', lcr_array)
                        team_likelihood_correct_response_dict['p1'] = lcr_array[0]
                        team_likelihood_correct_response_dict['p2'] = lcr_array[1]
                        team_likelihood_correct_response_dict['p3'] = lcr_array[2]
                        team_likelihood_correct_response_dict['common_knowledge'] = []
                        team_likelihood_correct_response_dict['joint_knowledge'] = []
                        team_likelihood_correct_response_dict['run_no'] = run_id
                        team_likelihood_correct_response_dict['demo_strategy'] = sim_vars['demo_strategy'][i]
                        team_likelihood_correct_response_dict['loop_count'] = int(sim_vars['loop_count'][i])
                        likelihood_correct_response = likelihood_correct_response.append(team_likelihood_correct_response_dict, ignore_index=True)


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

    learning_rate_index = []


    for i in range(len(learning_rate)):
        ilcr = learning_rate['ilcr'][i]
        lr_index = []
        for j in ilcr:
            if j >= 0.7:
                lr_index.extend([2])
            elif j >= 0.6:
                lr_index.extend([1])
            elif j >= 0.5:
                lr_index.extend([0])
        learning_rate_index.append(str(lr_index))

    # print('learning_rate_index:', learning_rate_index)

    # print('team_BEC_knowledge_level: ', team_BEC_knowledge_level)
    # print('likelihood_correct_response: ', likelihood_correct_response)

    # Long format data
    team_knowledge_level_long = pd.DataFrame()
    for know_id in ['p1', 'p2', 'p3', 'common_knowledge', 'joint_knowledge']:
        team_knowledge_level_long = pd.concat([team_knowledge_level_long, team_knowledge_level_min])
        BEC_knowledge_level.extend(team_BEC_knowledge_level[know_id])
        BEC_knowledge_level_expected.extend(team_BEC_knowledge_level_expected[know_id])
        ind_knowledge_type.extend([know_id]*len(team_BEC_knowledge_level[know_id]))
        lcr_var.extend(likelihood_correct_response[know_id])
        team_mix_var.extend(learning_rate_index)
        
        if 'p' in know_id:
            knowledge_type.extend(['individual']*len(team_BEC_knowledge_level[know_id]))
        else:
            knowledge_type.extend([know_id]*len(team_BEC_knowledge_level[know_id]))

    normalized_loop_count = team_knowledge_level_long['loop_count'].copy(deep=True)
    team_knowledge_level_long['BEC_knowledge_level'] = BEC_knowledge_level
    team_knowledge_level_long['BEC_knowledge_level_expected'] = BEC_knowledge_level_expected
    team_knowledge_level_long['knowledge_type'] = knowledge_type
    team_knowledge_level_long['ind_knowledge_type'] = ind_knowledge_type
    team_knowledge_level_long['likelihood_correct_response'] = lcr_var
    team_knowledge_level_long['normalized_loop_count'] = normalized_loop_count
    team_knowledge_level_long['team_mix'] = team_mix_var

    
    team_knowledge_level_long.to_csv('models/augmented_taxi2/team_knowledge_level_long.csv')
    team_knowledge_level_long.describe(include='all').to_csv('models/augmented_taxi2/descriptives.csv')

    
    ## concept-wise interaction count
    concept_ids = team_BEC_knowledge_level['knowledge_comp_id'].unique()
    unique_ids = team_knowledge_level_long['run_no'].unique()
    interaction_count = pd.DataFrame()

    for id in unique_ids:
        for c_id in range(len(concept_ids)):
            interaction_count_dict = {}
            interaction_count_dict['run_no'] = id
            interaction_count_dict['demo_strategy'] = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['demo_strategy'].iloc[0]
            interaction_count_dict['knowledge_comp_id'] = concept_ids[c_id]

            if c_id == len(concept_ids) - 1:
                max_loop_count = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['loop_count'].iloc[-1]
            else:
                max_loop_count = team_BEC_knowledge_level[(team_BEC_knowledge_level['run_no'] == id) & (team_BEC_knowledge_level['knowledge_comp_id'] == concept_ids[c_id+1])]['loop_count'].iloc[-1] - \
                                 team_BEC_knowledge_level[(team_BEC_knowledge_level['run_no'] == id) & (team_BEC_knowledge_level['knowledge_comp_id'] == concept_ids[c_id])]['loop_count'].iloc[-1]

            interaction_count_dict['max_loop_count'] = max_loop_count

            interaction_count = interaction_count.append(interaction_count_dict, ignore_index=True)

    # print('interaction_count: ', interaction_count)
    # interaction_count.to_csv('models/augmented_taxi2/interaction_count.csv')

    
    ## run-wise data
    run_data = pd.DataFrame()
    for id in unique_ids:
        run_data_dict = {}
        run_data_dict['run_no'] = id
        run_data_dict['demo_strategy'] = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['demo_strategy'].iloc[0]
        run_data_dict['max_loop_count'] = team_BEC_knowledge_level[team_BEC_knowledge_level['run_no'] == id]['loop_count'].iloc[-1]
        run_data_dict['team_mix'] = team_knowledge_level_long[team_knowledge_level_long['run_no'] == id]['team_mix'].iloc[0]
        
        run_data = run_data.append(run_data_dict, ignore_index=True)

    run_data.to_csv('models/augmented_taxi2/run_data.csv')



    # # filter negative knowledge levels
    # team_knowledge_level_long = team_knowledge_level_long[team_knowledge_level_long['BEC_knowledge_level'] >= 0]
    # team_knowledge_level_long = team_knowledge_level_long[team_knowledge_level_long['BEC_knowledge_level'] <= 1]


    ## rescale knowledge levels
    # unique_ids = team_knowledge_level_long['run_no'].unique()

    # print('unique_id: ', unique_ids)

    # for id in unique_ids:
    #     idx = team_knowledge_level_long[(team_knowledge_level_long['run_no'] == id) & (team_knowledge_level_long['knowledge_type']=='joint_knowledge')].index
    #     # print(team_knowledge_level_long.loc[idx, 'BEC_knowledge_level'])
    #     max_knowledge = np.max(team_knowledge_level_long.loc[idx, 'BEC_knowledge_level'])
    #     print('id: ', id)
        # print(max_knowledge)
        # if max_knowledge < 0.28 and max_knowledge > 0.2:
        #     factor = 1/0.28
        #     team_knowledge_level_long.loc[idx, 'BEC_knowledge_level'] = team_knowledge_level_long.loc[idx, 'BEC_knowledge_level'] * factor
        #     print('id: ', id)



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
    
    
    f, ax = plt.subplots(ncols=3)
    ax_id = 0
    for know_id in ['individual', 'common_knowledge', 'joint_knowledge']:
        sns.lineplot(data = team_knowledge_level_long[team_knowledge_level_long['knowledge_type']==know_id], x = 'loop_count', y = 'BEC_knowledge_level', hue = 'demo_strategy', ax=ax[ax_id], errorbar=('se', 1), err_style="band").set(title='Knowledge level for ' + know_id)
        ax_id += 1 
        # plt.savefig('models/augmented_taxi2/BEC_knowledge_level_' + know_id + '.png')
        # plt.close()

    
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


    ## plot team BEC knowledge level
    for id in unique_ids:
        idx = team_BEC_knowledge_level[(team_BEC_knowledge_level['run_no'] == id)].index

        # print(team_BEC_knowledge_level.columns)

        fig, axs = plt.subplots(2, 3, figsize=(9, 5), layout='constrained',
                        sharex=True, sharey=True)
        fig.suptitle('Run: ' + str(id))
        for nn, ax in enumerate(axs.flat):
            column_name = team_BEC_knowledge_level.columns[nn]
            y = team_BEC_knowledge_level.loc[idx,column_name]
            line, = ax.plot(team_BEC_knowledge_level.loc[idx, 'loop_count'], y, lw=2.5)
            ax.set_title(column_name, fontsize='small', loc='center')


            # plot verticle lines for visualizng end of concepts
            for kc_id in team_BEC_knowledge_level['knowledge_comp_id'].unique():
                max_idx = team_BEC_knowledge_level[(team_BEC_knowledge_level['knowledge_comp_id'] == kc_id) & (team_BEC_knowledge_level['run_no'] == id)].index.max()
                # print('kc_id:', kc_id, ' max_idx: ', max_idx)
                if not math.isnan(max_idx):
                    ax.axvline(team_BEC_knowledge_level.loc[max_idx, 'loop_count'], color='k', linestyle='--', linewidth=1)

            if nn == 4:
                break

        fig.supxlabel('Interaction Number')
        fig.supylabel('BEC Knowledge Level')


        ## Plot Unit knowledge level
        # fig2, axs2 = plt.subplots(2, 3, figsize=(9, 5), layout='constrained',
        #                 sharex=True, sharey=True)
        # for nn, ax in enumerate(axs2.flat):
        #     column_name = team_unit_knowledge_level.columns[nn]
        #     y = team_unit_knowledge_level.loc[idx,column_name]
        #     line, = ax.plot(team_unit_knowledge_level.loc[idx, 'loop_count'], y, lw=2.5)
        #     ax.set_title(column_name, fontsize='small', loc='center')

        #     # plot verticle lines for visulizng end of concepts
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
    sns.barplot(data = run_data, x = 'demo_strategy', y = 'max_loop_count', hue = 'team_mix', ax=ax_c, errorbar=('se',1)).set(title='Max number of interactions')




    plt.show()



if __name__ == "__main__":

    # process team knowledge data
    path = 'models/augmented_taxi2'
    files = os.listdir(path)
    # files = ['debug_obj_func_1.csv']
    
    # file_prefix_list = ['trials_set_1', 'trial_set_5', 'trial_set_6', 'trial_set_7']
    file_prefix_list = ['trials_set_1']
    
    # runs_to_exclude_list = [[3, 12, 24]]
    # runs_to_exclude_list = [[3, 12, 24, 7], [1,4,6,8], [], [1,3]]
    # runs_to_exclude_list = [[1,3]]
    runs_to_exclude_list = []

    
    runs_to_analyze_list = [[3, 12, 24]]
    # runs_to_analyze_list = []

    run_analysis_script(path, files, file_prefix_list, runs_to_exclude_list = runs_to_exclude_list, runs_to_analyze_list = runs_to_analyze_list)
    

   



    