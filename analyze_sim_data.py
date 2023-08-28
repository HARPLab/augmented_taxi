# Analyze simulation data

import pandas as pd
import ast
import json
import teams.teams_helpers as teams_helpers
import params_team as params
import numpy as np


def str_to_dict(string, var_type = None, splitter = ', '):
    # remove the curly braces from the string
    string = string.strip('{}')
 
    # split the string into key-value pairs
    pairs = string.split(splitter)

    # if splitter != ', ':
    #     pairs = [pair + '] ' for pair in pairs]
        # pairs[-1] = pairs[-1][:-2]
 
    # use a dictionary comprehension to create
    # the dictionary, converting the values to
    # integers and removing the quotes from the keys
    dict = {}
    print(pairs)
    for i in range(len(pairs)):
        pair = pairs[i]
        key, value = pair.split(': ')
        key = key.strip(' \' ')

        if splitter == ', ':
            value = value.strip(' \' ')  # strip is used only for leading and trailing characters
            value = value.strip(' array ')
            value = value.strip(' ([]) ')
        else:
            value = value.replace('array', '')
            value = value.replace('(', '')
            value = value.replace(')', '')
            # print('Knowledge_type: ', key)
            # print('Type before: ', type(value))
            # print('Check before: ', value)
            # value = ast.literal_eval(value)
            value = eval(value)
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
        else:
            print('data type: ', type(value))   
            dict[key] = value

    # return {key[1:-2]: float(value.strip(' \' ')) for key, value in (pair.split(': ') for pair in pairs)}
    
    return dict
 



if __name__ == "__main__":
    # params = Params('params_taxi.json')
    sim_vars = pd.read_csv('models/augmented_taxi2/sim_run_ck.csv')
    # print(len(sim_vars))

    team_unit_knowledge_level = pd.DataFrame()
    team_BEC_knowledge_level = pd.DataFrame()
    team_knowledge = pd.DataFrame()

    # process team knowledge data
    for i in range(len(sim_vars)):

        # unit knowledge level
        tk = sim_vars['unit_knowledge_level'][i]
        unit_knowledge_dict = str_to_dict(tk, var_type = 'float')
        unit_knowledge_dict['run_no'] = int(sim_vars['run_no'][i])
        team_unit_knowledge_level = team_unit_knowledge_level.append(unit_knowledge_dict, ignore_index=True)

        # BEC knowledge level
        bec_k = sim_vars['BEC_knowledge_level'][i]
        BEC_team_knowledge_dict = str_to_dict(bec_k, var_type = 'float')
        BEC_team_knowledge_dict['run_no'] = int(sim_vars['run_no'][i])
        team_BEC_knowledge_level = team_BEC_knowledge_level.append(BEC_team_knowledge_dict, ignore_index=True)

        # team knowledge constraints
        tkc = sim_vars['team_knowledge'][i]
        # tkc = tkc.replace('\'', '"')
        team_knowledge_dict = str_to_dict(tkc, splitter = ', \'')
        team_knowledge_dict['run_no'] = int(sim_vars['run_no'][i])

        print('team_knowledge_dict: ', team_knowledge_dict)
        team_knowledge = team_knowledge.append(team_knowledge_dict, ignore_index=True)

        # BEC constraints
        bec = sim_vars['min_BEC_constraints'][i]
        bec = bec.replace('array', '')
        bec = bec.replace('(', '')
        bec = bec.replace(')', '')
        bec = ast.literal_eval(bec)
        bec_copy = []
        for val in bec:
            bec_copy.extend(np.array([val]))
        bec = bec_copy

        # plot knowledge constraints
        teams_helpers.visualize_team_knowledge_constraints(bec, team_knowledge_dict, unit_knowledge_dict, BEC_team_knowledge_dict, params.mdp_class, weights=params.weights['val'])




    print(team_unit_knowledge_level)
    print(team_BEC_knowledge_level)
    print(team_knowledge)


    