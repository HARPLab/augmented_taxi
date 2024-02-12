# run simulations for reward teaching
# Python imports.
import sys
import dill as pickle
import numpy as np
import copy
from termcolor import colored
from multiprocessing import Process, Queue, Pool
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
from itertools import permutations, combinations

# Other imports.
sys.path.append("simple_rl")
import params_team as params
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
from teams import particle_filter_team as pf_team
import matplotlib as mpl
import teams.teams_helpers as team_helpers
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

import random
import pandas as pd

from main_team import run_reward_teaching
from analyze_sim_data import run_analysis_script

from pyDOE import lhs


def get_sim_conditions(team_composition_list, dem_strategy_list, sampling_condition_list, N_runs, run_start_id):
    sim_conditions = []
    team_comp_id = 0
    dem_strategy_id = 0
    sampling_cond_id = 0
    
    for run_id in range(run_start_id, run_start_id+N_runs):
        
        team_composition_for_run = team_composition_list[team_comp_id]
        dem_strategy = dem_strategy_list[dem_strategy_id]
        sampling_cond = sampling_condition_list[sampling_cond_id]
        sim_conditions.append([run_id, team_composition_for_run, dem_strategy, sampling_cond])
        
        # update sim params for next run
        if dem_strategy_id == len(dem_strategy_list)-1:
            dem_strategy_id = 0
        else:
            dem_strategy_id += 1

        if run_id % len(dem_strategy_list) == (len(dem_strategy_list)-1):
            if team_comp_id == len(team_composition_list)-1:
                team_comp_id = 0
            else:
                team_comp_id += 1

        if run_id % len(dem_strategy_list) == (len(dem_strategy_list)-1):
            if sampling_cond_id == len(sampling_condition_list)-1:
                sampling_cond_id = 0
            else:
                sampling_cond_id += 1


    return sim_conditions


def get_parameter_combination(params_to_study, num_samples):
    
    N = len(params_to_study)
    lhs_sample = lhs(N, samples=num_samples, criterion = 'maximin')

    sample_combinations = []
    for i in range(len(lhs_sample)):
        sample_combinations.append([])
        for j, param in enumerate(params_to_study):
            sample_combinations[i].append(params_to_study[param][0] + lhs_sample[i][j]*(params_to_study[param][1] - params_to_study[param][0]))

    with open('data/simulation/sim_experiments/sensitivity_analysis/param_combinations.pickle', 'wb') as f:
        pickle.dump(sample_combinations, f)

    return sample_combinations





if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)



    ## varying parameters
    N_runs_for_each_study_condition = 5
    run_start_id = 1
    sensitivity_run_start_id = 1
    N_combinations = 30

    file_prefix = 'trials_02_11_sensitivity_w_feedback'
    
    path = 'data/simulation/sim_experiments/sensitivity_analysis/'

    ## Learner model params sensitivity analysis
    # params_to_study = {'learning_factor_low': [0.6, 0.7], 'learning_factor_high': [0.75, 0.85], 'learning_rate': [0.0, 0.1], 'max_learning_factor': [0.85, 1.0]}
    
    ## Learner and teacher model params sensitivity analysis
    params_to_study = {'learning_factor_low': [0.6, 0.7], 'learning_factor_high': [0.75, 0.85], 'learning_rate': [0.0, 0.1], 'max_learning_factor': [0.85, 1.0], 'default_learning_factor_teacher': [0.6, 0.9]}
    

    # Experiemnt conditions to test - keep this fixed for a set of sensitivity runs
    team_composition_list = [[0,0,0]]   # [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    dem_strategy_list = ['joint_knowledge'] # ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']

    ##########################

    # fixed parameters
    learner_update_type = 'no_noise'
    sampling_condition_list = ['particles']  # Conditions: ['particles', 'cluster_random', 'cluster_weight']sampling of human responses from learner PF models
    #################################

    ### generate or load parametr combinations
    try:
        with open('data/simulation/sim_experiments/sensitivity_analysis/param_combinations.pickle', 'rb') as f:
            parameter_combinations = pickle.load(f)
    
        if len(parameter_combinations) != (sensitivity_run_start_id + N_combinations-1):
            parameter_combinations = get_parameter_combination(params_to_study, N_combinations)
            print('Parameter combinations generated: ', parameter_combinations)
        else:
            print('Parameter combinations loaded: ', parameter_combinations)
    except:
        parameter_combinations = get_parameter_combination(params_to_study, N_combinations)
        print('Parameter combinations generated: ', parameter_combinations)
        RuntimeError('Parameter combinations unavailable')

    ######################
    

    for sensitivity_run_id in range(sensitivity_run_start_id, sensitivity_run_start_id + N_combinations):
        
        # Learner and teacher model params sensitivity analysis
        team_params_learning = {'low': [parameter_combinations[sensitivity_run_id-1][0], parameter_combinations[sensitivity_run_id-1][2]/2, parameter_combinations[sensitivity_run_id-1][2]], 
                                'high': [parameter_combinations[sensitivity_run_id-1][1], parameter_combinations[sensitivity_run_id-1][2]/2, parameter_combinations[sensitivity_run_id-1][2]]}
        params.max_learning_factor = parameter_combinations[sensitivity_run_id-1][3]
        params.default_learning_factor_teacher = parameter_combinations[sensitivity_run_id-1][4]
        print('Sensitivity run: ', sensitivity_run_id, '. Team params: ', team_params_learning, '. Max learning factor: ', params.max_learning_factor, '. Learning_factor_teacher:', params.default_learning_factor_teacher)

        
        sim_conditions = get_sim_conditions(team_composition_list, dem_strategy_list, sampling_condition_list, N_runs_for_each_study_condition, run_start_id)

        # for each study parameter combination, N_runs of simulations

        for run_id in range(run_start_id, run_start_id+N_runs_for_each_study_condition):
            print('sim_conditions run_id:', sim_conditions[run_id - run_start_id], '. sim_conditions: ', sim_conditions )
            if run_id == sim_conditions[run_id - run_start_id][0]:
                team_composition_for_run = sim_conditions[run_id - run_start_id][1]
                dem_strategy_for_run = sim_conditions[run_id - run_start_id][2]
                sampling_cond_for_run = sim_conditions[run_id - run_start_id][3]
            else:
                RuntimeError('Error in sim conditions')
                break

            ilcr = np.zeros(params.team_size)
            rlcr = np.zeros([params.team_size, 2])

            for j in range(params.team_size):
                if team_composition_for_run[j] == 0: 
                    ilcr[j] = team_params_learning['low'][0]
                    rlcr[j,0] = team_params_learning['low'][1]
                    rlcr[j,1] = team_params_learning['low'][2]     
                elif team_composition_for_run[j] == 1:
                    ilcr[j] = team_params_learning['med'][0]
                    rlcr[j,0] = team_params_learning['med'][1]
                    rlcr[j,1] = team_params_learning['med'][2]
                elif team_composition_for_run[j] == 2:
                    ilcr[j] = team_params_learning['high'][0]
                    rlcr[j,0] = team_params_learning['high'][1]
                    rlcr[j,1] = team_params_learning['high'][2]
            
            print(colored('Simulation run: ' + str(run_id) + '. Demo strategy: ' + str(dem_strategy_for_run) + '. Team composition:' + str(team_composition_for_run), 'red'))
            # run_reward_teaching(params, pool, sim_params, demo_strategy = dem_strategy_for_run, experiment_type = 'simulated', team_learning_factor = team_learning_factor, viz_flag=[False, False, False], run_no = run_id, vars_filename=file_prefix)
            run_reward_teaching(params, pool, [params.default_learning_factor_teacher]*params.team_size, demo_strategy = dem_strategy_for_run, experiment_type = 'simulated', initial_team_learning_factor = ilcr, team_learning_rate = rlcr, \
                                viz_flag=[False, False, False], run_no = run_id, vars_filename_prefix=file_prefix, response_sampling_condition=sampling_cond_for_run, team_composition=team_composition_for_run, learner_update_type = learner_update_type, study_id = sensitivity_run_id)


            # file_name = [file_prefix + '_' + str(run_id) + '.csv']
            # print('Running analysis script... Reading data from: ', file_name)
            # run_analysis_script(path, file_name, file_prefix)

        
        # # save all the variables
        # vars_to_save.to_csv('models/augmented_taxi2/vars_to_save.csv', index=False)
        # with open('models/augmented_taxi2/vars_to_save.pickle', 'wb') as f:
        #     pickle.dump(vars_to_save, f)
        
        print('All simulation runs completed for this study setting..')
    print('All simulation runs completed..')
        
    # pool.close()
    # pool.join()


    