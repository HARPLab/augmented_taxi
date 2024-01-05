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
matplotlib.use('TkAgg')
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



if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

    
    # fixed parameters
    sampling_condition_list = ['particles']  # Conditions: ['particles', 'cluster_random', 'cluster_weight']sampling of human responses from learner PF models
    sim_params = {'min_correct_likelihood': 0}


    team_params_learning = {'low': [0.65, 0.05], 'med': [0.65, 0.05], 'high': [0.8, 0.05]}

    # team_params_learning = {'low': 0.5, 'med': 0.65, 'high': 0.8}
    team_composition_list = [[0,0,0], [0,0,2]]
    # dem_strategy_list = ['joint_knowledge']

    # team_composition_list = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    dem_strategy_list = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']
    

    
    
    N_runs = 9
    run_start_id = 1
    learner_update_type = 'noise'

    file_prefix = 'trials_01_04_w_updated_noise'
    
    path = 'data/simulation/sim_experiments/sensitivity_analysis/'


    sim_conditions = get_sim_conditions(team_composition_list, dem_strategy_list, sampling_condition_list, N_runs, run_start_id)




    # for run_id in range(run_start_id, run_start_id+N_runs):
    #     print('sim_conditions run_id:', sim_conditions[run_id - run_start_id], '. sim_conditions: ', sim_conditions )
    #     if run_id == sim_conditions[run_id - run_start_id][0]:
    #         team_composition_for_run = sim_conditions[run_id - run_start_id][1]
    #         dem_strategy_for_run = sim_conditions[run_id - run_start_id][2]
    #         sampling_cond_for_run = sim_conditions[run_id - run_start_id][3]
    #     else:
    #         RuntimeError('Error in sim conditions')
    #         break

    #     ilcr = np.zeros(params.team_size)
    #     rlcr = np.zeros([params.team_size, 2])

    #     for j in range(params.team_size):
    #         if team_composition_for_run[j] == 0: 
    #             ilcr[j] = team_params_learning['low'][0]
    #             rlcr[j,0] = team_params_learning['low'][1]
    #             rlcr[j,1] = 0     
    #         elif team_composition_for_run[j] == 1:
    #             ilcr[j] = team_params_learning['med'][0]
    #             rlcr[j,0] = team_params_learning['med'][1]
    #             rlcr[j,1] = 0
    #         elif team_composition_for_run[j] == 2:
    #             ilcr[j] = team_params_learning['high'][0]
    #             rlcr[j,0] = team_params_learning['high'][1]
    #             rlcr[j,1] = 0
        
    #     print(colored('Simulation run: ' + str(run_id) + '. Demo strategy: ' + str(dem_strategy_for_run) + '. Team composition:' + str(team_composition_for_run), 'red'))
    #     # run_reward_teaching(params, pool, sim_params, demo_strategy = dem_strategy_for_run, experiment_type = 'simulated', team_learning_factor = team_learning_factor, viz_flag=[False, False, False], run_no = run_id, vars_filename=file_prefix)
    #     run_reward_teaching(params, pool, sim_params, demo_strategy = dem_strategy_for_run, experiment_type = 'simulated', initial_team_learning_factor = ilcr, team_learning_rate = rlcr, \
    #                         viz_flag=[False, False, False], run_no = run_id, vars_filename=file_prefix, response_sampling_condition=sampling_cond_for_run, team_composition=team_composition_for_run, learner_update_type = learner_update_type)


    #     # file_name = [file_prefix + '_' + str(run_id) + '.csv']
    #     # print('Running analysis script... Reading data from: ', file_name)
    #     # run_analysis_script(path, file_name, file_prefix)

    
    # # # save all the variables
    # # vars_to_save.to_csv('models/augmented_taxi2/vars_to_save.csv', index=False)
    # # with open('models/augmented_taxi2/vars_to_save.pickle', 'wb') as f:
    # #     pickle.dump(vars_to_save, f)
    
    # print('All simulation runs completed..')
    
    # # pool.close()
    # # pool.join()


    