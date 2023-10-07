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
from itertools import permutations

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



if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

    N_runs = 100

    # weights_list = [[0.6, 0.15, 0.25], [0.5, 0.25, 0.25], [0.4, 0.3, 0.3], [0.8, 0.05, 0.1], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.5, 0.1, 0.4], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    
    # weights_list = [[0.2, 0.0, 0.8]]

    # try:
    #     with open('models/augmented_taxi2/vars_to_save.pickle', 'rb') as f:
    #         vars_to_save = pickle.load(f)
    # except:
    #     vars_to_save = None
    # print('Vars len so far: ', vars_to_save.shape[0])
        
    # # simulation old
    # resp_dist_list_history = []
    # for i in range(1, N_runs+1):
    #     # j = int(np.floor(i/len(weights_list)))
    #     j = 0
    #     resp_dist_list = []
    #     while len(resp_dist_list) == 0 or (resp_dist_list in resp_dist_list_history and (1 not in weights_list[j])):
    #         resp_dist_list = random.choices(['correct', 'incorrect', 'mixed'], weights = weights_list[j], k=40)
                
    #     resp_dist_list_history.append(resp_dist_list)
    #     dem_strategy = random.choice(['common_knowledge', 'joint_knowledge', 'individual_knowledge_low', 'individual_knowledge_high'])
    #     # dem_strategy = 'common_knowledge'
    #     # print('resp_dist_list_history: ', resp_dist_list_history)
    #     print('Simulation run: ', i, '. Demo strategy: ', dem_strategy)
    #     print('resp_dist_list: ', resp_dist_list)
    #     # vars_to_save = run_reward_teaching(params, pool, demo_strategy = dem_strategy, response_type = 'simulated', response_distribution_list= resp_dist_list, run_no = i, vars_to_save = vars_to_save)
    #     run_reward_teaching(params, pool, demo_strategy = dem_strategy, response_type = 'simulated', viz_flag=[False, False, False], response_distribution_list= resp_dist_list, run_no = i, vars_filename='sim_run_mixed_maj')

    sim_params = {'min_correct_likelihood': 0.5}
    start_id = 20


    team_params_learning = {'low': [0.5, 0.05], 'med': [0.6, 0.07], 'high': [0.7, 0.1]}
    print([*range(params.team_size)])
    # team_composition_list = list(permutations([*range(params.team_size)]))
    team_composition_list = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
    print(team_composition_list)
    for i in range(50, N_runs+1):
        team_composition_for_run = team_composition_list[np.random.choice(range(len(team_composition_list)))]
        print('team_composition_for_run: ', team_composition_for_run)

        ilcr = np.zeros(params.team_size)
        rlcr = np.zeros([params.team_size, 2])

        for j in range(params.team_size):
            if team_composition_for_run[j] == 0:
                ilcr[j] = random.random()*0.1 + team_params_learning['low'][0]  # between 0.5 and 0.6
                x = random.random()*0.05 + team_params_learning['low'][1]  # between 0.05 and 0.15
            elif team_composition_for_run[j] == 1:
                ilcr[j] = random.random()*0.1 + team_params_learning['med'][0]  # between 0.5 and 0.6
                x = random.random()*0.05 + team_params_learning['med'][1]  # between 0.05 and 0.15
            elif team_composition_for_run[j] == 2:
                ilcr[j] = random.random()*0.1 + team_params_learning['high'][0]  # between 0.5 and 0.6
                x = random.random()*0.05 + team_params_learning['high'][1]  # between 0.05 and 0.15
                
            rlcr[j,0] = x
            rlcr[j,1] = -x      
            
        
        # dem_strategy = random.choice(['common_knowledge', 'joint_knowledge', 'individual_knowledge_low', 'individual_knowledge_high'])
        dem_strategy_list = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']
        
        if (i-start_id)/(N_runs-start_id) < 0.25:
            dem_strategy = dem_strategy_list[0]
        elif (i-start_id)/(N_runs-start_id) < 0.5:
            dem_strategy = dem_strategy_list[1]
        elif (i-start_id)/(N_runs-start_id) < 0.75:
            dem_strategy = dem_strategy_list[2]
        else:
            dem_strategy = dem_strategy_list[3]

        print(colored('Simulation run: ' + str(i) + '. Demo strategy: ' + str(dem_strategy) + 'initial lcr: ' + str(ilcr) + 'rate lcr: ' + str(rlcr), 'red'))
        run_reward_teaching(params, pool, sim_params, demo_strategy = dem_strategy, experiment_type = 'simulated', team_likelihood_correct_response = ilcr,  team_learning_rate = rlcr, viz_flag=[False, False, False], run_no = i, vars_filename='new_hlm_sim_run_2')


    # # save all the variables
    # vars_to_save.to_csv('models/augmented_taxi2/vars_to_save.csv', index=False)
    # with open('models/augmented_taxi2/vars_to_save.pickle', 'wb') as f:
    #     pickle.dump(vars_to_save, f)
    
    print('All simulation runs completed..')
    
    # pool.close()
    # pool.join()


    