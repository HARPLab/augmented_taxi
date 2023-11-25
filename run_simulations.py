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



if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

    

    # team_params_learning = {'low': [0.4, 0.1], 'med': [0.5, 0.1], 'high': [0.6, 0.1]}
    team_params_learning = {'low': [0.5, 0.05], 'med': [0.6, 0.05], 'high': [0.7, 0.05]}
    # print([*range(params.team_size)])
    # team_composition_list = list(permutations([*range(2)]))
    # team_composition_list = list(combinations([*range(2)], params.team_size))

    # team_composition_list = [[0,2,2]]
    team_composition_list = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    dem_strategy_list = ['individual_knowledge_low', 'individual_knowledge_high', 'common_knowledge', 'joint_knowledge']
    # dem_strategy_list = ['individual_knowledge_high']
    team_comp_id = 0
    dem_strategy_id = 0

    sim_params = {'min_correct_likelihood': 0.5}
    N_runs = 80
    run_start_id = 1
    file_prefix = 'trial_set_13'
    path = 'models/augmented_taxi2'

    for run_id in range(run_start_id, run_start_id+N_runs):
        # team_composition_for_run = list(team_composition_list[np.random.choice(range(len(team_composition_list)))])

        # team_composition_for_run = []
        # for k in range(params.team_size):
        #     team_composition_for_run.append(random.choice([0,1,2]))

        team_composition_for_run = team_composition_list[team_comp_id]

        # if N_runs < 5:
        #     team_composition_for_run = team_composition_list[0]
        # else:
        #     team_composition_for_run = team_composition_list[1]


        print('team_composition_for_run: ', team_composition_for_run)

        ilcr = np.zeros(params.team_size)
        rlcr = np.zeros([params.team_size, 2])

        for j in range(params.team_size):
            if team_composition_for_run[j] == 0: 
                ilcr[j] = team_params_learning['low'][0]
                rlcr[j,0] = team_params_learning['low'][1]
                rlcr[j,1] = -0.1     
            elif team_composition_for_run[j] == 1:
                ilcr[j] = team_params_learning['med'][0]
                rlcr[j,0] = team_params_learning['med'][1]
                rlcr[j,1] = -0.1
            elif team_composition_for_run[j] == 2:
                ilcr[j] = team_params_learning['high'][0]
                rlcr[j,0] = team_params_learning['high'][1]
                rlcr[j,1] = -0.1


        dem_strategy = dem_strategy_list[dem_strategy_id]

        print(colored('Simulation run: ' + str(run_id) + '. Demo strategy: ' + str(dem_strategy) + 'initial lcr: ' + str(ilcr) + 'rate lcr: ' + str(rlcr), 'red'))
        # viz_flag = [demo_viz, test_viz, pf_knowledge_viz]
        run_reward_teaching(params, pool, sim_params, demo_strategy = dem_strategy, experiment_type = 'simulated', team_likelihood_correct_response = ilcr,  team_learning_rate = rlcr, viz_flag=[False, False, False], run_no = run_id, vars_filename=file_prefix)
        
        # file_name = [file_prefix + '_' + str(run_id) + '.csv']
        # print('Running analysis script... Reading data from: ', file_name)
        # run_analysis_script(path, file_name, file_prefix)

        # update sim params
        if dem_strategy_id == len(dem_strategy_list)-1:
            dem_strategy_id = 0
        else:
            dem_strategy_id += 1

        if team_comp_id == 0:
            team_comp_id = 1
        else:
            team_comp_id = 0

    
    # # save all the variables
    # vars_to_save.to_csv('models/augmented_taxi2/vars_to_save.csv', index=False)
    # with open('models/augmented_taxi2/vars_to_save.pickle', 'wb') as f:
    #     pickle.dump(vars_to_save, f)
    
    print('All simulation runs completed..')
    
    # pool.close()
    # pool.join()


    