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

    N_runs = 20

    # weights_list = [[0.6, 0.15, 0.25], [0.5, 0.25, 0.25], [0.4, 0.3, 0.3], [0.8, 0.05, 0.1], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.5, 0.1, 0.4], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    
    weights_list = [[0.0, 0.0, 1.0]]

    # try:
    #     with open('models/augmented_taxi2/vars_to_save.pickle', 'rb') as f:
    #         vars_to_save = pickle.load(f)
    # except:
    #     vars_to_save = None
    # print('Vars len so far: ', vars_to_save.shape[0])
        

    resp_dist_list_history = []
    for i in range(1, N_runs+1):
        # j = int(np.floor(i/len(weights_list)))
        j = 0
        resp_dist_list = []
        while len(resp_dist_list) == 0 or (resp_dist_list in resp_dist_list_history and (1 not in weights_list[j])):
            resp_dist_list = random.choices(['correct', 'incorrect', 'mixed'], weights = weights_list[j], k=20)
                
        resp_dist_list_history.append(resp_dist_list)
        # dem_strategy = random.choice(['common_knowledge', 'joint_knowledge'])
        dem_strategy = 'common_knowledge'
        # print('resp_dist_list_history: ', resp_dist_list_history)
        print('Simulation run: ', i, '. Demo strategy: ', dem_strategy)
        print('resp_dist_list: ', resp_dist_list)
        # vars_to_save = run_reward_teaching(params, pool, demo_strategy = dem_strategy, response_type = 'simulated', response_distribution_list= resp_dist_list, run_no = i, vars_to_save = vars_to_save)
        run_reward_teaching(params, pool, demo_strategy = dem_strategy, response_type = 'simulated', response_distribution_list= resp_dist_list, run_no = i, vars_filename='sim_run_ck_maj')


    # # save all the variables
    # vars_to_save.to_csv('models/augmented_taxi2/vars_to_save.csv', index=False)
    # with open('models/augmented_taxi2/vars_to_save.pickle', 'wb') as f:
    #     pickle.dump(vars_to_save, f)
    
    print('All simulation runs completed..')
    
    # pool.close()
    # pool.join()


    