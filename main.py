#!/usr/bin/env python
'''
-----------HEADER-----------
This variation of augmented taxi 2 is used to create an augmented taxi 2 
with multiple passengers that a player can click on in order to choose which 
one they would like to path plan towards.

This code mainly uses the simple_rl framework
most of the folders including BEC are not being used in this code and are only kept
in case future development may require their use.
'''
# Python imports.
import sys
#import dill as pickle
import numpy as np
import copy
from termcolor import colored
#import sage.all
#import sage.geometry.polyhedron.base as Polyhedron
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
from collections import defaultdict
from multiprocessing import Process, Queue, Pool

# Other imports.
sys.path.append("simple_rl")
import params
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import make_mdp
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

# either "continous" or "reset" 
game_type = 'continous'
#currently only thing being used basically
#=[{'x': 4, 'y': 1, 'dest_x': 3, 'dest_y': 2, 'in_taxi': 0}]
def generate_agent(mdp_class, data_loc, mdp_parameters, passengers=[{'x': 4, 'y': 1, 'dest_x': 3, 'dest_y': 2, 'in_taxi': 0}], visualize=False):
    '''try:
        with open('models/' + data_loc + '/vi_agent.pickle', 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        cf_data_dir = 'models/' + data_loc
        os.makedirs(cf_data_dir, exist_ok=True)
    '''

    
    while len(passengers) != 0:
        dest_x = passengers[0]['dest_x']
        dest_y = passengers[0]['dest_y']
        agent_start_x = mdp_parameters['agent']['x']
        agent_start_y = mdp_parameters['agent']['y']
        mdp_vi_dict = {}
        combined_mdp_passengers = []
        #print(mdp_parameters['passengers'])
        for i in passengers:
            combined_mdp_passengers.append(i)
            mdp_parameters['passengers'] = [i]
            #print(mdp_parameters['passengers'])
            temp_mdp_agent = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)
            mdp_vi_dict[(i['x'],i['y'])]=(temp_mdp_agent)
            #print("mdp_vi_dict:")
            #print((i['x'],i['y']))

        #print(mdp_list)
        #print(vi_agent_list)
        '''
            with open('models/' + data_loc + '/vi_agent.pickle', 'wb') as f:
                pickle.dump((mdp_agent, vi_agent), f)
        '''

    

        # Visualize agent
        if visualize:
            mdp_parameters['passengers'] = combined_mdp_passengers
            combined_mdp_agent = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)
            #print(combined_mdp_agent.get_passengers())
            cell_coords = combined_mdp_agent.visualize_state(combined_mdp_agent.cur_state)
            combined_mdp_agent.reset()
            for i in passengers:
                if i['x'] == cell_coords[0] and i['y'] == cell_coords[1]:
                    passengers.remove(i)
            mdp_parameters
            print(cell_coords)
            if not cell_coords == None:
                #print(cell_coords)
                mdp_agent = mdp_vi_dict[cell_coords]
                
                vi_agent = ValueIteration(mdp_agent, sample_rate=1)
                vi_agent.run_vi()
                #print(mdp_agent.get_passengers())
                fixed_agent = FixedPolicyAgent(vi_agent.policy)
                mdp_agent.visualize_agent(fixed_agent)
                mdp_agent.reset()  # reset the current state to the initial state
            if game_type == "continous":
                for i in range(len(passengers)):
                    passengers[i]["dest_x"] = agent_start_x
                    passengers[i]["dest_y"] = agent_start_y
                mdp_parameters["agent"]['x'] = dest_x
                mdp_parameters['agent']["y"] = dest_y
                temp_x = agent_start_x
                temp_y = agent_start_y
                agent_start_x = dest_x
                agent_start_y = dest_y
                dest_x = temp_x
                dest_y = temp_y
            
            

            #agent visualization:
            '''
            combined_vi_agent = ValueIteration(combined_mdp_agent, sample_rate=1)
            combined_vi_agent.run_vi()
            combined_fixed_agent = FixedPolicyAgent(combined_vi_agent.policy)
            combined_mdp_agent.visualize_agent(combined_fixed_agent, augmented_inputs=True)
            combined_mdp_agent.reset()  # reset the current state to the initial state
            '''
            
            '''
            for mdp,vi_agent in mdp_list,vi_agent_list:
                fixed_agent = FixedPolicyAgent(vi_agent.policy)
                mdp_agent.visualize_agent(fixed_agent, augmented_inputs=True)
                mdp_agent.reset()  # reset the current state to the initial state
                from simple_rl.tasks.taxi.taxi_visualizer import _draw_augmented_state
                mdp_agent.visualize_interaction(keys_map=params.keys_map)
            '''


if __name__ == "__main__":
    # a) generate an agent if you want to explore the Augmented Taxi MDP
    
    try:
        file = open('customizations.txt', mode = 'r')
        lines = file.readlines()
        file.close()
        #finding the first empty line and setting new_list to only contain the lines before that
        new_list = []
        for line in lines:
            line = line.split(',')
            if line == ['\n']:
                break
            line = [i.strip() for i in line]
            new_list.append(line)
        width = len(new_list[0][0])
        height = len(new_list)
        row = 0
        col = 0
        walls = []
        passenger_loc = []
        tolls = []
        agent = None
        battery = []
        for line in new_list:
            row += 1
            #print(line)
            for character in line[0]:
                #print(character)
                col += 1
                if character == 'a':
                    agent = dict(x=col,y=height-row+1,has_passenger=0)
                elif character == 'p':
                    passenger_loc.append((col,height-row+1))
                elif character == 'w':
                    walls.append(dict(x=col,y=height-row+1))
                elif character == 'd':
                    destination = (col,height-row+1) #will need to use this to help create passengers
                elif character == 'b':
                    battery.append(dict(x=col,y=height-row+1)) # assuming only one battery station currently
                elif character == 't':
                    tolls.append(dict(x=col,y=height-row+1))   
            col = 0
        params.mdp_parameters["agent"] = agent
        params.mdp_parameters["walls"] = walls
        params.mdp_parameters["tolls"] = tolls
        params.mdp_parameters["hotswap_station"] = battery
        params.mdp_parameters["height"] = height
        params.mdp_parameters["width"] = width
        print("battery:")
        print(params.mdp_parameters["hotswap_station"])
        print("agent:")
        print(params.mdp_parameters["agent"])
        print("walls:")
        print(params.mdp_parameters["walls"])
        print("tolls:")
        print(params.mdp_parameters["tolls"])
        print("height:")
        print(params.mdp_parameters["height"])
        print("width:")
        print(params.mdp_parameters["width"])
        passengers = []
        for passenger in passenger_loc:
            (x,y) = passenger
            (dest_x,dest_y) = destination
            passengers.append(dict(x=x,y=y,dest_x=dest_x,dest_y=dest_y,in_taxi=0))
        print(passengers)

        



    except:
        print("failed to parse loading default from params.py and custom passengers")
        passengers = [{'x': 6, 'y': 5, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0},{'x': 5, 'y': 2, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}, {'x': 6, 'y': 7, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}]
    finally:   
        generate_agent(params.mdp_class, params.data_loc['base'], params.mdp_parameters, passengers=passengers, visualize=True)

    # b) obtain a BEC summary of the agent's policy
    '''
    BEC_summary, visited_env_traj_idxs, particles_summary = obtain_summary(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'],
                            params.step_cost_flag, params.BEC['summary_variant'], pool, params.BEC['n_train_demos'], params.BEC['BEC_depth'],
                            params.BEC['n_human_models'], params.BEC['n_particles'], params.prior, params.posterior, params.BEC['obj_func_proportion'])
    '''
    # c) run through the closed-loop teaching framework
    #simulate_teaching_loop(params.mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, params.prior, params.BEC['n_particles'], params.BEC['n_human_models'], params.BEC['n_human_models_precomputed'], params.data_loc['BEC'], params.weights['val'], params.step_cost_flag)

    #n_human_models_real_time = 8 #never used

    # d) run remedial demonstration and test selection on previous participant responses from IROS
    # analyze_prev_study_tests(params.mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, params.prior, params.BEC['n_particles'], n_human_models_real_time, params.BEC['n_human_models_precomputed'], params.data_loc['BEC'], params.weights['val'], params.step_cost_flag, visualize_pf_transition=True)

    # e) compare the remedial demonstration selection when using 2-step dev/BEC vs. PF (assuming 3 static humans models for low, medium, and high difficulties)
    # contrast_PF_2_step_dev(params.mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, params.prior, params.BEC['n_particles'], n_human_models_real_time, params.data_loc['BEC'], params.weights['val'], params.step_cost_flag, visualize_pf_transition=False)

    # c) obtain test environments
    # obtain_test_environments(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
    #                          params.step_cost_flag, params.BEC['n_human_models'], params.prior, params.posterior, summary=BEC_summary, visualize_test_env=True, use_counterfactual=True)


    #pool.close()
    #pool.join()