import cProfile

import debugging_team as dt
import analyze_sim_data as asd
import params_team as params
import simulation.sim_helpers as sim_helpers
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')


def get_sim_conditions(team_composition_list, N_runs, run_start_id):
    sim_conditions = []
    team_comp_id = 0
    
    for run_id in range(run_start_id, run_start_id+N_runs):
        
        team_composition_for_run = team_composition_list[team_comp_id]
        sim_conditions.append([run_id, team_composition_for_run])
        
        # update sim params for next run
        if team_comp_id == len(team_composition_list)-1:
            team_comp_id = 0
        else:
            team_comp_id += 1


    return sim_conditions


if __name__ == "__main__":
    
    ## run response sampling code
    # path = 'data/simulation/sampling_tests'
    # N_runs = 1
    # run_start_id = 3
    # # condition_list = ['cluster_random', 'cluster_weight', 'particles']
    # condition_list = ['particles']

    # for run_id in range(run_start_id, run_start_id + N_runs):
    #     print('run: ', run_id)
        
    #     for condition in condition_list:
    #         filename = path + '/debug_trial_12_29_noise_' + condition + '_set_' + str(run_id) + '.csv'
    #         dt.run_sim(condition, filename)
    #################################


    ## run check VMF distribution code
    # dt.check_VMF_distribution()

    ## func_to_choose_plot_view_params
    # dt.func_to_choose_plot_view_params()

    # check Gaussian noise params in resampling
    # dt.check_resampling_gaussian_noise()

    # dt.check_pf_update()
    #################################

    ## debug knowledge calculation
    # dt.debug_knowledge_calculation()

    # split pickle files fo multiple runs
    # dt.split_pickle_file('models/augmented_taxi2', 'debug_trials_01_22_no_noise_w_feedback_study_1_run_4.pickle')
    ################################

    # save initial pf for various learning factors
    # sim_helpers.save_pf_models(500)

    ################################################

    # run individual simulations to check PF udpate process (fixed conditions)
    # path = 'models/augmented_taxi2'
    # file = 'debug_trials_01_23_no_noise_w_feedback_study_1_run_70.pickle'
    
    # params.default_learning_factor_teacher = 0.8
    # N_runs = 1
    # run_start_id = 33
    # viz_flag = True

    # for run_id in range(run_start_id, run_start_id+ N_runs):
    #     print('run_id: ', run_id)
        # asd.simulate_individual_runs(params, path, file, run_id, viz_flag = viz_flag, vars_filename_prefix = 'lf_update_set_m2_sim_no_noise')
        # asd.simulate_individual_runs_w_feedback(params, path, file, run_id, viz_flag = viz_flag, vars_filename_prefix = 'sim_teacher_noise_learner_no_noise_w_feedback')
    ################################
    
    # # run individual simulations to check PF udpate process (variable conditions)
    # path = 'models/augmented_taxi2'
    # # file = 'debug_trials_01_09_no_noise_study_1_run_8.pickle'
    # file = 'debug_trials_01_23_no_noise_w_feedback_study_1_run_70.pickle'
    
    # N_runs = 20
    # run_start_id = 1
    # viz_flag = False
    
    # params.default_learning_factor_teacher = 0.8
    # params.max_learning_factor = 0.9
    # learner_update_type = 'no_noise'

    # # team_params_learning = {'low': [0.65, 0.025, 0.05],'high': [0.8, 0.025, 0.05]}  # incorrect learns more
    # team_params_learning = {'low': [0.65, 0.05, 0.025],'high': [0.8, 0.05, 0.025]}  # correct learns more

    # # team_composition_list = [[0,0,2]]
    # team_composition_list = [[0,0,0], [0,0,2], [0,2,2], [2,2,2]]
    
    # sim_conditions = get_sim_conditions(team_composition_list, N_runs, run_start_id)

    # # run simulation 
    # for run_id in range(run_start_id, run_start_id+N_runs):
    #     print('sim_conditions run_id:', sim_conditions[run_id - run_start_id], '. sim_conditions: ', sim_conditions )
    #     if run_id == sim_conditions[run_id - run_start_id][0]:
    #         team_composition_for_run = sim_conditions[run_id - run_start_id][1]
    #     else:
    #         RuntimeError('Error in sim conditions')
    #         break

    #     ilcr = np.zeros(params.team_size)
    #     rlcr = np.zeros([params.team_size, 2])

    #     for j in range(params.team_size):
    #         if team_composition_for_run[j] == 0: 
    #             ilcr[j] = team_params_learning['low'][0]
    #             rlcr[j,0] = team_params_learning['low'][1]
    #             rlcr[j,1] = team_params_learning['low'][2]     
    #         elif team_composition_for_run[j] == 1:
    #             ilcr[j] = team_params_learning['med'][0]
    #             rlcr[j,0] = team_params_learning['med'][1]
    #             rlcr[j,1] = team_params_learning['med'][2]
    #         elif team_composition_for_run[j] == 2:
    #             ilcr[j] = team_params_learning['high'][0]
    #             rlcr[j,0] = team_params_learning['high'][1]
    #             rlcr[j,1] = team_params_learning['high'][2]
        
    #     print('Simulation run: ' + str(run_id) + '. Demo strategy: ' + '. Team composition:' + str(team_composition_for_run))
    #     args = (team_composition_for_run, ilcr, rlcr)
    #     asd.simulate_individual_runs_w_feedback(params, path, file, run_id, viz_flag = viz_flag, feedback_flag=True, review_flag=True, vars_filename_prefix = 'sim_teacher_noise_learner_no_noise_w_feedback_correct', args = args)
    
    ################################
    # run individual simulations to check effect of feedback (only one person in the team)
    # path = 'models/augmented_taxi2'
    # # file = 'debug_trials_01_09_no_noise_study_1_run_8.pickle'
    # file = 'debug_trials_01_23_no_noise_w_feedback_study_1_run_70.pickle'
    
    # N_runs = 10
    # run_start_id = 51
    # viz_flag = False
    
    # params.default_learning_factor_teacher = 0.8
    # params.max_learning_factor = 1.0
    # learner_update_type = 'no_noise'
    # params.team_size = 1
    # team_params_learning = {'low': [0, 0.025, 0.05]}  # incorrect learns more
    # # team_params_learning = {'low': [0.65, 1, 0.05, 0.025]}  # correct learns more

    # team_composition_list = [[0]]
    
    # sim_conditions = get_sim_conditions(team_composition_list, N_runs, run_start_id)

    # # run simulation 
    # team_params_learning_array = np.linspace(0.55, 1)
    # for run_id in range(run_start_id, run_start_id+N_runs):
    #     print('sim_conditions run_id:', sim_conditions[run_id - run_start_id], '. sim_conditions: ', sim_conditions )
    #     if run_id == sim_conditions[run_id - run_start_id][0]:
    #         team_composition_for_run = sim_conditions[run_id - run_start_id][1]
    #     else:
    #         RuntimeError('Error in sim conditions')
    #         break

    #     ilcr = np.zeros(params.team_size)
    #     rlcr = np.zeros([params.team_size, 2])

    #     for j in range(params.team_size):
    #         ilcr[j] = team_params_learning_array[run_id - run_start_id]
    #         rlcr[j,0] = team_params_learning['low'][1]
    #         rlcr[j,1] = team_params_learning['low'][2]     

        
    #     print('Simulation run: ' + str(run_id),  '.ilcr: ', ilcr, '. rlcr: ', rlcr)
    #     args = (team_composition_for_run, ilcr, rlcr)
    #     asd.simulate_individual_runs_w_feedback(params, path, file, run_id, viz_flag = viz_flag, vars_filename_prefix = 'sim_teacher_noise_learner_no_noise_w_feedback_reverse_uf_test', args = args)
    
    # #################################

    # plot pf update process
    # path = 'models/augmented_taxi2'
    # file_prefix = 'debug_trials_01_23_no_noise_w_feedback_study_1_run_70'
    # asd.plot_pf_updates(path, file_prefix)
        
    ##########################

    # # check information gain vs learning factor
    path = 'models/augmented_taxi2'
    dt.check_ig_uf_relation(path)

    ###########################

    # check constraints area
    # dt.check_constraints_area()

    x=1