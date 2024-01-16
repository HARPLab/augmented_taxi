import cProfile

import debugging_team as dt


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
    dt.split_pickle_file('data/simulation/sim_experiments/new_data', 'debug_trials_01_09_no_noise_study_1_run_5.pickle')