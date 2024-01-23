import cProfile

import debugging_team as dt
import analyze_sim_data as asd
import params_team as params
import simulation.sim_helpers as sim_helpers

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')


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

    # # save initial pf for various learning factors
    # sim_helpers.save_pf_models(500)

    # run individual simulations to check PF udpate process
    path = 'models/augmented_taxi2'
    file = 'debug_trials_01_22_no_noise_w_feedback_study_1_run_3.pickle'
    
    params.default_learning_factor_teacher = 0.8
    N_runs = 10
    run_start_id = 1
    viz_flag = False

    for run_id in range(run_start_id, run_start_id+ N_runs):
        print('run_id: ', run_id)
        # asd.simulate_individual_runs(params, path, file, run_id, viz_flag = viz_flag, vars_filename_prefix = 'lf_update_set_m2_sim_noise')
        asd.simulate_individual_runs_w_feedback(params, path, file, run_id, viz_flag = viz_flag, vars_filename_prefix = 'lf_update_set_m2_sim_no_noise_w_feedback')
    ################################