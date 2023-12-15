import cProfile

import debugging_team as dt


if __name__ == "__main__":
    
    ## run response sampling code
    N_runs = 3
    run_start_id = 1
    # condition_list = ['cluster_random', 'cluster_weight', 'particles']
    condition_list = ['cluster_random', 'cluster_weight']

    for run_id in range(run_start_id, run_start_id + N_runs):
        print('run: ', run_id)
        
        for condition in condition_list:
            filename = 'debug_response_N100_no_learning_' + condition + '_set_' + str(run_id) + '.csv'
            dt.run_sim(condition, filename)
    #################################


    ## run check VMF distribution code
    # dt.check_VMF_distribution()


    