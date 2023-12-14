import cProfile

import debugging_team as dt


if __name__ == "__main__":
    
    ## run response sampling code
    N_runs = 10
    # condition_list = ['cluster_random', 'cluster_weight', 'particles']
    condition_list = ['particles']

    for i in range(N_runs):
        print('run: ', i)
        
        for condition in condition_list:
            filename = 'debug_response_N100_' + condition + '_set_' + str(i) + '.csv'
            dt.run_sim(condition, filename)
    ##################################


    ## run check VMF distribution code
    # dt.check_VMF_distribution()


    