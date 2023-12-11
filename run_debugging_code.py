import cProfile

import debugging_team as dt


if __name__ == "__main__":
    
    N_runs = 10

    for i in range(N_runs):
        print('run: ', i)
        
        for condition in ['cluster_random', 'cluster_weight', 'particles']:
            
            filename = 'debug_response_' + condition + '_set_' + str(i) + '.csv'
            dt.run_sim(condition, filename)


    