import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize



import pymc as pm
import dill as pickle
import pandas as pd
import seaborn as sns
import copy
import numpy as np
import arviz as az
from pyDOE import lhs, fullfact

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split


import params_team as params
import teams.teams_helpers as team_helpers
import analyze_study_3_data as asd

from multiprocessing import Pool


np.random.seed(237)



# def simulate_objective(learning_params):
# def simulate_objective(u, delta_c, delta_i):
def simulate_objective(x):

    u, delta_c, delta_i = x

    mdp_domain = 'augmented_taxi2'
    viz_flag = True
    learner_update_type = 'no_noise'

    # sim params
    max_learning_factor = params.max_learning_factor
    # initial_learning_factor = copy.deepcopy(learning_params['initial_learning_factor'])
    # learning_factor_delta = copy.deepcopy(learning_params['learning_factor_delta'])

    # initial_learning_factor = [learning_params[0]]
    # learning_factor_delta = [learning_params[1]/2, learning_params[1]]

    # learning_factor_delta = [learning_factor_delta_incorrect/2, learning_factor_delta_incorrect]

    # learning_factor_delta = [learning_factor_delta_correct, learning_factor_delta_incorrect]

    print('initial_learning_factor:', u, 'learning_factor_delta:', [delta_c, delta_i])

    # initialize (simulated) learner particle filters
    initial_learner_pf = copy.deepcopy(all_learner_pf['p1'])

    # prior interaction data
    prior_test_constraints = [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]  # constraints of the first concept/KC
    all_test_constraints = {  1: [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], \
                                2: [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], \
                                3: [np.array([[1, 1, 0]])]}
    min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -4]])]

    
    # initialize dataframes to save probability data
    simulated_interaction_data = pd.DataFrame()


    # simulate teaching loop
    prev_kc_id = 1
    demo_id = 1
    unit_constraints = [np.array([[0, 0, -1]])]
    objective = 0
    # objective = []

    learning_factor = u

    # plotting
    color_dict = {'demo': 'blue', 'remedial demo': 'purple', 'diagnostic test': 'red',  'remedial test': 'pink', 'diagnostic feedback': 'yellow', 'remedial feedback': 'orange', 'final test': 'green'}
    


    for user_id, user_interaction_data in prepared_interaction_data.iterrows():

        vars_filename_prefix = 'param_fit_user_' + str(user_id) + 'init_lf_' + str(u) + 'lf_delta_' + str(delta_c) + '_' + str(delta_i)
        # vars_filename_prefix = 'param_fit_user_' + str(user_id) + 'init_lf_' + str(initial_learning_factor) + 'lf_delta_' + str(learning_factor_delta)
        N_final_tests_correct = 0

        # print('user_id:', user_id, '.Len user_data:', len(user_interaction_data['kc_id']))

        # params to plot
        user_interaction_type = []
        user_learning_factor = []
        user_prob_BEC = []
        user_prob_KC = []
        user_loop_id = []
        concept_end_id = []
        plot_id = 0
        for loop_id in range(len(user_interaction_data['kc_id'])):
            
            # print('user_data_kcid:', user_data['kc_id'])
            current_kc_id = user_interaction_data['kc_id'][loop_id]
            current_interaction_type = user_interaction_data['interaction_types'][loop_id]
            # is_opt_response = user_data['is_opt_response'][loop_id]
            current_interaction_constraints = user_interaction_data['interaction_constraints'][loop_id]
            current_test_constraints = user_interaction_data['test_constraints'][loop_id]   

            # Not needed for now!
            # if current_interaction_type != 'final test' and current_kc_id <= max_kc:
            #     test_constraints = all_test_constraints[current_kc_id]
            # else:
            #     test_constraints = min_BEC_constraints

            if current_kc_id > prev_kc_id:
                # learning_factor = copy.deepcopy([initial_learning_factor])
                learning_factor = u
                # print('New KC! Resetting learning factor to initial value: ', learning_factor)
                concept_end_id.append(plot_id-1)

            # updates for various interaction types
            # Prior
            if current_interaction_type == 'prior':
                learner_pf = copy.deepcopy(initial_learner_pf)
            
            # Demo
            if current_interaction_type == 'demo':
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = 'Learner belief after demo. Interaction ID:  ' + str(loop_id) + ' for KC: ' + str(current_kc_id), viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)


            # Diagnostic Test
            if current_interaction_type == 'diagnostic test':
                # Nothing changes
                response_type = user_interaction_data['test_response_type'][loop_id][0]


            # Diagnostic Feedback
            if current_interaction_type == 'diagnostic feedback':
                # print('response_type: ', response_type)
                # if response_type == 'correct':
                #     learning_factor[0] = min(learning_factor[0] + learning_factor_delta[0], max_learning_factor)
                # elif response_type == 'incorrect':
                #     learning_factor[0] = min(learning_factor[0] + learning_factor_delta[1], max_learning_factor)
                # else:
                #     RuntimeError('Invalid response type')

                if response_type == 'correct':
                    learning_factor = min(learning_factor + delta_c, max_learning_factor)
                elif response_type == 'incorrect':
                    learning_factor = min(learning_factor + delta_i, max_learning_factor)
                else:
                    RuntimeError('Invalid response type')

                # updated learner model with corrective feedback
                plot_title =  ' Learner after corrective feedback for KC ' + str(current_kc_id)
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename = vars_filename_prefix, model_type = learner_update_type)

            # Remedial Demo
            if current_interaction_type == 'remedial demo':
                plot_title =  'Learner belief after remedial demo. Interaction ID: ' + str(loop_id) + ' for KC ' + str(current_kc_id)
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)
                
            # Remedial Test
            if current_interaction_type == 'remedial test':
                response_type = user_interaction_data['test_response_type'][loop_id][0]

            # Remedial Feedback
            if current_interaction_type == 'remedial feedback':
                # print('response_type: ', response_type)
                if response_type == 'correct':
                    learning_factor = min(learning_factor + delta_c, max_learning_factor)
                elif response_type == 'incorrect':
                    learning_factor = min(learning_factor + delta_i, max_learning_factor)
                else:
                    RuntimeError('Invalid response type')

                # updated learner model with corrective feedback
                plot_title =  ' Learner after remedial feedback for KC ' + str(current_kc_id)
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename = vars_filename_prefix, model_type = learner_update_type)

            # Final Test Performance
            if current_interaction_type == 'final test':
                if user_interaction_data['is_opt_response'][loop_id] == 1:
                    N_final_tests_correct += 1

            if 'prior_' not in current_interaction_type:
                # calculate probability of correct response
                learner_pf.calc_particles_probability(current_test_constraints)
                prop_particles_KC = learner_pf.particles_prob_correct


                learner_pf.calc_particles_probability(min_BEC_constraints)
                prop_particles_BEC = learner_pf.particles_prob_correct
                # print('loop_id: ', loop_id, 'interaction: ', current_interaction_type, 'prop_particles_BEC: ', prop_particles_BEC)

                # update loop vars
                user_interaction_type.append(current_interaction_type)
                user_learning_factor.append(learning_factor)
                user_prob_BEC.append(prop_particles_BEC)
                user_prob_KC.append(prop_particles_KC)
                user_loop_id.append(plot_id)
                plot_id += 1

            # update loop kcid
            prev_kc_id = current_kc_id

        
        
        # user_plot_data = pd.DataFrame({'loop_id': user_loop_id, 'interaction_type': user_interaction_type, 'learning_factor': user_learning_factor, 'prob_KC': user_prob_KC, 'prob_BEC': user_prob_BEC})
            
        # plt.figure(user_id)
        # ax0 = plt.gca()
        # print('user_id: ', user_id, 'user_plot_data:', user_plot_data)
        # sns.lineplot(data=user_plot_data, x='loop_id', y='prob_BEC', ax=ax0, color = 'blue').set(title='Obj. func. learning dynamics for user: ' + str(user_id))
        # sns.lineplot(data=user_plot_data, x='loop_id', y='prob_KC', ax=ax0, color = 'brown')
        # sns.lineplot(data=user_plot_data, x='loop_id', y='learning_factor', ax=ax0, color = 'green')

            
        # for id, row in user_plot_data.iterrows():
        #     print('id:', id)
        #     if row['interaction_type'] != 'prior':
        #         plt.axvspan(user_plot_data['loop_id'].iloc[id-1], row['loop_id'], alpha=0.2, color=color_dict[row['interaction_type']])
        #         plt.text(row['loop_id']-0.5, 0.3, row['interaction_type'], rotation=90, fontsize=12, weight="bold")

        # for id in concept_end_id:
        #     plt.axvline(x=id, color='black', linestyle='--', linewidth=2)
        
        # plt.show()

        # calculate final probability
        learner_pf.calc_particles_probability(min_BEC_constraints)
        prop_particles_BEC = learner_pf.particles_prob_correct

        # final test performance
        test_perf = N_final_tests_correct/6

        # update objective function
        objective += np.abs(prop_particles_BEC - test_perf)
        # objective.append(prop_particles_BEC)

        # print('user_id: ', user_id, 'N_final_tests_correct: ', N_final_tests_correct, 'prop_particles_BEC: ', prop_particles_BEC, 'test_perf: ', test_perf, 'objective: ', prop_particles_BEC - test_perf)

        

    return objective/len(prepared_interaction_data)
        
    # return objective

#######################################



#######################################

def get_parameter_combination(params_to_study, num_samples):
    
    N = len(params_to_study)
    lhs_sample = lhs(N, samples=num_samples, criterion = 'maximin')


    sample_combinations = []
    for i in range(len(lhs_sample)):
        sample_combinations.append([])
        for j, param in enumerate(params_to_study):
            sample_combinations[i].append(params_to_study[param][0] + lhs_sample[i][j]*(params_to_study[param][1] - params_to_study[param][0]))

    with open('data/simulation/sim_experiments/parameter_estimation/param_combinations.pickle', 'wb') as f:
        pickle.dump(sample_combinations, f)

    return sample_combinations

##################################

def load_train_test_data(learner_type = 'low'):

    domain = 'at'
    filename = 'data/simulation/sim_experiments/parameter_estimation/train_test_data_' + learner_type + '_domain_' + domain + '.pickle'

    try:
        # load train test data
        with open(filename, 'rb') as f:
            [X_train, X_test, y_train, y_test] = pickle.load(f)
    except:
        # prepare train test data
        with open ('data/prepared_interaction_data.pickle', 'rb') as f:
            all_interaction_data = pickle.load(f)

        with open('data/user_data_w_flag.pickle', 'rb') as f:
            all_user_data = pickle.load(f)

        
        if learner_type == 'test':
            user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & \
                                ((all_user_data['N_final_correct_at'] == 2))]
        elif learner_type == 'low':
            user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & \
                            ((all_user_data['N_final_correct_at'] == 2) | (all_user_data['N_final_correct_at'] == 3) | (all_user_data['N_final_correct_at'] == 4))]
        elif learner_type == 'high':
            user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & \
                            ((all_user_data['N_final_correct_at'] == 5) | (all_user_data['N_final_correct_at'] == 6))]
            
        
        # input and output data
        unique_user_ids = user_data['user_id'].unique()
        prepared_interaction_data = pd.DataFrame()

        for user_id in unique_user_ids:
            prepared_interaction_data = prepared_interaction_data.append(all_interaction_data[all_interaction_data['user_id'] == user_id], ignore_index=True)

        interaction_output = user_data['N_final_correct_at']/6
        
        # split into test train datasets
        X_train, X_test, y_train, y_test = train_test_split(prepared_interaction_data, interaction_output, test_size=0.25, random_state=0)

        
        with open(filename, 'wb') as f:
            pickle.dump([X_train, X_test, y_train, y_test], f)
    
    return X_train, X_test, y_train, y_test

#################################

def load_abc_data(learner_type = 'low'):

    domain = 'at'
    filename = 'data/simulation/sim_experiments/parameter_estimation/abc_data_' + learner_type + '_domain_' + domain + '.pickle'

    try:
        # load data
        with open(filename, 'rb') as f:
            [prepared_interaction_data, interaction_output] = pickle.load(f)

    except:
        # prepare train test data
        with open ('data/prepared_interaction_data.pickle', 'rb') as f:
            all_interaction_data = pickle.load(f)

        with open('data/user_data_w_flag.pickle', 'rb') as f:
            all_user_data = pickle.load(f)

        
        if learner_type == 'test':
            user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & \
                                ((all_user_data['N_final_correct_at'] == 2))]
        elif learner_type == 'low':
            user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & \
                            ((all_user_data['N_final_correct_at'] == 2) | (all_user_data['N_final_correct_at'] == 3) | (all_user_data['N_final_correct_at'] == 4))]
        elif learner_type == 'high':
            user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & \
                            ((all_user_data['N_final_correct_at'] == 5) | (all_user_data['N_final_correct_at'] == 6))]
        
        # input and output data
        unique_user_ids = user_data['user_id'].unique()
        prepared_interaction_data = pd.DataFrame()

        for user_id in unique_user_ids:
            prepared_interaction_data = prepared_interaction_data.append(all_interaction_data[all_interaction_data['user_id'] == user_id], ignore_index=True)

        interaction_output = user_data['N_final_correct_at']/6

        with open(filename, 'wb') as f:
            pickle.dump([prepared_interaction_data, interaction_output], f)

    return prepared_interaction_data, interaction_output

###########################



def plot_param_distribution(filename):
    
    max_error = 0.1
    
    with open (filename, 'rb') as f:
        parameter_estimation_output = pickle.load(f)

    # fig = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 4))
        
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1, projection='3d')

    fig2, ax2 = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 4))
    
    # scatter plot
    hist_plot_data = pd.DataFrame()
    for params_id in range(len(parameter_estimation_output)):
        if parameter_estimation_output['objective'][params_id] < max_error:
            plot_color = 'blue'
            hist_plot_data = hist_plot_data.append(parameter_estimation_output.iloc[params_id], ignore_index=True)
        else:
            plot_color = 'red'
        

            
        # axs[0].scatter(parameter_estimation_output['initial_learning_factor'][params_id], parameter_estimation_output['learning_factor_delta_correct'][params_id], color=plot_color)
        # axs[1].scatter(parameter_estimation_output['initial_learning_factor'][params_id], parameter_estimation_output['learning_factor_delta_incorrect'][params_id], color=plot_color)
        # axs[2].scatter(parameter_estimation_output['learning_factor_delta_correct'][params_id], parameter_estimation_output['learning_factor_delta_incorrect'][params_id], color=plot_color)   
        
        axs.scatter(parameter_estimation_output['initial_learning_factor'][params_id], parameter_estimation_output['learning_factor_delta_correct'][params_id], parameter_estimation_output['learning_factor_delta_incorrect'][params_id], color=plot_color)



    # axs[0].set_xlabel('Initial learning factor')
    # axs[0].set_ylabel('Learning factor delta correct')
    # axs[1].set_xlabel('Initial learning factor')
    # axs[1].set_ylabel('Learning factor delta incorrect')
    # axs[2].set_xlabel('Learning factor delta correct')
    # axs[2].set_ylabel('Learning factor delta incorrect')

    axs.set_xlabel('Initial learning factor')
    axs.set_ylabel('Learning factor delta correct')
    axs.set_zlabel('Learning factor delta incorrect')



    # histogram plot
    sns.histplot(data=hist_plot_data, x='initial_learning_factor', ax=ax2[0], stat='probability', kde=True, color='blue')
    sns.histplot(data=hist_plot_data, x='learning_factor_delta_correct', ax=ax2[1], stat='probability', kde=True, color='blue')
    sns.histplot(data=hist_plot_data, x='learning_factor_delta_incorrect', ax=ax2[2], stat='probability', kde=True, color='blue')
    
    plt.show()

###############################################


if __name__ == "__main__":

    params.team_size = 1
    params.max_learning_factor = 1.0
    

    all_learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = [0.8], team_prior = params.team_prior, pf_flag='learner', model_type = 'no_noise')




    ################################

    # ## Custom grid search
    # learner_type = 'test'  # low, high, test
    # dataset_type = 'train'  # train or test
    # N_runs = 2
    # filename_prefix = 'data/simulation/sim_experiments/parameter_estimation/parameter_estimation_output_' + learner_type + '_' + dataset_type

    # # load train test data
    # x_train, x_test, y_train, y_test = load_train_test_data(learner_type = learner_type)

    # if dataset_type == 'train':
    #     prepared_interaction_data = x_train
    #     interaction_output = y_train
    # elif dataset_type == 'test':
    #     prepared_interaction_data = x_test
    #     interaction_output = y_test
    # else:
    #     RuntimeError('Invalid dataset type')

    # params_list = {'initial_learning_factor': np.arange(0.5, 0.9, 0.05), 'learning_factor_delta_correct': np.arange(0.0, 0.2, 0.02), 'learning_factor_delta_incorrect': np.arange(0.0, 0.2, 0.02)}
    # pg = list(ParameterGrid(params_list))
    # params_to_eval = []
    # for pg_ind in pg:
    #     if (pg_ind['learning_factor_delta_incorrect'] > pg_ind['learning_factor_delta_correct']) and (pg_ind['initial_learning_factor'] > 0.5):
    #         # params_to_eval.append(pg_ind)

    #         for run_id in range(N_runs):
    #             params_to_eval.append([pg_ind['initial_learning_factor'], pg_ind['learning_factor_delta_correct'], pg_ind['learning_factor_delta_incorrect']])
    
    # print(len(params_to_eval))
    

    # parameter_estimation_output = pd.DataFrame()

    # # Prepare parameters for parallel processing
    # # params_grid_run_eval = [(params_to_eval[params_id]) for params_id in range(len(params_to_eval))]

    # # Set up multiprocessing Pool
    # with Pool() as pool:
    #     results = pool.map(simulate_objective, params_to_eval)

    # # Flatten the list of results
    # objective = [item for sublist in results for item in sublist]

    # for params_id in range(len(params_to_eval)):

    #     for run_id in range(N_runs):

    #         print('Running params_id:', params_id, 'params:', params_to_eval[params_id])
    #         # objective = simulate_objective(params_to_eval[params_id]['initial_learning_factor'], params_to_eval[params_id]['learning_factor_delta_correct'], params_to_eval[params_id]['learning_factor_delta_incorrect'])
    #         # print('Objective:', objective)

    #         output_data = {'params_id': params_id, 'run_id': run_id, 'initial_learning_factor': params_to_eval[params_id]['initial_learning_factor'], 'learning_factor_delta_correct': params_to_eval[params_id]['learning_factor_delta_correct'], \
    #                        'learning_factor_delta_incorrect': params_to_eval[params_id]['learning_factor_delta_incorrect']}
                
    #         parameter_estimation_output = parameter_estimation_output.append(output_data, ignore_index=True)

    # parameter_estimation_output['objective'] = objective

    # with open(filename_prefix + '.pickle', 'wb') as f:
    #     pickle.dump(parameter_estimation_output, f)

    # parameter_estimation_output.to_csv(filename_prefix + '.csv', index=False)

    #####################
    ## Custom search on lsd params
    # run_start_id = 1
    # N_combinations = 50
    # N_runs = 10

    # params_to_study = {'initial_learning_factor': [0.5, 1], 'learning_factor_delta_correct': [0.0, 0.2], 'learning_factor_delta_incorrect': [0.0, 0.2]}

    # try:
    #     with open('data/simulation/sim_experiments/parameter_estimation/param_combinations.pickle', 'rb') as f:
    #         parameter_combinations = pickle.load(f)
    
    #     if len(parameter_combinations) != (run_start_id + N_combinations-1):
    #         parameter_combinations = get_parameter_combination(params_to_study, N_combinations)
    #         print('Parameter combinations generated: ', parameter_combinations)
    #     else:
    #         print('Parameter combinations loaded: ', parameter_combinations)
    # except:
    #     parameter_combinations = get_parameter_combination(params_to_study, N_combinations)
    #     print('Parameter combinations generated: ', parameter_combinations)

    




    # parameter_estimation_output = pd.DataFrame()
    
    # for params_id in range(run_start_id, run_start_id + N_combinations):

    #     for run_id in range(N_runs):

    #         print('Running params_id:', params_id, 'run_id:', run_id, 'params:', parameter_combinations[params_id-1])
        
    #         # Learner and teacher model params sensitivity analysis
    #         learning_params = {'initial_learning_factor': [parameter_combinations[params_id-1][0]], 
    #                            'learning_factor_delta': [parameter_combinations[params_id-1][1], parameter_combinations[params_id-1][2]]}
            
    #         objective = simulate_objective(learning_params)

    #         output_data = {'params_id': params_id, 'run_id': run_id, 'initial_learning_factor': parameter_combinations[params_id-1][0], 'learning_factor_delta_correct': parameter_combinations[params_id-1][1], 'learning_factor_delta_incorrect': parameter_combinations[params_id-1][2], 'objective': objective}
            
    #         parameter_estimation_output = parameter_estimation_output.append(output_data, ignore_index=True)


    # with open('data/simulation/sim_experiments/parameter_estimation/parameter_estimation_output.pickle', 'wb') as f:
    #     pickle.dump(parameter_estimation_output, f)
    
    # parameter_estimation_output.to_csv('data/simulation/sim_experiments/parameter_estimation/parameter_estimation_output.csv', index=False)
    
    ######################################
    # ## Bayesian hyperparameter optimization (somewhat worksl gave the highest params values for high learners; point estimate only!)

    params_to_study = {'initial_learning_factor': (0.5, 1), 'learning_factor_delta_incorrect': (0.04, 0.2)}

    params_to_study = {'initial_learning_factor': np.arange(0.5, 1, 0.05), 'learning_factor_delta_incorrect': np.arange(0.02, 0.2, 0.02)}

    learner_type = 'high'

    prepared_interaction_data, interaction_output = load_abc_data(learner_type = learner_type)

    res = gp_minimize(simulate_objective,                  # the function to minimize
                      [(0.51, 1), (0.02, 0.2), (0.02, 0.2)],      # the bounds on each dimension of x
                      acq_func="EI",      # the acquisition function
                      n_calls=100,         # the number of evaluations of f
                      n_random_starts=20,  # the number of random initialization points
                      noise=0,              # the noise level (optional)
                      random_state=1234,   # the random seed
                      n_jobs=-1)       # # Set n_jobs to -1 for parallel processing
    
    with open('data/simulation/sim_experiments/parameter_estimation/bayesian_optimization_output_' + learner_type + '.pickle', 'wb') as f:
        pickle.dump(res, f)

    print(res)
    # # #############

    # ## Random CV search (Not working; need to create an estimator object to pass to RandomizedSearchCV)
    # # Create a scorer object based on the custom objective function
    # custom_scorer = make_scorer(simulate_objective, greater_is_better=False)

    # # Initialize RandomizedSearchCV with the custom objective function
    # random_search = RandomizedSearchCV(estimator=None, param_distributions=params_to_study, scoring=custom_scorer, n_iter=100, random_state=42)

    # # Perform the random search
    # random_search.fit(X=None, y=None)  # Pass X and y as None since we're not doing actual modeling

    # # Get the best parameters found by the random search
    # best_params = random_search.best_params_

    # # Get the best score found by the random search
    # best_score = -random_search.best_score_  # negate the score because make_scorer was set with greater_is_better=False

    # print("Best Parameters:", best_params)
    # print("Best Score:", best_score)


    ###############################################

    # # Bayesian model (not working)
    # basic_model = pm.Model()

    # with basic_model:

    #     # Priors for unknown model parameters
    #     u = pm.Normal('u', sigma=1)
    #     # delta_c = pm.Normal('delta_c', sigma=1)
    #     delta_i = pm.Normal('delta_i', sigma=1)

    #     # learning_params = {'initial_learning_factor': [u], 'learning_factor_delta': [delta_i/2, delta_i]}

    #     # # Expected value of outcome
    #     # mu = simulate_objective(u, delta_i)

    #     # # Likelihood (sampling distribution) of observations
    #     # Y_obs = pm.Normal('Y_obs', mu=mu, sigma=u_sd, observed=interaction_output)

    #     # Custom likelihood (negative because PyMC maximizes log likelihood)
    #     pm.Potential('likelihood', -simulate_objective(u, delta_c, delta_i))


    #     # draw 500 posterior samples
    #     trace = pm.sample(500)

    # pm.summary(trace)

    ## SMC - Approximate Bayesian Computation (trying...)
    # learner_type = 'test'

    # prepared_interaction_data, interaction_output = load_abc_data(learner_type = learner_type)

    # with pm.Model() as model_lv:
    #     u = pm.Normal("u", sigma=1)
    #     delta_c = pm.Normal("delta_c", sigma=1)
    #     delta_i = pm.Normal("delta_i", sigma=1)

    #     sim = pm.Simulator("sim", simulate_objective, params=(u, delta_c, delta_i), epsilon=10, observed=interaction_output)

    #     idata_lv = pm.sample_smc()

    # ## SMC-ABC practice
    # def normal_sim(rng, a, b, size=1000):
    #     return rng.normal(a, b, size=size)

    # data = np.random.normal(loc=0, scale=1, size=1000)

    # with pm.Model() as example:
    #     a = pm.Normal("a", mu=0, sigma=5)
    #     b = pm.HalfNormal("b", sigma=1)
    #     s = pm.Simulator("s", normal_sim, params=(a, b), sum_stat="sort", epsilon=1, observed=data)

    #     idata = pm.sample_smc()
    #     idata.extend(pm.sample_posterior_predictive(idata))

    # az.plot_trace(idata, kind="rank_vlines")
    # az.summary(idata, kind="stats")
    # az.plot_ppc(idata, num_pp_samples=500)


    ####
    # objective = simulate_objective(0.8, 0.04, 0.08)

    # print(objective)

    ############################

    # # pratcice
    # # Initialize random number generator
    # RANDOM_SEED = 8927
    # rng = np.random.default_rng(RANDOM_SEED)
    # az.style.use("arviz-darkgrid")

    # # True parameter values
    # alpha, sigma = 1, 1
    # beta = [1, 2.5]

    # # Size of dataset
    # size = 100

    # # Predictor variable
    # X1 = np.random.randn(size)
    # X2 = np.random.randn(size) * 0.2

    # # Simulate outcome variable
    # Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

    # basic_model = pm.Model()

    # with basic_model:
    #     # Priors for unknown model parameters
    #     alpha = pm.Normal("alpha", mu=0, sigma=10)
    #     beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    #     sigma = pm.HalfNormal("sigma", sigma=1)

    #     # Expected value of outcome
    #     mu = alpha + beta[0] * X1 + beta[1] * X2

    #     print('mu:', mu)

    #     # Likelihood (sampling distribution) of observations
    #     Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    ###########################

    # plot custom grid search results

    # filename = 'data/simulation/sim_experiments/parameter_estimation/parameter_estimation_output_test_train.pickle'
    # plot_param_distribution(filename)