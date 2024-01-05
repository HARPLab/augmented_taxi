import params
import dill as pickle
import numpy as np
import policy_summarization.BEC_helpers as BEC_helpers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from policy_summarization import policy_summarization_helpers
import json
from collections import defaultdict
from simple_rl.utils import mdp_helpers
import policy_summarization.multiprocessing_helpers as mp_helpers
import os
import warnings
import pandas as pd
import dill as pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
For managing and processing data related to the user study 
'''


def find_human_response():

    path = 'data/'

    with open(path + '/dfs_f23_processed.pickle', 'rb') as f:
        dfs_users_processed, dfs_trials_processed, dfs_domain_processed  = pickle.load(f)

    # dfs_users_processed.to_csv(path + '/dfs_users_processed.csv')
    # dfs_trials_processed.to_csv(path + '/dfs_trials_processed.csv')
    # dfs_domain_processed.to_csv(path + '/dfs_domain_processed.csv')


    # with open(path + '/dfs_f23.pickle', 'rb') as f:
    #     dfs_f23_1, dfs_f23_2, dfs_f23_3  = pickle.load(f)

    # dfs_f23_1.to_csv(path + '/dfs_f23_1.csv')
    # dfs_f23_2.to_csv(path + '/dfs_f23_2.csv')
    # dfs_f23_3.to_csv(path + '/dfs_f23_3.csv')

    dfs_trials_processed_at = dfs_trials_processed[(dfs_trials_processed['domain'] == 'at')]
    dfs_trials_processed_at.to_csv(path + '/dfs_trials_processed_at.csv')

    # print(dfs_trials_processed_at)

    unique_user_ids = dfs_trials_processed_at['user_id'].unique()
    N_interactions = []
    N_final_correct = []
    user_ids_low_learners = []
    user_ids_high_learners = []
    N_interactions_low_learners = []
    N_interactions_high_learners = []
    for user_id in unique_user_ids:

        N_interactions_user = len(dfs_trials_processed_at[dfs_trials_processed_at['user_id'] == user_id]) - 6
        N_interactions.append(N_interactions_user) # remove 6 final tests

        N_final_correct_user = len(dfs_trials_processed_at[(dfs_trials_processed_at['user_id'] == user_id) & (dfs_trials_processed_at['interaction_type'] == 'final test') & (dfs_trials_processed_at['is_opt_response'] == 1)])
        N_final_correct.append(N_interactions_user)

        if N_final_correct_user < 5:
            user_ids_low_learners.append(user_id)
            N_interactions_low_learners.append(N_interactions_user)
        else:
            user_ids_high_learners.append(user_id)
            N_interactions_high_learners.append(N_interactions_user)

        



    valid_data_idx = np.where(np.array(N_interactions) != 0)[0]
    print('valid_data_idx: ', len(valid_data_idx))

    valid_N_interactions = np.array(N_interactions)[valid_data_idx]
    valid_N_final_correct = np.array(N_final_correct)[valid_data_idx]
    # print('valid_N_interactions: ', valid_N_interactions)
    # print('N_interactions: ', N_interactions[valid_data_idx], 'len unique_user_ids: ', len(unique_user_ids[valid_data_idx]), 'len N_interactions: ', len(N_interactions[valid_data_idx]))

    # plot
    fig, ax = plt.subplots(ncols=3)
    ax[0].hist(valid_N_interactions, bins='auto')
    ax[0].set_title('Histogram of learning interactions for each user')
    ax[0].set_xlabel('Number of interactions')
    ax[0].set_ylabel('Count')
    ax[0].grid(True)

    ax[1].hist(np.array(N_interactions_low_learners)[np.where(np.array(N_interactions_low_learners) != 0)[0]], bins='auto')
    ax[1].set_title('Number of interactions for low leaners')
    ax[1].set_xlabel('Number of interactions')
    ax[1].set_ylabel('Count')
    ax[1].grid(True)

    ax[2].hist(np.array(N_interactions_high_learners)[np.where(np.array(N_interactions_high_learners) != 0)[0]], bins='auto')
    ax[2].set_title('Number of interactions for high leaners')
    ax[2].set_xlabel('Number of interactions')
    ax[2].set_ylabel('Count')
    ax[2].grid(True)

    fig2, ax2 = plt.subplots()
    ax2.hist(valid_N_final_correct, bins='auto')
    ax2.set_title('Histogram of final correct responses for each user')
    ax2.set_xlabel('Number of correct responses')
    ax2.set_ylabel('Count')
    ax2.grid(True)
    
    plt.show()






    return 1







if __name__ == "__main__":

    # find human response distribution
    find_human_response()