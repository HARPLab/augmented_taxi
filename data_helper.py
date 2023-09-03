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
from policy_summarization.flask_user_study_utils import extract_mdp_dict

'''
For managing data related to the user study 
'''

def plot_BEC_histogram(data_loc, weights, step_cost_flag):
    try:
        with open('models/' + data_loc + '/raw_BEC_lengths.pickle', 'rb') as f:
            BEC_lengths = pickle.load(f)
    except:
        with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
            policy_constraints, min_subset_constraints_record, env_record, traj_record = pickle.load(f)

        BEC_lengths = np.zeros(len(min_subset_constraints_record))
        for j, min_subset_constraints in enumerate(min_subset_constraints_record):
            BEC_lengths[j] = BEC_helpers.calculate_BEC_length(min_subset_constraints, weights, step_cost_flag)[0]

        with open('models/' + data_loc + '/raw_BEC_lengths.pickle', 'wb') as f:
            pickle.dump(BEC_lengths, f)

    kmeans = KMeans(n_clusters=6).fit(BEC_lengths.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_

    # ms = MeanShift().fit(BEC_lengths.reshape(-1, 1))
    # cluster_centers = ms.cluster_centers_

    # db = DBSCAN().fit(BEC_lengths.reshape(-1, 1))
    # for idx in db.core_sample_indices_:
    #     print(BEC_lengths[idx])
    # print(db.components_)

    # find the boundaries of the kmeans cluster
    domain = np.linspace(0, 3.5, 351)
    predicted = kmeans.predict(domain.reshape(-1, 1))
    print(min(domain[predicted == 0]))
    print(max(domain[predicted == 0]))
    print(min(domain[predicted == 1]))
    print(max(domain[predicted == 1]))
    print(min(domain[predicted == 2]))
    print(max(domain[predicted == 2]))
    print(min(domain[predicted == 3]))
    print(max(domain[predicted == 3]))
    print(min(domain[predicted == 4]))
    print(max(domain[predicted == 4]))
    print(min(domain[predicted == 5]))
    print(max(domain[predicted == 5]))

    # figure generated for ICRA paper
    plt.figure(figsize=(8, 3))
    plt.hist(x=BEC_lengths, bins=200, alpha=0.7, color='tab:blue')
    plt.grid(axis='y', alpha=0.75)
    plt.plot(cluster_centers, np.zeros(cluster_centers.shape), 'o', color='tab:red')
    plt.ylim(-60, plt.ylim()[1]) # to show all of BEC cluster centers
    plt.xlabel('BEC Area')
    plt.ylabel('Frequency')
    plt.title('Histogram of Demonstration BEC Areas')
    plt.tight_layout()
    # plt.savefig('augmented_taxi_auto.png', dpi=200, transparent=True)

    # plt.clf()
    #
    # plt.hist(x=BEC_lengths, bins=20, alpha=0.7)
    # plt.plot(cluster_centers, np.zeros(cluster_centers.shape), 'ro')
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('BEC Length')
    # plt.ylabel('Count')
    # plt.title('Augmented Taxi | Auto Bin | Total count: ' + str(len(BEC_lengths)))
    # # plt.title('Two Goal | Auto Bin | Total count: ' + str(len(BEC_lengths)))
    # # plt.title('Skateboard | Auto Bin | Total count: ' + str(len(BEC_lengths)))
    # plt.tight_layout()
    #
    # plt.savefig('augmented_taxi_20.png', dpi=200)
    # # plt.savefig('two_goal_20.png', dpi=200)
    # # plt.savefig('skateboard_20.png', dpi=200)

def extract_test_demonstrations(data_loc):
    '''
    Extract the desired test demonstrations from a larger collection of test demonstrations
    :param data_loc:
    :return:
    '''
    test_difficulty = 'high'
    data_loc_pull = data_loc
    data_loc_push = data_loc + '/testing/test_' + test_difficulty

    with open('models/' + data_loc_pull + '/test_environments_' + test_difficulty + '.pickle', 'rb') as f:
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(f)

    # selected test demonstrations
    if data_loc == 'augmented_taxi2':
        if test_difficulty == 'low':
            test_idxs = [1, 5, 13, 14, 18, 9]
        elif test_difficulty == 'medium':
            test_idxs = [5, 1, 26, 8, 2, 18]
        else:
            test_idxs = [0, 20, 8, 16, 1, 11]

    elif data_loc == 'colored_tiles':
        if test_difficulty == 'low':
            test_idxs = [1, 4, 8, 11, 0, 7]
        elif test_difficulty == 'medium':
            test_idxs = [1, 6, 2, 7, 9, 14]
        else:
            test_idxs = [0, 7, 14, 20, 18, 15]
    else:
        if test_difficulty == 'low':
            test_idxs = [8, 10, 9, 11, 12, 14]
        elif test_difficulty == 'medium':
            test_idxs = [0, 3, 5, 1, 11, 23]
        else:
            test_idxs = [6, 9, 3, 7, 8, 29]

    test_wt_vi_traj_tuples_subset = []
    test_BEC_lengths_subset = []
    test_BEC_constraints_subset = []
    test_selected_env_traj_tracers_subset = []

    # debugging code
    # for j, test_wt_vi_traj_tuple in enumerate(test_wt_vi_traj_tuples):
    #     print('Visualizing test environment {} with BEC length of {}'.format(j, test_BEC_lengths[j]))
    #     vi_candidate = test_wt_vi_traj_tuple[1]
    #     mdp = vi_candidate.mdp
    #     # mdp.weights.dot(mdp.accumulate_reward_features(test_wt_vi_traj_tuple[2]).T)

    for idx in test_idxs:
        test_wt_vi_traj_tuples_subset.append(test_wt_vi_traj_tuples[idx])
        test_BEC_lengths_subset.append(test_BEC_lengths[idx])
        test_BEC_constraints_subset.append(test_BEC_constraints[idx])
        test_selected_env_traj_tracers_subset.append(selected_env_traj_tracers[idx])

    with open('models/' + data_loc_push + '/test_environments.pickle', 'wb') as f:
        pickle.dump((test_wt_vi_traj_tuples_subset, test_BEC_lengths_subset, test_BEC_constraints_subset, test_selected_env_traj_tracers_subset), f)


def combine_summaries(data_loc):
    '''
    Simply combine three numbered summary files into one, nonnumbered master summary file
    '''
    with open('models/' + data_loc + '/BEC_summary_baseline.pickle', 'rb') as f:
        BEC_summary1 = pickle.load(f)
    with open('models/' + data_loc + '/BEC_summary_counterfactual_only.pickle', 'rb') as f:
        BEC_summary2 = pickle.load(f)
    with open('models/' + data_loc + '/BEC_summary_feature_only.pickle', 'rb') as f:
        BEC_summary3 = pickle.load(f)
    with open('models/' + data_loc + '/BEC_summary_proposed.pickle', 'rb') as f:
        BEC_summary4 = pickle.load(f)

    BEC_summary = []
    BEC_summary.extend(BEC_summary1)
    BEC_summary.extend(BEC_summary2)
    BEC_summary.extend(BEC_summary3)
    BEC_summary.extend(BEC_summary4)

    with open('models/' + data_loc + '/BEC_summary_combined.pickle', 'wb') as f:
        pickle.dump(BEC_summary, f)

def check_training_testing_overlap():
    '''
    See if any of the training demonstrations overlaps with any of the testing demonstrations (if there's overlap, it
    will print 'True')
    '''
    data_locs = ['augmented_taxi2', 'colored_tiles', 'skateboard2']
    test_difficulties = ['low', 'medium', 'high']

    for data_loc in data_locs:
        print(data_loc)

        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            summary = pickle.load(f)

        for test_difficulty in test_difficulties:
            data_loc_test = 'models/' + data_loc + '/testing/test_' + test_difficulty + '/test_environments.pickle'

            with open(data_loc_test, 'rb') as f:
                test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = pickle.load(f)

            for j, test_tuple in enumerate(test_wt_vi_traj_tuples):
                if policy_summarization_helpers._in_summary(test_tuple[1].mdp, summary, test_tuple[2][0][0]):
                    print('Overlap, Test difficulty: {}, Test #: {}'.format(test_difficulty, j))

def create_testing_dictionaries(test_env_dict, mapping):
    '''
    Translate the selected testing pickle files into dictionaries so that they can be uploaded and used in the web user study
    The tests can be generated using these dictionaries.
    '''

    for data_loc in mapping.keys():
        for test_difficulty in mapping[data_loc].keys():
            data_loc_test = 'models/' + data_loc + '/testing/test_' + test_difficulty + '/test_environments.pickle'

            with open(data_loc_test, 'rb') as f:
                test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(f)

            pairs = mapping[data_loc][test_difficulty]

            test_difficulty_dicts = []

            for pair in pairs:
                pair_mdp_dict = []
                for element in pair:
                    test_wt_vi_traj_tuple = test_wt_vi_traj_tuples[element]
                    vi = test_wt_vi_traj_tuple[1]
                    mdp = test_wt_vi_traj_tuple[1].mdp
                    optimal_traj = test_wt_vi_traj_tuple[2]
                    test_mdp_dict = test_wt_vi_traj_tuple[3]

                    test_mdp_dict = extract_mdp_dict(vi, mdp, optimal_traj, test_mdp_dict, data_loc, element=element, test_difficulty=test_difficulty)
                    pair_mdp_dict.append(test_mdp_dict)

                test_difficulty_dicts.append(pair_mdp_dict)

            test_env_dict[data_loc][test_difficulty] = test_difficulty_dicts

    # save the testing mdp information as a json
    with open('data.json', 'w') as f:
        json.dump(test_env_dict, f)

def print_training_summary_lengths():
    data_locs = ['augmented_taxi', 'two_goal', 'skateboard']
    folders = ['k0', 'k1', 'k3', 'k4_k5', 'k6', 'k7_k8_k2']

    n_variable_nonvariable_videos_dict = {
        'augmented_taxi': {
            'k0': [2, 0],
            'k1': [2, 0],
            'k3': [3, 2],
            'k4_k5': [3, 2],
            'k6': [3, 2],
            'k7_k8_k2': [3, 2],
        },
        'two_goal': {
            'k0': [3, 0],
            'k1': [3, 0],
            'k3': [2, 3],
            'k4_k5': [2, 3],
            'k6': [2, 3],
            'k7_k8_k2': [2, 3],
        },
        'skateboard': {
            'k0': [2, 0],
            'k1': [2, 0],
            'k3': [3, 2],
            'k4_k5': [3, 2],
            'k6': [3, 2],
            'k7_k8_k2': [3, 2],
        }
    }


    training_traj_lengths = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for data_loc in data_locs:
        for folder_num, folder in enumerate(folders):

            with open('models/' + data_loc + '/training/' + folder + '/BEC_summary.pickle', 'rb') as f:
                summary_collection = pickle.load(f)

            # number of redundant videos
            n_variable_videos = n_variable_nonvariable_videos_dict[data_loc][folder][0]

            # number of nonshared videos
            n_nonvariable_videos = n_variable_nonvariable_videos_dict[data_loc][folder][1]

            between_loop_count = 1
            within_loop_count = 1

            total_count = 0
            while total_count < len(summary_collection):
                traj_length = len(summary_collection[total_count][1])
                key = str(within_loop_count)

                if within_loop_count <= n_variable_videos:
                    if between_loop_count == 1:
                        key += 'a'
                    elif between_loop_count == 2:
                        key += 'b'
                    elif between_loop_count == 3:
                        key += 'c'
                    else:
                        raise RuntimeError("Unexpected behavior.")

                training_traj_lengths[data_loc][folder][key] = traj_length

                within_loop_count += 1
                if within_loop_count > (n_variable_videos + n_nonvariable_videos):
                    within_loop_count = 1
                    between_loop_count += 1

                total_count += 1

    print(training_traj_lengths)

    with open('training_traj_lengths.json', 'w') as f:
        json.dump(training_traj_lengths, f)

def obtain_outlier_human_scores(test_env_dict):
    with open('dfs_full.pickle', 'rb') as f:
        df_training, df_testing, df_testing_sandbox, df_training_survey, df_post_survey = pickle.load(f)

    df_testing_filtered = df_testing.copy()
    flagged_ids = ['5efb33ad6ad15505fd008366', '5fbfe145e52a44000a9c2966', '60f6f802c0ede08f7cf69720', '61423a70286bdb2a2d226fa7']
    for id in flagged_ids:
        df_testing_filtered = df_testing_filtered[
            df_testing_filtered.uniqueid != id]

    test_demos_raw_human_scores = {
        'augmented_taxi2':
            {
                'low': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
                'medium': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
                'high': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
            },
        'colored_tiles':
            {
                'low': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
                'medium': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
                'high': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
            },
        'skateboard2':
            {
                'low': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
                'medium': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
                'high': {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []},
            }
    }

    test_demos_mean_human_scores = {
        'augmented_taxi2':
            {
                'low': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'medium': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'high': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
            },
        'colored_tiles':
            {
                'low': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'medium': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'high': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
            },
        'skateboard2':
            {
                'low': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'medium': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'high': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
            }
    }

    test_demos_std_human_scores = {
        'augmented_taxi2':
            {
                'low': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'medium': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'high': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
            },
        'colored_tiles':
            {
                'low': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'medium': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'high': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
            },
        'skateboard2':
            {
                'low': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'medium': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                'high': {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
            }
    }

    for i in df_testing.index:
        domain = df_testing.at[i, 'domain']
        moves_list = df_testing.moves[i][1:-1].replace('\', ', '').split('\'')[1:-1]
        test_difficulty = df_testing.test_mdp[i]['test_difficulty']
        tag = df_testing.test_mdp[i]['tag']

        mdp = test_env_dict[domain][test_difficulty][tag][1].mdp

        mdp.reset()
        trajectory = []
        cur_state = mdp.get_init_state()

        for idx in range(len(moves_list)):
            # assumes that the user study only allows actions that change the state of the MDP
            reward, next_state = mdp.execute_agent_action(moves_list[idx])
            trajectory.append((cur_state, moves_list[idx], next_state))

            # deepcopy occurs within transition function
            cur_state = next_state

        human_reward = mdp.weights.dot(mdp.accumulate_reward_features(trajectory).T)

        # test_demos_raw_human_scores[domain][test_difficulty][str(tag)].append(human_reward[0][0])
        # not using the mean and std given by outliers 2.8 std dev away
        if df_testing.at[i, 'uniqueid'] not in flagged_ids:
            test_demos_raw_human_scores[domain][test_difficulty][str(tag)].append(human_reward[0][0])

        df_testing.at[i, 'scaled_human_reward'] = human_reward[0][0]

    for domain in test_demos_raw_human_scores.keys():
        for test_difficulty in test_demos_raw_human_scores[domain].keys():
            for tag in test_demos_raw_human_scores[domain][test_difficulty].keys():
                test_demos_mean_human_scores[domain][test_difficulty][tag] = np.mean(test_demos_raw_human_scores[domain][test_difficulty][tag])
                test_demos_std_human_scores[domain][test_difficulty][tag] = np.std(test_demos_raw_human_scores[domain][test_difficulty][tag])


    outlier_uniqueids = {}
    outlier_conditions = {}

    sigma = 3

    for i in df_testing.index:
        domain = df_testing.at[i, 'domain']
        test_difficulty = df_testing.test_mdp[i]['test_difficulty']
        tag = df_testing.test_mdp[i]['tag']

        if df_testing.at[i, 'scaled_human_reward'] < test_demos_mean_human_scores[domain][test_difficulty][
            str(tag)] - sigma * test_demos_std_human_scores[domain][test_difficulty][str(tag)]:
            if not df_testing.at[i, 'uniqueid'] in outlier_uniqueids.keys():
                outlier_uniqueids[df_testing.at[i, 'uniqueid']] = 1
                outlier_conditions[df_testing.at[i, 'uniqueid']] = df_testing.at[i, 'condition']
            else:
                outlier_uniqueids[df_testing.at[i, 'uniqueid']] += 1

    outlier_count = np.array(list(outlier_uniqueids.values())).mean() + sigma * np.array(
        list(outlier_uniqueids.values())).std()

    for key in outlier_uniqueids.keys():
        if outlier_uniqueids[key] > outlier_count:
            print(key)
            print(outlier_uniqueids[key])

    print('==============')


def obtain_test_demos_low_human_scores(df_testing, test_env_dict):
    # if I want to try and scale by the worst possible human score (as the lowerbound) and the best possible score (as the upperbound)
    test_demos_low_human_scores = {
        'augmented_taxi2':
            {
                'low': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                        '4': float('inf'), '5': float('inf')},
                'medium': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                           '4': float('inf'), '5': float('inf')},
                'high': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                         '4': float('inf'), '5': float('inf')},
            },
        'colored_tiles':
            {
                'low': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                        '4': float('inf'), '5': float('inf')},
                'medium': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                           '4': float('inf'), '5': float('inf')},
                'high': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                         '4': float('inf'), '5': float('inf')},
            },
        'skateboard2':
            {
                'low': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                        '4': float('inf'), '5': float('inf')},
                'medium': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                           '4': float('inf'), '5': float('inf')},
                'high': {'0': float('inf'), '1': float('inf'), '2': float('inf'), '3': float('inf'),
                         '4': float('inf'), '5': float('inf')},
            }
    }

    for i in df_testing.index:
        domain = df_testing.at[i, 'domain']
        moves_list = df_testing.moves[i][1:-1].replace('\', ', '').split('\'')[1:-1]
        test_difficulty = df_testing.test_mdp[i]['test_difficulty']
        tag = df_testing.test_mdp[i]['tag']

        mdp = test_env_dict[domain][test_difficulty][tag][1].mdp

        mdp.reset()
        trajectory = []
        cur_state = mdp.get_init_state()

        for idx in range(len(moves_list)):
            # assumes that the user study only allows actions that change the state of the MDP
            reward, next_state = mdp.execute_agent_action(moves_list[idx])
            trajectory.append((cur_state, moves_list[idx], next_state))

            # deepcopy occurs within transition function
            cur_state = next_state

        human_reward = mdp.weights.dot(mdp.accumulate_reward_features(trajectory).T)

        if human_reward < test_demos_low_human_scores[domain][test_difficulty][str(tag)]:
            test_demos_low_human_scores[domain][test_difficulty][str(tag)] = human_reward[0][0]

    return test_demos_low_human_scores

def process_human_scores(test_env_dict, type='binary'):
    with open('dfs.pickle', 'rb') as f:
        df_training, df_testing, df_testing_sandbox, df_training_survey, df_post_survey = pickle.load(f)

    if type == 'scaled' or type == 'scale-truncated':
        test_demos_low_human_scores = obtain_test_demos_low_human_scores(df_testing, test_env_dict)

    for i in df_testing.index:
        domain = df_testing.at[i, 'domain']
        moves_list = df_testing.moves[i][1:-1].replace('\', ', '').split('\'')[1:-1]
        test_difficulty = df_testing.test_mdp[i]['test_difficulty']
        tag = df_testing.test_mdp[i]['tag']

        mdp = test_env_dict[domain][test_difficulty][tag][1].mdp

        mdp.reset()
        trajectory = []
        cur_state = mdp.get_init_state()

        for idx in range(len(moves_list)):
            # assumes that the user study only allows actions that change the state of the MDP
            reward, next_state = mdp.execute_agent_action(moves_list[idx])
            trajectory.append((cur_state, moves_list[idx], next_state))

            # deepcopy occurs within transition function
            cur_state = next_state

        human_reward = mdp.weights.dot(mdp.accumulate_reward_features(trajectory).T)

        if type == 'binary':
            # record binary reward
            # if human_reward == df_testing.test_mdp[i]['opt_traj_reward']:
            #     if test_difficulty == 'high':
            #         df_testing.at[i, 'scaled_human_reward'] = 3
            #     elif test_difficulty == 'medium':
            #         df_testing.at[i, 'scaled_human_reward'] = 2
            #     else:
            #         df_testing.at[i, 'scaled_human_reward'] = 1
            # else:
            #     df_testing.at[i, 'scaled_human_reward'] = 0
            if human_reward == df_testing.test_mdp[i]['opt_traj_reward']:
                df_testing.at[i, 'scaled_human_reward'] = 1
            else:
                df_testing.at[i, 'scaled_human_reward'] = 0
        elif type == 'raw':
            # record raw data
            df_testing.at[i, 'scaled_human_reward'] = human_reward[0][0]
            # print(human_reward)
            # print(df_testing.test_mdp[i]['opt_traj_reward'])
            # print('\n')
        elif type == 'scaled':
            if df_testing.test_mdp[i]['opt_traj_reward'] == test_demos_low_human_scores[domain][test_difficulty][str(tag)]:
                df_testing.at[i, 'scaled_human_reward'] = 1.0
            else:
                df_testing.at[i, 'scaled_human_reward'] = (human_reward[0][0] - test_demos_low_human_scores[domain][test_difficulty][str(tag)]) / (df_testing.test_mdp[i]['opt_traj_reward'] - test_demos_low_human_scores[domain][test_difficulty][str(tag)])
        elif type == 'scale-trunacted':
            # record scale-truncated binary data
            test_demos_low_human_scores = obtain_test_demos_low_human_scores(df_testing, test_env_dict)
            if df_testing.test_mdp[i]['opt_traj_reward'] == test_demos_low_human_scores[domain][test_difficulty][str(tag)]:
                df_testing.at[i, 'scaled_human_reward'] = 1.0
            elif (human_reward[0][0] - test_demos_low_human_scores[domain][test_difficulty][str(tag)]) / (df_testing.test_mdp[i]['opt_traj_reward'] - test_demos_low_human_scores[domain][test_difficulty][str(tag)]) >= 0.95:
                df_testing.at[i, 'scaled_human_reward'] = 1.0
            else:
                df_testing.at[i, 'scaled_human_reward'] = 0.0
                # df_testing.at[i, 'scaled_human_reward'] = (human_reward[0][0] - test_demos_low_human_scores[domain][test_difficulty][str(tag)]) / (
                #             df_testing.test_mdp[i]['opt_traj_reward'] - test_demos_low_human_scores[dCopy of Study feedbackomain][test_difficulty][str(tag)])
        else:
            raise Exception("Unknown score processing type.")

        # record corresponding test difficulty
        if test_difficulty == 'low':
            df_testing.at[i, 'test_difficulty'] = 0
        elif test_difficulty == 'medium':
            df_testing.at[i, 'test_difficulty'] = 1
        else:
            df_testing.at[i, 'test_difficulty'] = 2

    # with open('dfs_processed.pickle', 'wb') as f:
    #     pickle.dump((df_training, df_testing, df_testing_sandbox, df_training_survey, df_post_survey), f)

def obtain_human_trajectories(test_env_dict):
    '''
    Extract human trajectories for each test difficulty level from pandas dfs and store them in a dictionary
    '''

    with open('dfs.pickle', 'rb') as f:
        df_training, df_testing, df_testing_sandbox, df_training_survey, df_post_survey = pickle.load(f)

    trajectory_dict = {
        'augmented_taxi2':
            {
                'low':
                    {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                'medium': {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                },
                'high': {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                },
            },
        'colored_tiles':
            {
                'low':
                    {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                'medium': {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                },
                'high': {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                },
            },
        'skateboard2':
            {
                'low':
                    {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                'medium': {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                },
                'high': {
                    '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                },
            }
    }

    # remove ids that were flagged for IROS22 study
    flagged_ids = ['5efb33ad6ad15505fd008366',
                            '5fbfe145e52a44000a9c2966',
                            '60f6f802c0ede08f7cf69720', '61423a70286bdb2a2d226fa7']

    for id in flagged_ids:
        df_testing = df_testing[df_testing.uniqueid != id]

    for i in df_testing.index:
        domain = df_testing.at[i, 'domain']
        moves_list = df_testing.moves[i][1:-1].replace('\', ', '').split('\'')[1:-1]
        test_difficulty = df_testing.test_mdp[i]['test_difficulty']
        tag = df_testing.test_mdp[i]['tag']

        mdp = test_env_dict[domain][test_difficulty][tag][1].mdp

        # get human trajectory
        mdp.reset()
        trajectory = []
        cur_state = mdp.get_init_state()

        for idx in range(len(moves_list)):
            # assumes that the user study only allows actions that change the state of the MDP
            reward, next_state = mdp.execute_agent_action(moves_list[idx])
            trajectory.append((cur_state, moves_list[idx], next_state))

            # deepcopy occurs within transition function
            cur_state = next_state

        # get optimal trajectory
        mdp.reset()
        opt_trajectory = []
        opt_moves_list = df_testing.test_mdp[i]['opt_actions']
        cur_state = mdp.get_init_state()

        for idx in range(len(opt_moves_list)):
            reward, next_state = mdp.execute_agent_action(opt_moves_list[idx])
            opt_trajectory.append((cur_state, opt_moves_list[idx], next_state))

            # deepcopy occurs within transition function
            cur_state = next_state

        human_reward = mdp.weights.dot(mdp.accumulate_reward_features(trajectory).T)

        trajectory_dict[domain][test_difficulty][str(tag)].append((mdp, trajectory, df_testing.test_mdp[i]['opt_traj_reward'], human_reward, opt_trajectory))

    with open('human_trajectories.pickle', 'wb') as f:
        pickle.dump(trajectory_dict, f)

def view_trajectories(visualize=False):
    '''
    View human trajectories (obtained through user studies) to get a general sense of human understanding and performance
    '''

    try:
        with open('filtered_human_responses.pickle', 'rb') as f:
            filtered_human_traj_dict, filtered_mdp_dict, filtered_count_dict, filtered_opt_reward_dict, filtered_human_reward_dict, filtered_opt_traj_dict = pickle.load(
                f)
    except:
        with open('human_trajectories.pickle', 'rb') as f:
            trajectory_dict = pickle.load(f)

        filtered_human_traj_dict = {
            'augmented_taxi2':
                {
                    'low':
                        {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'colored_tiles':
                {
                    'low':
                        {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'skateboard2':
                {
                    'low':
                        {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                }
        }
        filtered_count_dict = {
            'augmented_taxi2':
                {
                    'low':
                        {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'colored_tiles':
                {
                    'low':
                        {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'skateboard2':
                {
                    'low':
                        {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                }
        }
        filtered_mdp_dict = {
            'augmented_taxi2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'colored_tiles':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'skateboard2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                }
        }
        filtered_opt_reward_dict = {
            'augmented_taxi2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'colored_tiles':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'skateboard2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                }
        }
        filtered_human_reward_dict = {
            'augmented_taxi2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'colored_tiles':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'skateboard2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                }
        }
        filtered_opt_traj_dict = {
            'augmented_taxi2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'colored_tiles':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                },
            'skateboard2':
                {
                    'low':
                        {
                            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                        },
                    'medium': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                    'high': {
                        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [],
                    },
                }
        }

        # account for duplicate test responses
        for domain in trajectory_dict.keys():
            # print(domain)
            for difficulty in trajectory_dict[domain].keys():
                # print(difficulty)
                for tag in trajectory_dict[domain][difficulty].keys():
                    # print(tag)
                    for traj in trajectory_dict[domain][difficulty][tag]:
                        # print(traj[3] / traj[2]) # human reward / optimal reward
                        if traj[1] not in filtered_human_traj_dict[domain][difficulty][tag]:
                            filtered_human_traj_dict[domain][difficulty][tag].append(traj[1])
                            filtered_mdp_dict[domain][difficulty][tag].append(traj[0])
                            filtered_count_dict[domain][difficulty][tag].append(1)
                            filtered_opt_reward_dict[domain][difficulty][tag].append(traj[2])
                            filtered_human_reward_dict[domain][difficulty][tag].append(traj[3])
                            filtered_opt_traj_dict[domain][difficulty][tag].append(traj[4])
                            # traj[0].visualize_trajectory(traj[1])
                        else:
                            for j, stored_traj in enumerate(filtered_human_traj_dict[domain][difficulty][tag]):
                                if traj[1] == stored_traj:
                                    filtered_count_dict[domain][difficulty][tag][j] += 1
                                    break

    if visualize:
        # go through potentially duplicate trajectories and visualize them
        for domain in filtered_human_traj_dict.keys():
            print('domain: {}'.format(domain))
            for difficulty in filtered_human_traj_dict[domain].keys():
                print('difficulty: {}'.format(difficulty))
                for tag in filtered_human_traj_dict[domain][difficulty].keys():
                    print('tag: {}'.format(tag))
                    print('total number of trajectories: {}'.format(sum(filtered_count_dict[domain][difficulty][tag])))
                    total_num_unique_traj = len(filtered_human_traj_dict[domain][difficulty][tag])
                    for i, traj in enumerate(filtered_human_traj_dict[domain][difficulty][tag]):
                        print('trajectory {}/{}'.format(i + 1, total_num_unique_traj))
                        print('duplicate # of this trajectory: {}'.format(filtered_count_dict[domain][difficulty][tag][i]))
                        # print('opt reward vs human reward: {} vs {}'.format(filtered_opt_reward_dict[domain][difficulty][tag][i], filtered_human_reward_dict[domain][difficulty][tag][i]))
                        print('optimal?: {}'.format(
                            filtered_opt_reward_dict[domain][difficulty][tag][i] ==
                            filtered_human_reward_dict[domain][difficulty][tag][i][0][0]))
                        filtered_mdp_dict[domain][difficulty][tag][i].visualize_trajectory(filtered_human_traj_dict[domain][difficulty][tag][i])

    with open('filtered_human_responses.pickle', 'wb') as f:
        pickle.dump((filtered_human_traj_dict, filtered_mdp_dict, filtered_count_dict, filtered_opt_reward_dict, filtered_human_reward_dict, filtered_opt_traj_dict), f)

def consolidate_counterfactual_constraints():
    '''
    Consolidate counterfactual constraints into a single file (testing if this is quicker than loading individually from multiple files)
    '''
    data_loc = 'augmented_taxi2'
    from multiprocessing import Pool
    from tqdm import tqdm

    pool = Pool(min(params.n_cpu, 1))

    def load_constraints(args):
        model_idx = args
        precomputed_PF_constraints = []

        for env_idx in range(64):
            with open('models/' + data_loc + '/counterfactual_data_precomputed/model' + str(model_idx) + '/cf_data_env'
                      + str(env_idx).zfill(5) + '.pickle', 'rb') as f:
                constraints_env = pickle.load(f)
                precomputed_PF_constraints.append(constraints_env)

        return precomputed_PF_constraints

    args = [(i) for i in range(2500)]
    precomputed_PF_constraints = list(tqdm(pool.imap(load_constraints, args), total=len(args)))
    pool.close()
    pool.join()

    with open('models/' + data_loc + '/' + 'precomputed_PF_constraints.pickle', 'wb') as f:
        pickle.dump(precomputed_PF_constraints, f)


def save_user_study_json(mapping):
    '''
    Save the demonstrations, diagnostic tests, and final tests to be used in the user study as a json
    '''

    import random
    from policy_summarization import BEC

    user_study_dict = {
            'augmented_taxi2': {"demo": {}, "diagnostic test": {}, "final test": {}},
            'colored_tiles': {"demo": {}, "diagnostic test": {}, "final test": {}},
            'skateboard2': {"demo": {}, "diagnostic test": {}, "final test": {}}
    }

    for data_loc in mapping.keys():
        print(data_loc)

        # a) save the teaching demonstrations
        demo_idx = 0
        diagnostic_test_idx = 0
        with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)

        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary, visited_env_traj_idxs, particles = pickle.load(f)

        for unit_idx, unit in enumerate(BEC_summary):
            unit_constraints = []
            running_variable_filter = unit[0][4]

            # save the information for each demonstration that is part of this unit into a dictionary
            for subunit_idx, subunit in enumerate(unit):
                print("Unit {}/{}, demo {}/{}:".format(unit_idx + 1, len(BEC_summary), subunit_idx + 1,
                                                       len(unit)))
                demo_constraints = subunit[3]
                unit_constraints.extend(demo_constraints)

                best_env_idx, best_traj_idx = subunit[2]
                filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
                with open(filename, 'rb') as f:
                    wt_vi_traj_env = pickle.load(f)
                vi = wt_vi_traj_env[0][1]
                mdp_dict = wt_vi_traj_env[0][3]

                user_study_dict[data_loc]["demo"][str(demo_idx)] = extract_mdp_dict(vi, subunit[0], subunit[1], mdp_dict, data_loc, env_traj_idxs=(best_env_idx, best_traj_idx), variable_filter=running_variable_filter, constraints=demo_constraints)
                demo_idx += 1

            # b) save the diagnostic tests
            # obtain the constraints conveyed by the unit's demonstrations
            min_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints)
            random.shuffle(min_constraints) # shuffle the order of the constraints so that it's not always the same
            # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
            preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(data_loc, unit, visited_env_traj_idxs, min_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter, mdp_features_record)

            for test in preliminary_tests:
                best_env_idx, best_traj_idx = test[2]
                filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
                with open(filename, 'rb') as f:
                    wt_vi_traj_env = pickle.load(f)
                vi = wt_vi_traj_env[0][1]
                mdp_dict = wt_vi_traj_env[0][3]

                test_mdp = test[0]
                opt_traj = test[1]
                test_constraints = test[3]

                user_study_dict[data_loc]["diagnostic test"][str(diagnostic_test_idx)] = extract_mdp_dict(vi, test_mdp, opt_traj, mdp_dict, data_loc, element=-3, env_traj_idxs=(best_env_idx, best_traj_idx), variable_filter=running_variable_filter, constraints=test_constraints)
                diagnostic_test_idx += 1

        # c) save the final, held-out set of tests
        for test_difficulty in mapping[data_loc].keys():
            data_loc_test = 'models/' + data_loc + '/testing/test_' + test_difficulty + '/test_environments.pickle'

            with open(data_loc_test, 'rb') as f:
                test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(f)

            pairs = mapping[data_loc][test_difficulty]

            test_difficulty_dicts = []

            for pair in pairs:
                pair_mdp_dict = []
                for element in pair:
                    test_wt_vi_traj_tuple = test_wt_vi_traj_tuples[element]
                    vi = test_wt_vi_traj_tuple[1]
                    mdp = test_wt_vi_traj_tuple[1].mdp
                    optimal_traj = test_wt_vi_traj_tuple[2]
                    test_mdp_dict = test_wt_vi_traj_tuple[3]

                    test_mdp_dict = extract_mdp_dict(vi, mdp, optimal_traj, test_mdp_dict, data_loc, element=element, test_difficulty=test_difficulty)
                    pair_mdp_dict.append(test_mdp_dict)

                test_difficulty_dicts.append(pair_mdp_dict)

            user_study_dict[data_loc]["final test"][test_difficulty] = test_difficulty_dicts

    with open('models/user_study_dict.json', 'w') as f:
        json.dump(user_study_dict, f)

if __name__ == "__main__":
    os.chdir('../../..')

    data_loc = params.data_loc['BEC']

    # specify which potential pairs of demonstrations within each difficulty to use based on semantics (e.g. pair one
    # high difficulty that detours to the right with another high difficulty one detours to the left)
    mapping = {
        'augmented_taxi2':
            {
                'low': [[0, 1], [2, 3], [4, 5]],
                'medium': [[0, 1], [2, 3], [4, 5]],
                'high': [[0, 1], [2, 3], [4, 5]],
            },
        'colored_tiles':
            {
                'low': [[0, 1], [2, 3], [4, 5]],
                'medium': [[0, 1], [2, 3], [4, 5]],
                'high': [[0, 1], [2, 3], [4, 5]],
            },
        'skateboard2':
            {
                'low': [[0, 1], [2, 3], [4, 5]],
                'medium': [[0, 1], [2, 3], [4, 5]],
                'high': [[0, 1], [2, 3], [4, 5]],
            }
    }

    try:
        with open('test_env_dict.pickle', 'rb') as f:
            test_env_dict = pickle.load(f)
    except:
        test_env_dict = {
            'augmented_taxi2': {},
            'colored_tiles': {},
            'skateboard2': {}
        }
        for data_loc in mapping.keys():
            for test_difficulty in mapping[data_loc].keys():
                data_loc_test = 'models/' + data_loc + '/testing/test_' + test_difficulty + '/test_environments.pickle'

                with open(data_loc_test, 'rb') as f:
                    test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(
                        f)
                    test_env_dict[data_loc][test_difficulty] = test_wt_vi_traj_tuples

        with open('test_env_dict.pickle', 'wb') as f:
            pickle.dump(test_env_dict, f)

    # combine_summaries(data_loc)
    # extract_test_demonstrations(data_loc)
    # plot_BEC_histogram(data_loc, params.weights['val'], params.step_cost_flag)
    # check_training_testing_overlap()
    # create_testing_dictionaries(test_env_dict, mapping)
    # print_training_summary_lengths()
    # process_human_scores(test_env_dict, type='binary')  # binary, raw, scaled, or scale-truncated
    # obtain_outlier_human_scores(test_env_dict)

    # going through IROS human participant trajectories
    # obtain_human_trajectories(test_env_dict)
    # view_trajectories(visualize=False)


    # preparing for closed-loop teaching user study
    save_user_study_json(mapping)