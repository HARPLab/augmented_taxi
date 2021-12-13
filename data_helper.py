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

                    test_mdp_dict['opt_actions'] = [sas[1] for sas in optimal_traj]
                    test_mdp_dict['opt_traj_length'] = len(optimal_traj)
                    test_mdp_dict['opt_traj_reward'] = mdp.weights.dot(mdp.accumulate_reward_features(optimal_traj).T)[0][0]
                    test_mdp_dict['test_difficulty'] = test_difficulty
                    # to be able to trace the particular environment (0-5)
                    test_mdp_dict['tag'] = element

                    # also obtain all possible optimal trajectories
                    all_opt_trajs = mdp_helpers.rollout_policy_recursive(mdp, vi, optimal_traj[0][0], [])
                    # extract all of the actions
                    all_opt_actions = []
                    for opt_traj in all_opt_trajs:
                        all_opt_actions.append([sas[1] for sas in opt_traj])
                    test_mdp_dict['all_opt_actions'] = all_opt_actions

                    # delete unserializable numpy arrays that aren't necessary
                    try:
                        del test_mdp_dict['weights_lb']
                        del test_mdp_dict['weights_ub']
                        del test_mdp_dict['weights']
                    except:
                        pass

                    # print(test_mdp_dict)
                    # print(test_mdp_dict['env_code'])

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

def process_human_scores(test_env_dict, mapping):
    with open('dfs.pickle', 'rb') as f:
        df_training_exp1, df_training_exp2, df_testing, df_testing_sandbox, df_training_survey_exp1, df_training_survey_exp2, df_post_survey = pickle.load(f)

    for data_loc in mapping.keys():
        for test_difficulty in mapping[data_loc].keys():

            data_loc_test = 'models/' + data_loc + '/testing/test_' + test_difficulty + '/test_environments.pickle'

            with open(data_loc_test, 'rb') as f:
                test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = pickle.load(f)
                test_env_dict[data_loc][test_difficulty] = test_wt_vi_traj_tuples

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
            reward, next_state = mdp.execute_agent_action(moves_list[idx])
            trajectory.append((cur_state, moves_list[idx], next_state))

            # deepcopy occurs within transition function
            cur_state = next_state

        human_reward = mdp.weights.dot(mdp.accumulate_reward_features(trajectory).T)

        # record binary reward
        if human_reward == df_testing.test_mdp[i]['opt_traj_reward']:
            df_testing.at[i, 'scaled_human_reward'] = 1
        else:
            df_testing.at[i, 'scaled_human_reward'] = 0

        # record raw data
        # df_testing.at[i, 'scaled_human_reward'] = human_reward

        # # record scaled data
        # df_testing.at[i, 'scaled_human_reward'] = human_reward / df_testing.test_mdp[i]['opt_traj_reward']

        # # record truncated scaled data
        # scaled_reward = human_reward / df_testing.test_mdp[i]['opt_traj_reward']
        # if scaled_reward < 0:
        #     df_testing.at[i, 'scaled_human_reward'] = 0
        # else:
        #     df_testing.at[i, 'scaled_human_reward'] = scaled_reward

        # # record truncated binary reward
        # if human_reward / df_testing.test_mdp[i]['opt_traj_reward'] >= 0.95:
        #     df_testing.at[i, 'scaled_human_reward'] = 1
        # else:
        #     df_testing.at[i, 'scaled_human_reward'] = 0

        # record corresponding test difficulty
        if test_difficulty == 'low':
            df_testing.at[i, 'test_difficulty'] = 0
        elif test_difficulty == 'medium':
            df_testing.at[i, 'test_difficulty'] = 1
        else:
            df_testing.at[i, 'test_difficulty'] = 2

    with open('dfs_processed.pickle', 'wb') as f:
        pickle.dump((df_training_exp1, df_training_exp2, df_testing, df_testing_sandbox, df_training_survey_exp1, df_training_survey_exp2, df_post_survey), f)

if __name__ == "__main__":
    data_loc = params.data_loc['BEC']
    test_env_dict = {
        'augmented_taxi2': {},
        'colored_tiles': {},
        'skateboard2': {}
    }

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


    # combine_summaries(data_loc)
    # extract_test_demonstrations(data_loc)
    # plot_BEC_histogram(data_loc, params.weights['val'], params.step_cost_flag)
    # check_training_testing_overlap()
    create_testing_dictionaries(test_env_dict, mapping)
    # print_training_summary_lengths()
    # process_human_scores(test_env_dict, mapping)