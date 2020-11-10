import params
import dill as pickle
import numpy as np
import policy_summarization.BEC_helpers as BEC_helpers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from policy_summarization import policy_summarization_helpers
import json
from collections import defaultdict

'''
For managing data related to the user study 
'''

def plot_BEC_histogram(data_loc, weights, step_cost_flag):
    try:
        with open('models/' + data_loc + '/raw_BEC_lengths.pickle', 'rb') as f:
            BEC_lengths = pickle.load(f)
    except:
        with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
            min_subset_constraints_record, env_record, traj_record = pickle.load(f)

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
    plt.savefig('augmented_taxi_auto.png', dpi=200, transparent=True)

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
    data_loc_pull = data_loc + '/old_test_demos/second_batch/test_full/test_' + test_difficulty
    data_loc_push = data_loc + '/old_test_demos/second_batch/test_subset/test_' + test_difficulty

    with open('models/' + data_loc_pull + '/test_environments.pickle', 'rb') as f:
        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = pickle.load(f)

    # selected test demonstrations
    # augment_taxi_low
    # test_idxs = [0, 1, 2, 6, 13, 17]
    # augment_taxi_medium
    # test_idxs = [11, 29, 40, 47, 65, 95]
    # augment_taxi_high
    # test_idxs = [12, 22, 28, 50, 51, 63]

    # two goal low
    # test_idxs = [0, 1, 11, 30, 46, 61]
    # two goal medium
    # test_idxs = [1, 9, 15, 16, 24, 27]
    # two goal high
    # test_idxs = [5, 9, 16, 19, 25, 39]

    # skateboard low
    # test_idxs = [1, 2, 3, 6, 10, 11]
    # skateboard medium
    # test_idxs = [6, 13, 17, 18, 19, 38]
    # skateboard high
    # test_idxs = [2, 4, 26, 29, 34, 39]

    test_wt_vi_traj_tuples_subset = []
    test_BEC_lengths_subset = []
    test_BEC_constraints_subset = []

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

    with open('models/' + data_loc_push + '/test_environments.pickle', 'wb') as f:
        pickle.dump((test_wt_vi_traj_tuples_subset, test_BEC_lengths_subset, test_BEC_constraints_subset), f)


def combine_summaries(data_loc):
    '''
    Simply combine three numbered summary files into one, nonnumbered master summary file
    '''
    # with open('models/' + data_loc + '/BEC_summary1.pickle', 'rb') as f:
    #     BEC_summary1 = pickle.load(f)
    # with open('models/' + data_loc + '/BEC_summary2.pickle', 'rb') as f:
    #     BEC_summary2 = pickle.load(f)
    # with open('models/' + data_loc + '/BEC_summary3.pickle', 'rb') as f:
    #     BEC_summary3 = pickle.load(f)

    with open('models/' + data_loc + '/training/k7_k8_k2/BEC_summary1.pickle', 'rb') as f:
        BEC_summary1_k7_k8_k2 = pickle.load(f)
    with open('models/' + data_loc + '/training/k7_k8_k2//BEC_summary2.pickle', 'rb') as f:
        BEC_summary2_k7_k8_k2 = pickle.load(f)
    with open('models/' + data_loc + '/training/k7_k8_k2//BEC_summary3.pickle', 'rb') as f:
        BEC_summary3_k7_k8_k2 = pickle.load(f)

    with open('models/' + data_loc + '/training/k4_k5/BEC_summary1.pickle', 'rb') as f:
        BEC_summary1_k4_k5 = pickle.load(f)
    with open('models/' + data_loc + '/training/k4_k5/BEC_summary2.pickle', 'rb') as f:
        BEC_summary2_k4_k5 = pickle.load(f)
    with open('models/' + data_loc + '/training/k4_k5/BEC_summary3.pickle', 'rb') as f:
        BEC_summary3_k4_k5 = pickle.load(f)

    with open('models/' + data_loc + '/training/k3/BEC_summary1.pickle', 'rb') as f:
        BEC_summary1_k3 = pickle.load(f)
    with open('models/' + data_loc + '/training/k3/BEC_summary2.pickle', 'rb') as f:
        BEC_summary2_k3 = pickle.load(f)
    with open('models/' + data_loc + '/training/k3/BEC_summary3.pickle', 'rb') as f:
        BEC_summary3_k3 = pickle.load(f)

    with open('models/' + data_loc + '/training/k6/BEC_summary1.pickle', 'rb') as f:
        BEC_summary1_k6 = pickle.load(f)
    with open('models/' + data_loc + '/training/k6/BEC_summary2.pickle', 'rb') as f:
        BEC_summary2_k6 = pickle.load(f)
    with open('models/' + data_loc + '/training/k6/BEC_summary3.pickle', 'rb') as f:
        BEC_summary3_k6 = pickle.load(f)

    with open('models/' + data_loc + '/training/k0/BEC_summary1.pickle', 'rb') as f:
        BEC_summary1_k0 = pickle.load(f)
    with open('models/' + data_loc + '/training/k0/BEC_summary2.pickle', 'rb') as f:
        BEC_summary2_k0 = pickle.load(f)
    with open('models/' + data_loc + '/training/k0/BEC_summary3.pickle', 'rb') as f:
        BEC_summary3_k0 = pickle.load(f)

    with open('models/' + data_loc + '/training/k1/BEC_summary1.pickle', 'rb') as f:
        BEC_summary1_k1 = pickle.load(f)
    with open('models/' + data_loc + '/training/k1/BEC_summary2.pickle', 'rb') as f:
        BEC_summary2_k1 = pickle.load(f)
    with open('models/' + data_loc + '/training/k1/BEC_summary3.pickle', 'rb') as f:
        BEC_summary3_k1 = pickle.load(f)


    BEC_summary = []
    # BEC_summary.extend(BEC_summary1)
    # BEC_summary.extend(BEC_summary2)
    # BEC_summary.extend(BEC_summary3)

    BEC_summary.extend(BEC_summary1_k7_k8_k2)
    BEC_summary.extend(BEC_summary1_k4_k5)
    BEC_summary.extend(BEC_summary1_k3)
    BEC_summary.extend(BEC_summary1_k6)
    BEC_summary.extend(BEC_summary1_k0)
    BEC_summary.extend(BEC_summary1_k1)

    BEC_summary.extend(BEC_summary2_k7_k8_k2)
    BEC_summary.extend(BEC_summary2_k4_k5)
    BEC_summary.extend(BEC_summary2_k3)
    BEC_summary.extend(BEC_summary2_k6)
    BEC_summary.extend(BEC_summary2_k0)
    BEC_summary.extend(BEC_summary2_k1)

    BEC_summary.extend(BEC_summary3_k7_k8_k2)
    BEC_summary.extend(BEC_summary3_k4_k5)
    BEC_summary.extend(BEC_summary3_k3)
    BEC_summary.extend(BEC_summary3_k6)
    BEC_summary.extend(BEC_summary3_k0)
    BEC_summary.extend(BEC_summary3_k1)

    with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
        pickle.dump(BEC_summary, f)

def check_training_testing_overlap():
    '''
    See if any of the training demonstrations overlaps with any of the testing demonstrations (if there's overlap, it
    will print 'True')
    '''
    data_locs = ['augmented_taxi', 'two_goal', 'skateboard']
    test_difficulties = ['low', 'medium', 'high']

    for data_loc in data_locs:
        print(data_loc)

        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            summary = pickle.load(f)

        for test_difficulty in test_difficulties:
            data_loc_test = 'models/' + data_loc + '/testing/test_' + test_difficulty + '/test_environments.pickle'

            with open(data_loc_test, 'rb') as f:
                test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = pickle.load(f)

            overlap = False
            for test_tuple in test_wt_vi_traj_tuples:
                if policy_summarization_helpers._in_summary(test_tuple[1].mdp, summary, test_tuple[2][0]):
                    overlap = True

            print(overlap)

def create_testing_dictionaries(test_env_dict, mapping):
    '''
    Translate the selected testing pickle files into dictionaries so that they can be uploaded and used in the web user study
    '''

    for data_loc in mapping.keys():
        for test_difficulty in mapping[data_loc].keys():
            data_loc_test = 'models/' + data_loc + '/testing/test_' + test_difficulty + '/test_environments.pickle'

            with open(data_loc_test, 'rb') as f:
                test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = pickle.load(f)

            pairs = mapping[data_loc][test_difficulty]

            test_difficulty_dicts = []

            for pair in pairs:
                pair_mdp_dict = []
                for element in pair:
                    test_wt_vi_traj_tuple = test_wt_vi_traj_tuples[element]

                    mdp = test_wt_vi_traj_tuple[1].mdp
                    optimal_traj = test_wt_vi_traj_tuple[2]
                    test_mdp_dict = test_wt_vi_traj_tuple[3]

                    # postprocessing because MDP wasn't saved correctly in policy_summarization_helpers.py
                    if data_loc == 'augmented_taxi':
                        test_mdp_dict['agent']['x'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('x')
                        test_mdp_dict['agent']['y'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('y')
                        test_mdp_dict['agent']['has_passenger'] = mdp.init_state.get_objects_of_class("agent")[
                            0].get_attribute('has_passenger')

                        test_mdp_dict['walls'] = []
                        for wall in mdp.walls:
                            test_mdp_dict['walls'].append({'x': wall['x'], 'y': wall['y']})

                        test_mdp_dict['tolls'] = []
                        for toll in mdp.tolls:
                            test_mdp_dict['tolls'].append({'x': toll['x'], 'y': toll['y']})

                        test_mdp_dict['passengers'][0]['x'] = mdp.init_state.get_objects_of_class("passenger")[
                            0].get_attribute('x')
                        test_mdp_dict['passengers'][0]['y'] = mdp.init_state.get_objects_of_class("passenger")[
                            0].get_attribute('y')
                        test_mdp_dict['passengers'][0]['dest_x'] = mdp.init_state.get_objects_of_class("passenger")[
                            0].get_attribute('dest_x')
                        test_mdp_dict['passengers'][0]['dest_y'] = mdp.init_state.get_objects_of_class("passenger")[
                            0].get_attribute('dest_y')
                        test_mdp_dict['passengers'][0]['in_taxi'] = mdp.init_state.get_objects_of_class("passenger")[
                            0].get_attribute('in_taxi')

                        test_mdp_dict['env_code'] = mdp.env_code
                    elif data_loc == 'two_goal':
                        test_mdp_dict['agent']['x'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('x')
                        test_mdp_dict['agent']['y'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('y')

                        test_mdp_dict['walls'] = []
                        for wall in mdp.walls:
                            test_mdp_dict['walls'].append({'x': wall['x'], 'y': wall['y']})

                        test_mdp_dict['env_code'] = mdp.env_code
                    else:
                        test_mdp_dict['agent']['x'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('x')
                        test_mdp_dict['agent']['y'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('y')
                        test_mdp_dict['agent']['has_skateboard'] = mdp.init_state.get_objects_of_class("agent")[
                            0].get_attribute('has_skateboard')

                        test_mdp_dict['walls'] = []
                        for wall in mdp.walls:
                            test_mdp_dict['walls'].append({'x': wall['x'], 'y': wall['y']})

                        test_mdp_dict['skateboard'][0]['x'] = mdp.init_state.get_objects_of_class("skateboard")[
                            0].get_attribute('x')
                        test_mdp_dict['skateboard'][0]['y'] = mdp.init_state.get_objects_of_class("skateboard")[
                            0].get_attribute('y')
                        test_mdp_dict['skateboard'][0]['on_agent'] = mdp.init_state.get_objects_of_class("skateboard")[
                            0].get_attribute('on_agent')

                        test_mdp_dict['env_code'] = mdp.env_code

                    print(test_mdp_dict)

                    test_mdp_dict['opt_actions'] = [sas[1] for sas in optimal_traj]
                    test_mdp_dict['opt_traj_length'] = len(optimal_traj)
                    test_mdp_dict['opt_traj_reward'] = mdp.weights.dot(mdp.accumulate_reward_features(optimal_traj).T)[0][0]
                    test_mdp_dict['test_difficulty'] = test_difficulty
                    # to be able to trace the particular environment (0-5)
                    test_mdp_dict['tag'] = element

                    # delete unserializable numpy arrays that aren't necessary
                    del test_mdp_dict['weights_lb']
                    del test_mdp_dict['weights_ub']
                    del test_mdp_dict['weights']

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
        'augmented_taxi': {},
        'two_goal': {},
        'skateboard': {}
    }

    mapping = {
        'augmented_taxi':
            {
                'low': [[0, 4], [1, 2], [3, 5]],
                'medium': [[0, 2], [1, 4], [3, 5]],
                'high': [[0, 1], [2, 3], [4, 5]],
            },
        'two_goal':
            {
                'low': [[0, 2], [3, 4], [1, 5]],
                'medium': [[1, 5], [2, 3], [0, 4]],
                'high': [[0, 3], [1, 4], [2, 5]],
            },
        'skateboard':
            {
                'low': [[0, 2], [1, 3], [4, 5]],
                'medium': [[0, 1], [2, 3], [4, 5]],
                'high': [[0, 1], [2, 3], [4, 5]],
            }
    }


    # combine_summaries()
    # extract_test_demonstrations(data_loc)
    # plot_BEC_histogram(data_loc, params.weights['val'], params.step_cost_flag)
    # check_training_testing_overlap()
    # create_testing_dictionaries(test_env_dict, mapping)
    # print_training_summary_lengths()
    process_human_scores(test_env_dict, mapping)

    # mdp_parameters = {
    # 'agent': {'x': 4, 'y': 1, 'has_passenger': 0},
    # 'walls': [{'x': 1, 'y': 3}, {'x': 1, 'y': 2}],
    # 'passengers': [{'x': 4, 'y': 1, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}],
    # 'tolls': [{'x': 3, 'y': 1}],
    # 'available_tolls': [{"x": 2, "y": 3}, {"x": 3, "y": 3}, {"x": 4, "y": 3},
    # {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2},
    # {"x": 2, "y": 1}, {"x": 3, "y": 1}],
    # 'traffic': [],  # probability that you're stuck
    # 'fuel_station': [],
    # 'width': 4,
    # 'height': 3,
    # 'gamma': 1,
    # }


