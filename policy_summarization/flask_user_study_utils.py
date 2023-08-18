import difflib
import dill as pickle
import policy_summarization.multiprocessing_helpers as mp_helpers
from simple_rl.utils import mdp_helpers

def normalize_trajectories(trajectory, actions, trajectory_counterfactual, actions_counterfactual):
    '''

    Args:
        trajectory: List of states (tuples) defining a trajectory
        actions: List of strings representing actions
        trajectory_counterfactual: List of states (tuples) defining a counterfactual trajectory
        actions_counterfactual: List of strings representing counterfactual actions

    Returns: Two lists of actions that are of the same length, such that a 'no-op' action is inserted in the shorter
    trajectory whenever it's fitting to wait for the other to catch up (i.e. as it waits on an anchor point, a state
    that's shared by the two trajectories)

    '''

    # subtract one since the code below was original created to work with trajectories of (state, action, next_state) tuples,
    # which will have one less element than a trajectory of simply individual states (i.e. state1, state2, state3, etc)
    len_traj = len(trajectory) - 1
    len_counter = len(trajectory_counterfactual) - 1

    anchor_points_wait = []
    matcher = difflib.SequenceMatcher(None, trajectory, trajectory_counterfactual, autojunk=False)
    matches = matcher.get_matching_blocks()

    for match in matches:
        # add states in overlap
        for i in range(match[2]):
            anchor_points_wait.append(trajectory[match[0] + i])

    # print(anchor_points_wait)

    anchor_points_wait.reverse()
    cur_anchor_point = anchor_points_wait.pop()

    step_traj_temp = 0
    step_counter_temp = 0

    normalized_actions = []
    normalized_actions_counterfactual = []

    while (step_traj_temp < len_traj or step_counter_temp < len_counter):
        state = trajectory[step_traj_temp]
        counter_state = trajectory_counterfactual[step_counter_temp]

        # wait
        if (state == cur_anchor_point and state != counter_state):
            step_traj_temp -= 1
            normalized_actions.append('no-op')
        else:
            normalized_actions.append(actions[step_traj_temp])

        if (counter_state == cur_anchor_point and state != counter_state):
            step_counter_temp -= 1
            normalized_actions_counterfactual.append('no-op')
        else:
            normalized_actions_counterfactual.append(actions_counterfactual[step_counter_temp])

        # consider anchor points one at a time
        if (state == counter_state) and (state == cur_anchor_point):
            if len(anchor_points_wait) > 0:
                cur_anchor_point = anchor_points_wait.pop()

        step_traj_temp += 1
        step_counter_temp += 1

    # print('Actions: {}'.format(normalized_actions))
    # print('C_Actions: {}'.format(normalized_actions_counterfactual))

    return normalized_actions, normalized_actions_counterfactual

def obtain_constraint(data_loc, mdp_parameters, opt_traj, opt_traj_features):
    best_env_idx, best_traj_idx = mdp_parameters['env_traj_idxs']

    filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
    with open(filename, 'rb') as f:
        wt_vi_traj_env = pickle.load(f)
    mdp = wt_vi_traj_env[0][1].mdp
    agent = wt_vi_traj_env[0][1]

    human_actions = mdp_parameters['human_actions']

    human_traj = mdp_helpers.rollout_policy(mdp, agent, cur_state=opt_traj[0][0], action_seq=human_actions)
    human_reward_features = mdp.accumulate_reward_features(human_traj)

    constraint = opt_traj_features - human_reward_features

    return constraint

def extract_mdp_dict(vi, mdp, optimal_traj, mdp_dict, data_loc, element=-1, test_difficulty='none', env_traj_idxs=None, variable_filter=None):
    '''
    Extract the MDP information from a demonstration / test tuple (e.g. to be later put into a json)
    '''

    # update the MDP parameters to begin with the desired start state
    if data_loc == 'augmented_taxi2':
        mdp_dict['agent']['x'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('x')
        mdp_dict['agent']['y'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('y')
        mdp_dict['agent']['has_passenger'] = mdp.init_state.get_objects_of_class("agent")[
            0].get_attribute('has_passenger')

        mdp_dict['passengers'][0]['x'] = mdp.init_state.get_objects_of_class("passenger")[
            0].get_attribute('x')
        mdp_dict['passengers'][0]['y'] = mdp.init_state.get_objects_of_class("passenger")[
            0].get_attribute('y')
        mdp_dict['passengers'][0]['dest_x'] = mdp.init_state.get_objects_of_class("passenger")[
            0].get_attribute('dest_x')
        mdp_dict['passengers'][0]['dest_y'] = mdp.init_state.get_objects_of_class("passenger")[
            0].get_attribute('dest_y')
        mdp_dict['passengers'][0]['in_taxi'] = mdp.init_state.get_objects_of_class("passenger")[
            0].get_attribute('in_taxi')

        if (len(mdp.init_state.get_objects_of_class("hotswap_station")) > 0):
            mdp_dict['hotswap_station'][0]['x'] = mdp.init_state.get_objects_of_class("hotswap_station")[
                0].get_attribute('x')
            mdp_dict['hotswap_station'][0]['y'] = mdp.init_state.get_objects_of_class("hotswap_station")[
                0].get_attribute('y')
        else:
            mdp_dict['hotswap_station'] = []

        opt_locations = [[sas[0].get_agent_x(), sas[0].get_agent_y(), sas[0].objects["passenger"][0]["in_taxi"]] for sas in
            optimal_traj]
        opt_locations.append([optimal_traj[-1][-1].get_agent_x(), optimal_traj[-1][-1].get_agent_y(),
                              optimal_traj[-1][-1].objects["passenger"][0]["in_taxi"]])
        mdp_dict['opt_locations'] = opt_locations

    elif data_loc == 'colored_tiles':
        mdp_dict['agent']['x'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('x')
        mdp_dict['agent']['y'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('y')

        opt_locations = [[sas[0].get_agent_x(), sas[0].get_agent_y()] for sas in optimal_traj]
        opt_locations.append([optimal_traj[-1][-1].get_agent_x(), optimal_traj[-1][-1].get_agent_y()])
        mdp_dict['opt_locations'] = opt_locations
    else:
        mdp_dict['agent']['x'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('x')
        mdp_dict['agent']['y'] = mdp.init_state.get_objects_of_class("agent")[0].get_attribute('y')
        mdp_dict['agent']['has_skateboard'] = mdp.init_state.get_objects_of_class("agent")[
            0].get_attribute('has_skateboard')

        if len(mdp.init_state.get_objects_of_class("skateboard")) > 0:
            mdp_dict['skateboard'][0]['x'] = mdp.init_state.get_objects_of_class("skateboard")[
                0].get_attribute('x')
            mdp_dict['skateboard'][0]['y'] = mdp.init_state.get_objects_of_class("skateboard")[
                0].get_attribute('y')
            mdp_dict['skateboard'][0]['on_agent'] = mdp.init_state.get_objects_of_class("skateboard")[
                0].get_attribute('on_agent')
        else:
            mdp_dict['skateboard'] = []

        opt_locations = [[sas[0].get_agent_x(), sas[0].get_agent_y()] for sas in optimal_traj]
        opt_locations.append([optimal_traj[-1][-1].get_agent_x(), optimal_traj[-1][-1].get_agent_y()])
        mdp_dict['opt_locations'] = opt_locations

    mdp_dict['opt_actions'] = [sas[1] for sas in optimal_traj]
    mdp_dict['opt_traj_length'] = len(optimal_traj)
    mdp_dict['opt_traj_reward'] = mdp.weights.dot(mdp.accumulate_reward_features(optimal_traj).T)[0][0]
    mdp_dict['test_difficulty'] = test_difficulty
    # to be able to trace the particular environment (0-5), or know if it's a training demonstration (-1),
    # or if it's a test demonstration whose normalized trajectory should be shown (-2), or a diagnostic test (-3)
    mdp_dict['tag'] = element

    # also obtain all possible optimal trajectories if value iteration object is provided
    all_opt_trajs = mdp_helpers.rollout_policy_recursive(mdp, vi, optimal_traj[0][0], [])
    # extract all of the actions
    all_opt_actions = []
    for opt_traj in all_opt_trajs:
        all_opt_actions.append([sas[1] for sas in opt_traj])
    mdp_dict['all_opt_actions'] = all_opt_actions

    # create placeholders for normalized trajectories actions (for user study)
    mdp_dict['normalized_opt_actions'] = []
    mdp_dict['normalized_opt_locations'] = []
    mdp_dict['normalized_human_actions'] = []
    mdp_dict['normalized_human_locations'] = []
    mdp_dict['human_actions'] = []

    if variable_filter is not None:
        mdp_dict['variable_filter'] = variable_filter.tolist()
    else:
        mdp_dict['variable_filter'] = []

    if env_traj_idxs is not None:
        env_idx, traj_idx = env_traj_idxs
        mdp_dict['env_traj_idxs'] = (int(env_idx), int(traj_idx)) # typecast into python int (e.g. from numpy int64) for future json serialization
    else:
        mdp_dict['env_traj_idxs'] = ()

    # delete unserializable numpy arrays that aren't necessary
    try:
        del mdp_dict['weights_lb']
        del mdp_dict['weights_ub']
        del mdp_dict['weights']
    except:
        pass

    # print(mdp_dict)
    # print(mdp_dict['env_code'])

    return mdp_dict