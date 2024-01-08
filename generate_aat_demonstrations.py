import pickle
import numpy as np
from collections import defaultdict
import policy_summarization.BEC_helpers as BEC_helpers
from termcolor import colored
from policy_summarization import computational_geometry as cg

import sage.all
import sage.geometry.polyhedron.base as Polyhedron

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import policy_summarization.BEC_visualization as BEC_viz
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

def extract_distances(robot_positions, mudsteps):
    total_dist = 0.0
    mud_dist = 0.0

    for point_idx, point in enumerate(robot_positions):
        if point_idx == 0:
            prev_x = point.x
            prev_y = point.y
            continue
        dist = np.sqrt((prev_x - point.x) ** 2 + (prev_y - point.y) ** 2)
        total_dist += dist
        if mudsteps[point_idx]:
            mud_dist += dist
        prev_x = point.x
        prev_y = point.y

    return total_dist, mud_dist

def generate_scenarios_constraints(directory_name, counterfactual_action_costs):
    data_dir = f'aat_data/{directory_name}'

    scenarios = np.arange(7, 19)

    scenarios_constraints = defaultdict(list)
    scenarios_feature_counts = defaultdict(list)

    for scenario in scenarios:
        simulation_name = 'scenario' + str(scenario)

        with open(f'{data_dir}/in_mud_{simulation_name}.pickle', 'rb') as f:
            mudsteps = pickle.load(f)

        with open(f'{data_dir}/robot_positions_{simulation_name}.pickle', 'rb') as f:
            robot_positions = pickle.load(f)

        total_dist, mud_dist = extract_distances(robot_positions, mudsteps)
        scenario_feature_counts = np.array([[mud_dist, 0, total_dist]])

        scenarios_feature_counts[scenario].append(scenario_feature_counts)

        for counterfactual_action_cost in counterfactual_action_costs:
            counterfactual_directory_name = 'regular_' + counterfactual_action_cost
            counterfactual_data_dir = f'aat_data/{counterfactual_directory_name}'
            with open(f'{counterfactual_data_dir}/in_mud_{simulation_name}.pickle', 'rb') as f:
                counterfactual_mudsteps = pickle.load(f)
            with open(f'{counterfactual_data_dir}/robot_positions_{simulation_name}.pickle', 'rb') as f:
                counterfactual_robot_positions = pickle.load(f)

            counterfactual_total_dist, counterfactual_mud_dist = extract_distances(counterfactual_robot_positions, counterfactual_mudsteps)
            counterfactual_scenario_feature_counts = np.array([[counterfactual_mud_dist, 0, counterfactual_total_dist]])

            scenarios_feature_counts[scenario].append(counterfactual_scenario_feature_counts)

            scenarios_constraints[scenario].append(scenario_feature_counts - counterfactual_scenario_feature_counts)

    with open('aat_data/scenarios_constraints.pickle', 'wb') as f:
        pickle.dump((scenarios_constraints, scenarios_feature_counts), f)



def sample_human_models_uniform_2D(constraints, n_models):
    '''
    Summary: sample representative weights that the human could currently attribute to the agent, by greedily selecting
    points that minimize the maximize distance to any other point (k-centers problem)

    Currently assuming information only lies along the x / y axes and that the y axis can be ignored
    '''

    sample_human_models = []


    constraints_matrix = np.vstack(constraints)

    # obtain x, y, z coordinates on the sphere that obey the constraints
    valid_sph_x, valid_sph_y, valid_sph_z = cg.sample_valid_region(constraints_matrix, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

    if len(valid_sph_x) == 0:
        print(colored("Was unable to sample valid human models within the BEC (which is likely too small).",
                    'red'))
        return sample_human_models

    projected_polygon_rho = []
    projected_polygon_phi = []
    for x in range(len(valid_sph_x)):
        rho, phi = cg.cart2pol(valid_sph_x[x], valid_sph_z[x])
        projected_polygon_rho.append(rho)
        projected_polygon_phi.append(phi)

    min_phi = min(projected_polygon_phi)
    max_phi = max(projected_polygon_phi)

    # add an offset factor to account from transition between 2D and 3D coordinate systems
    phi_list = np.linspace(min_phi, max_phi, n_models + 2)[1:-1] - np.pi/2

    sample_human_models = []
    for phi in phi_list:
        # add an offset factor to account from transition between 2D and 3D coordinate systems
        sample_human_models.append(cg.sph2cat(np.pi, phi))

    return sample_human_models

def convert_human_models_to_mud_percentages(sample_human_models):
    # formula for conversion (note that signs also have be flipped): 1 / mud_weight_normalized_by_action_weight = mud_delay_percent (i.e. what I can directly put into scenario_.txt)
    mud_percentages = []
    for sample_human_model in sample_human_models:
        normalized_sample_human_model = sample_human_model / abs(sample_human_model[2])
        mud_percentages.append((-1 / normalized_sample_human_model)[0]) # multiply by -1 to account for difference perspective between cost (planning) and reward (RL)
    return mud_percentages


def visualize_human_models(human_models):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    for weights in human_models:
        ax.scatter(weights[0], weights[1], weights[2], marker='o', c='r', s=100)

    ax.set_xlabel('$\mathregular{w_0}$: Mud')
    ax.set_ylabel('$\mathregular{w_1}$: Recharge')
    ax.set_zlabel('$\mathregular{w_2}$: Action')

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    # ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    # ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    # ax.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    if matplotlib.get_backend() == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    plt.show()

if __name__ == "__main__":
    # formula for conversion (note that signs also have be flipped): 1 / mud_weight_normalized_by_action_weight = mud_delay_percent
    w = np.array([[-2, 0, -1]])  # mud weight, step weight
    weights = w / np.linalg.norm(w[0, :], ord=2)

    prior = [np.array([[0, 0, -1]])]
    # prior = [np.array([[ 0.,  0., -1.]]), np.array([[-2.52778247, 0., 3.21958631]])]

    step_cost_flag = False
    n_human_models = 3

    sample_human_models = sample_human_models_uniform_2D(prior, n_human_models)
    mud_percentages = convert_human_models_to_mud_percentages(sample_human_models)
    # visualize_human_models(sample_human_models)

    directory_name = 'regular_50'
    counterfactual_action_costs = ['0_58', '1_73'] # round 1
    # counterfactual_action_costs = ['0_17', '0_35', '0_55'] # round 2

    generate_scenarios_constraints(directory_name, counterfactual_action_costs)

    with open('aat_data/scenarios_constraints.pickle', 'rb') as f:
        scenarios_constraints, scenarios_feature_counts = pickle.load(f)

    # analytics
    std_devs = np.zeros((len(scenarios_feature_counts.keys()), 3))
    for scenario_idx, scenario in enumerate(scenarios_feature_counts.keys()):
        std_devs[scenario_idx, :] = np.stack(scenarios_feature_counts[scenario]).squeeze().std(axis=0)
        print(std_devs[scenario_idx, 2] / std_devs[scenario_idx, 0])
    print(std_devs)
    print(std_devs.mean(axis=0))

    scenarios_constraints_list = list(scenarios_constraints.values())

    print(scenarios_feature_counts)
    for scenario_constraints in scenarios_constraints_list:
        info_gain = BEC_helpers.calculate_information_gain(prior, scenario_constraints, weights, False)
        print(info_gain)
        print(scenario_constraints)

    for scenario_idx, scenario_constraints in enumerate(scenarios_constraints_list):
        print('Scenario: {}'.format(list(scenarios_constraints.keys())[scenario_idx]))
        constraints_record = prior.copy()
        constraints_record.extend(scenario_constraints)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        solid_angle = BEC_helpers.calc_solid_angles([constraints_record])[0]
        print(solid_angle)

        ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_record)
        poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation

        min_constraints = BEC_helpers.remove_redundant_constraints(constraints_record, weights, step_cost_flag)
        print(min_constraints)
        for constraints in [min_constraints]:
            BEC_viz.visualize_planes(constraints, fig=fig, ax=ax)

        # visualize spherical polygon
        BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)

        ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='g', s=100)
        ax.set_xlabel('$\mathregular{w_0}$: Mud')
        ax.set_ylabel('$\mathregular{w_1}$: Recharge')
        ax.set_zlabel('$\mathregular{w_2}$: Action')

        # for sample_human_model in sample_human_models:
        #     ax.scatter(sample_human_model[0], sample_human_model[1], sample_human_model[2], marker='o', c='r', s=100)
        #     ax.set_xlabel('$\mathregular{w_0}$: Mud')
        #     ax.set_ylabel('$\mathregular{w_1}$: Recharge')
        #     ax.set_zlabel('$\mathregular{w_2}$: Action')

        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])

        if matplotlib.get_backend() == 'TkAgg':
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

        plt.show()

