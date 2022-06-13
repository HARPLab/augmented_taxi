# Python imports.
import sys
import dill as pickle
import numpy as np
import copy
from termcolor import colored
from pathos.multiprocessing import ProcessPool as Pool
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
import sage.geometry.polyhedron.library as plib
from scipy.spatial import geometric_slerp

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Other imports.
sys.path.append("simple_rl")
import params
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import make_mdp
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers
from policy_summarization import BEC
import os
import policy_summarization.multiprocessing_helpers as mp_helpers
from simple_rl.utils import mdp_helpers
import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.multiprocessing_helpers as mp_helpers
from pypoman import compute_polygon_hull, indicate_violating_constraints
import polytope as pc
import policy_summarization.BEC_visualization as BEC_viz


def toll_weight(pool, data_loc, step_cost_flag, weights):
    mdp_class = 'augmented_taxi2'
    w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    w_normalized = w / np.linalg.norm(w[0, :], ord=2)

    # environment in thesis proposal
    mdp_parameters = {
        'agent': {'x': 3, 'y': 2, 'has_passenger': 0},
        'walls': [{'x': 1, 'y': 3}, {'x': 1, 'y': 2}],
        'passengers': [{'x': 2, 'y': 3, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}],
        'tolls': [{'x': 2, 'y': 2}],
        'available_tolls': [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}],
        'traffic': [],  # probability that you're stuck
        'fuel_station': [],
        'hotswap_station': [{'x': 4, 'y': 3}],
        'available_hotswap_stations': [{'x': 4, 'y': 3}],
        'width': 4,
        'height': 3,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized,
        'weights_lb': w_normalized,
        'weights_ub': w_normalized
    }

    w_human = np.array([[-3, 1, -1]])  # incorrect #1 (still maintains the correct sign of each variable)
    # w_human = np.array([[-0.5, 3.5, -1]])  # incorrect #2 (still maintains the correct sign of each variable)
    w_normalized_human = w_human / np.linalg.norm(w_human[0, :], ord=2)
    print(w_normalized_human)

    mdp_parameters_human = {
        'agent': {'x': 3, 'y': 2, 'has_passenger': 0},
        'walls': [{'x': 1, 'y': 3}, {'x': 1, 'y': 2}],
        'passengers': [{'x': 2, 'y': 3, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}],
        'tolls': [{'x': 2, 'y': 2}],
        'available_tolls': [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}],
        'traffic': [],  # probability that you're stuck
        'fuel_station': [],
        'hotswap_station': [{'x': 4, 'y': 3}],
        'available_hotswap_stations': [{'x': 4, 'y': 3}],
        'width': 4,
        'height': 3,
        'gamma': 1,
        'env_code': [],
        'weights': w_normalized_human,
        'weights_lb': w_normalized_human,
        'weights_ub': w_normalized_human
    }


    # prior = []
    # prior = [np.array([[-1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 0, -1]])]     # knowing the right signs
    prior = [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])]
    posterior = [np.array([[1, 1, 0]]), np.array([[-1,  0,  2]]), np.array([[0, -1, -4]])]

    # mdp_agent.visualize_trajectory(trajectory_human)

    mdp_agent_human = make_mdp.make_custom_mdp(mdp_class, mdp_parameters_human)
    vi_agent_human = ValueIteration(mdp_agent_human, sample_rate=1)
    vi_agent_human.run_vi()
    trajectory_human = mdp_helpers.rollout_policy(mdp_agent_human, vi_agent_human)

    # a) obtain constraints through one-step deviation from a human's response
    _, min_subset_constraints_record, env_record, traj_record, _, _ = BEC.extract_constraints(data_loc, step_cost_flag, pool, vi_traj_triplets=[
        (1, vi_agent_human, trajectory_human)])

    min_subset_constraints_record = min_subset_constraints_record[0][0]

    min_subset_constraints_record_one_step = min_subset_constraints_record.copy()

    # b) obtain constraints through sampling potential weights
    # (approximately) uniformly divide up the valid BEC area along 2-sphere
    sample_human_models = BEC_helpers.sample_human_models_uniform(prior, 8)
    min_subset_constraints_record = prior.copy()
    min_subset_constraints_record_sample = []

    # mdp_agent = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)
    # vi_agent = ValueIteration(mdp_agent, sample_rate=1)
    # vi_agent.run_vi()
    # trajectory = mdp_helpers.rollout_policy(mdp_agent, vi_agent)

    # mdp_agent.visualize_trajectory(trajectory)

    mu_sa = mdp_agent_human.accumulate_reward_features(trajectory_human, discount=True)

    for j, sample_human_model in enumerate(sample_human_models):
        sample_human_model = sample_human_model / np.linalg.norm(sample_human_model[0, :], ord=2)

        print('{}: {}'.format(j, sample_human_model))
        mdp_parameters_subopt = mdp_parameters_human.copy()
        mdp_parameters_subopt['weights'] = sample_human_model
        mdp_agent_subopt = make_mdp.make_custom_mdp(mdp_class, mdp_parameters_subopt)
        vi_agent_subopt = ValueIteration(mdp_agent_subopt, sample_rate=1)
        vi_agent_subopt.run_vi()

        if vi_agent_subopt.stabilized:
            trajectory_subopt = mdp_helpers.rollout_policy(mdp_agent_subopt, vi_agent_subopt)

            traj_hyp_list = []
            for a in trajectory_subopt:
                traj_hyp_list.append(a[1])
            print(traj_hyp_list)

            mu_sb = mdp_agent_subopt.accumulate_reward_features(trajectory_subopt, discount=True)

            min_subset_constraints_record.append(mu_sa - mu_sb)
            min_subset_constraints_record_sample.append(mu_sa - mu_sb)


    # combine the two
    min_subset_constraints_record.extend(min_subset_constraints_record_one_step)

    print(min_subset_constraints_record)
    min_subset_constraints_record = BEC_helpers.remove_redundant_constraints(min_subset_constraints_record, weights, step_cost_flag)
    print(min_subset_constraints_record)

    # incorrect - correct (for general environment), #1
    min_subset_constraints_record.extend([np.array([[0, 0, 9]]) - np.array([[0, 1, 11]])])
    # min_subset_constraints_record = [np.array([[0, 0, 9]]) - np.array([[0, 1, 11]])]

    # incorrect - correct (for general environment), #2
    # min_subset_constraints_record.extend([np.array([[1, 1, 9]]) - np.array([[0, 1, 11]])])


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_subset_constraints_record)
    poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # visualize spherical polygon
    BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)

    ieqs_posterior = BEC_helpers.constraints_to_halfspace_matrix_sage(posterior)
    poly_posterior = Polyhedron.Polyhedron(ieqs=ieqs_posterior)  # automatically finds the minimal H-representation
    BEC_viz.visualize_spherical_polygon(poly_posterior, fig=fig, ax=ax, plot_ref_sphere=False, color='g')

    BEC_viz.visualize_planes(min_subset_constraints_record, fig=fig, ax=ax)

    ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100)
    if mdp_class == 'augmented_taxi2':
        ax.set_xlabel('$\mathregular{w_0}$: Mud')
        ax.set_ylabel('$\mathregular{w_1}$: Recharge')
    elif mdp_class == 'two_goal2':
        ax.set_xlabel('X: Goal 1 (grey)')
        ax.set_ylabel('Y: Goal 2 (green)')
    else:
        ax.set_xlabel('X: Goal')
        ax.set_ylabel('Y: Skateboard')
    ax.set_zlabel('$\mathregular{w_2}$: Action')

    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    if matplotlib.get_backend() == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    plt.show()

if __name__ == "__main__":
    pool = Pool(1)
    data_loc = params.data_loc['BEC']
    step_cost_flag = params.step_cost_flag
    weights = params.weights['val']

    toll_weight(pool, data_loc, step_cost_flag, weights)