import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
from filterpy.monte_carlo import systematic_resample, residual_resample
from termcolor import colored
from spherical_geometry import great_circle_arc as gca
from scipy.stats import truncexpon as truncexpon, expon
import copy
from numpy.random import uniform
from numpy.random import randn
import numpy.matlib

import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz
import policy_summarization.computational_geometry as cg

fs = 16

class Particles():
    def __init__(self, positions):
        self.positions = np.array(positions)
        self.weights = np.ones(len(positions)) / len(positions)

        # copy over
        self.positions_prev = self.positions.copy()
        self.weights_prev = self.weights.copy()

        # todo: i'm not really utilizing clusters yet
        self.clusters = np.ones(len(positions))

        self.entropy = calc_entropy(self, [])

def observation_probability(x, constraints, k=4):
    prob = 1
    p = x.shape[1]

    for constraint in constraints:
        dot = constraint.dot(x.T)

        if dot >= 0:
            # use the uniform dist
            prob *= 0.12779
        else:
            # use the VMF dist
            prob *= (k ** (p / 2 - 1)) / (special.iv((p / 2 - 1), k) * (2 * np.pi) ** (p / 2)) * np.exp(k * dot)[0][0]

    return prob

def reweight_particles(particles, constraints):
    '''
    :param particles: particles
    :param constraints: normal of constraints / mean direction of VMF
    :param k: concentration parameter of VMF
    :return: probability of x under this composite distribution (uniform + VMF)
    '''
    particles.weights_prev = particles.weights.copy()

    for j, x in enumerate(particles.positions):
        particles.weights[j] = particles.weights[j] * observation_probability(x, constraints)

    # normalize weights and update particles
    particles.weights /= sum(particles.weights)


def plot_particles(particles, centroid=None, fig=None, ax=None):
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca(projection='3d')

    for j, position in enumerate(particles.positions):
        ax.scatter(position[0, 0], position[0, 1], position[0, 2], s=particles.weights[j]*800, color='tab:blue')

    if centroid == None:
        centroid = spherical_centroid(particles.positions.squeeze().T, particles.weights)
    ax.scatter(centroid[0], centroid[1], centroid[2], marker='o', c='r', s=100)

    if matplotlib.get_backend() == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


def calc_n_eff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, indexes, noise=0.1):
    particles.weights_prev = particles.weights.copy()
    particles.positions_prev = particles.positions.copy()

    particles.positions[:] = particles.positions[indexes]
    for j, position in enumerate(particles.positions):
        perturbed_position = position + np.random.random(position.shape) * noise
        particles.positions[j] = perturbed_position / np.linalg.norm(perturbed_position, ord=2)

    particles.weights = np.ones(len(particles.positions)) / len(particles.positions)

    particles.clusters = indexes

def distance(x, y, axis=0):
    return np.sqrt((np.power(x - y, 2)).sum(axis=axis))

def geodist(x, y, eps=1e-6):
    dotprod = y.T @ x
    assert ((-1 - eps) <= dotprod).all() and (dotprod <= (1 + eps)).all()
    dotprod = dotprod.clip(-1, 1)
    return np.arccos(dotprod)

def avg_distance(ps, m):
    # Allow m to be a vector *or* a matrix
    if len(m.shape) == 1:
        m = m[:, None]
    return geodist(ps, m).mean(axis=1)

# inspired by https://skeptric.com/calculate-centroid-on-sphere/#Attempting-to-Implement-Algorithm-A1
def improve_centroid(c, ps, weights):
    '''
    :param c: old centroid
    :param ps: points
    :return: new centroid
    '''
    # weight every particle equally
    # ans = (ps / np.sqrt(1 - np.power(c@ps, 2))).sum(axis=-1)

    # weight different particles differently
    ans = (ps / np.sqrt(1 - np.power(c@ps, 2))) * np.tile(weights, (3, 1))
    ans = ans.sum(axis=-1)
    norm = np.sqrt(ans @ ans)
    return ans / norm

def fixpoint(f, x0, eps=1e-5, maxiter=1000, **kwargs):
    for _ in range(maxiter):
        x = f(x0, **kwargs)
        if geodist(x, x0) < eps:
            return x
        x0 = x
    raise Exception("Did not converge")

def spherical_centroid(ps, weights, eps=1e-5, maxiter=10000):
    return fixpoint(improve_centroid, np.zeros((3,)), ps=ps, weights=weights, eps=eps, maxiter=maxiter)

# http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
def normalized_weighted_variance(particles, spherical_centroid=None):
    sum_weights = np.sum(particles.weights)

    eff_dof = (sum_weights ** 2) / (np.sum(particles.weights ** 2))

    if spherical_centroid is None:
        spherical_centroid = spherical_centroid(particles.positions.squeeze().T)

    # take the trace of the covariance matrix as the variance measure
    weighted_var = 0
    for j, particle_position in enumerate(particles.positions):
        # geodesic distance
        weighted_var += particles.weights[j] * gca.length(spherical_centroid, particle_position, degrees=False) ** 2 / sum_weights * (eff_dof / (eff_dof - 1))

    # normalize
    weighted_var = weighted_var / len(particles.positions)

    return weighted_var[0]

def calc_entropy(particles, constraints):
    '''
    Implement entropy measure for particle filter as described in Particle Filter Based Entropy (boers_mandal_ICIF2010)

    Spherical discretization taken from : A new method to subdivide a spherical surface into equal-area cells,
    Malkin 2019 https://arxiv.org/pdf/1612.03467.pdf
    '''

    # sort particles into bins
    positions_spherical = []  # azi (-pi, pi), ele (0, pi)
    for position in particles.positions:
        positions_spherical.append((cg.cart2sph(*position[0])))

    ele_bin_edges = np.array([0.1721331, 0.34555774, 0.52165622, 0.69434434, 0.86288729,
           1.04435092, 1.20822512, 1.39251443, 1.57079633, 1.74907822,
           1.93336753, 2.09724173, 2.27870536, 2.44724832, 2.61993643,
           2.79603491, 2.96945956, 3.1416])
    azi_bin_edges =\
        {0: np.array([0.        , 2.0943951 , 4.1887902 , 6.28318531]),
         1: np.array([0.        , 0.6981317 , 1.3962634 , 2.0943951 , 2.7925268 ,
       3.4906585 , 4.1887902 , 4.88692191, 5.58505361, 6.28318531]),
         2: np.array([0.        , 0.41887902, 0.83775804, 1.25663706, 1.67551608,
       2.0943951 , 2.51327412, 2.93215314, 3.35103216, 3.76991118,
       4.1887902 , 4.60766923, 5.02654825, 5.44542727, 5.86430629,
       6.28318531]),
         3: np.array([0.        , 0.31415927, 0.62831853, 0.9424778 , 1.25663706,
       1.57079633, 1.88495559, 2.19911486, 2.51327412, 2.82743339,
       3.14159265, 3.45575192, 3.76991118, 4.08407045, 4.39822972,
       4.71238898, 5.02654825, 5.34070751, 5.65486678, 5.96902604,
       6.28318531]),
         4: np.array([0.        , 0.26179939, 0.52359878, 0.78539816, 1.04719755,
       1.30899694, 1.57079633, 1.83259571, 2.0943951 , 2.35619449,
       2.61799388, 2.87979327, 3.14159265, 3.40339204, 3.66519143,
       3.92699082, 4.1887902 , 4.45058959, 4.71238898, 4.97418837,
       5.23598776, 5.49778714, 5.75958653, 6.02138592, 6.28318531]),
         5: np.array([0.        , 0.20943951, 0.41887902, 0.62831853, 0.83775804,
       1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
       2.0943951 , 2.30383461, 2.51327412, 2.72271363, 2.93215314,
       3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
       4.1887902 , 4.39822972, 4.60766923, 4.81710874, 5.02654825,
       5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458 ,
       6.28318531]),
         6: np.array([0.        , 0.20943951, 0.41887902, 0.62831853, 0.83775804,
       1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
       2.0943951 , 2.30383461, 2.51327412, 2.72271363, 2.93215314,
       3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
       4.1887902 , 4.39822972, 4.60766923, 4.81710874, 5.02654825,
       5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458 ,
       6.28318531]),
         7: np.array([0.        , 0.17453293, 0.34906585, 0.52359878, 0.6981317 ,
       0.87266463, 1.04719755, 1.22173048, 1.3962634 , 1.57079633,
       1.74532925, 1.91986218, 2.0943951 , 2.26892803, 2.44346095,
       2.61799388, 2.7925268 , 2.96705973, 3.14159265, 3.31612558,
       3.4906585 , 3.66519143, 3.83972435, 4.01425728, 4.1887902 ,
       4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
       5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
       6.10865238, 6.28318531]),
         8: np.array([0.        , 0.17453293, 0.34906585, 0.52359878, 0.6981317 ,
       0.87266463, 1.04719755, 1.22173048, 1.3962634 , 1.57079633,
       1.74532925, 1.91986218, 2.0943951 , 2.26892803, 2.44346095,
       2.61799388, 2.7925268 , 2.96705973, 3.14159265, 3.31612558,
       3.4906585 , 3.66519143, 3.83972435, 4.01425728, 4.1887902 ,
       4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
       5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
       6.10865238, 6.28318531]),
         9: np.array([0.        , 0.17453293, 0.34906585, 0.52359878, 0.6981317 ,
       0.87266463, 1.04719755, 1.22173048, 1.3962634 , 1.57079633,
       1.74532925, 1.91986218, 2.0943951 , 2.26892803, 2.44346095,
       2.61799388, 2.7925268 , 2.96705973, 3.14159265, 3.31612558,
       3.4906585 , 3.66519143, 3.83972435, 4.01425728, 4.1887902 ,
       4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
       5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
       6.10865238, 6.28318531]),
         10: np.array([0.        , 0.17453293, 0.34906585, 0.52359878, 0.6981317 ,
       0.87266463, 1.04719755, 1.22173048, 1.3962634 , 1.57079633,
       1.74532925, 1.91986218, 2.0943951 , 2.26892803, 2.44346095,
       2.61799388, 2.7925268 , 2.96705973, 3.14159265, 3.31612558,
       3.4906585 , 3.66519143, 3.83972435, 4.01425728, 4.1887902 ,
       4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
       5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
       6.10865238, 6.28318531]),
         11: np.array([0.        , 0.20943951, 0.41887902, 0.62831853, 0.83775804,
       1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
       2.0943951 , 2.30383461, 2.51327412, 2.72271363, 2.93215314,
       3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
       4.1887902 , 4.39822972, 4.60766923, 4.81710874, 5.02654825,
       5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458 ,
       6.28318531]),
         12: np.array([0.        , 0.20943951, 0.41887902, 0.62831853, 0.83775804,
       1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
       2.0943951 , 2.30383461, 2.51327412, 2.72271363, 2.93215314,
       3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
       4.1887902 , 4.39822972, 4.60766923, 4.81710874, 5.02654825,
       5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458 ,
       6.28318531]),
         13: np.array([0.        , 0.26179939, 0.52359878, 0.78539816, 1.04719755,
       1.30899694, 1.57079633, 1.83259571, 2.0943951 , 2.35619449,
       2.61799388, 2.87979327, 3.14159265, 3.40339204, 3.66519143,
       3.92699082, 4.1887902 , 4.45058959, 4.71238898, 4.97418837,
       5.23598776, 5.49778714, 5.75958653, 6.02138592, 6.28318531]),
         14: np.array([0.        , 0.31415927, 0.62831853, 0.9424778 , 1.25663706,
       1.57079633, 1.88495559, 2.19911486, 2.51327412, 2.82743339,
       3.14159265, 3.45575192, 3.76991118, 4.08407045, 4.39822972,
       4.71238898, 5.02654825, 5.34070751, 5.65486678, 5.96902604,
       6.28318531]),
         15: np.array([0.        , 0.41887902, 0.83775804, 1.25663706, 1.67551608,
       2.0943951 , 2.51327412, 2.93215314, 3.35103216, 3.76991118,
       4.1887902 , 4.60766923, 5.02654825, 5.44542727, 5.86430629,
       6.28318531]),
         16: np.array([0.        , 0.6981317 , 1.3962634 , 2.0943951 , 2.7925268 ,
    3.4906585 , 4.1887902 , 4.88692191, 5.58505361, 6.28318531]),
         17: np.array([0.        , 2.0943951 , 4.1887902 , 6.28318531])}

    # dictionary with keys based on elevation, and values based on associated number of azimuth bins
    weight_dict = {0: np.zeros(3), 1: np.zeros(9), 2: np.zeros(15), 3: np.zeros(20), 4: np.zeros(24), 5: np.zeros(30),
                   6: np.zeros(30), 7: np.zeros(36), 8: np.zeros(36), 9: np.zeros(36), 10: np.zeros(36),
                   11: np.zeros(30), 12: np.zeros(30), 13: np.zeros(24), 14: np.zeros(20), 15: np.zeros(15),
                   16: np.zeros(9), 17: np.zeros(3)}

    azimuths, elevations = zip(*positions_spherical)

    # sort the points into elevation bins
    elevation_bins = np.digitize(elevations, ele_bin_edges)

    # for each point, sort into the correct azimuth bin
    for j, elevation_bin in enumerate(elevation_bins):
        azimuth_bin = np.digitize(azimuths[j], azi_bin_edges[elevation_bin])
        weight_dict[elevation_bin][azimuth_bin] += particles.weights[j]

    entropy = 0
    for ele_bin in weight_dict.keys():
        for azi_prob in weight_dict[ele_bin]:
            if azi_prob > 0:
                entropy += azi_prob * np.log(azi_prob)
    entropy = entropy * -1

    # perform some basic checks (e.g. that the weights of the particles sum to one)
    sum = 0
    for item in list(weight_dict.values()):
        sum += np.sum(np.array(item))
    assert np.isclose(sum, 1)

    assert entropy >= 0, "Entropy shouldn't be negative!"

    return entropy

def calc_info_gain(particles, new_constraints):
    new_particles = copy.deepcopy(particles)
    new_particles = update_particle_filter(new_particles, new_constraints)

    return particles.entropy - new_particles.entropy

def update_particle_filter(particles, constraints, c=0.5):
    reweight_particles(particles, constraints)
    # print('prior entropy: {}'.format(particles.entropy))

    n_eff = calc_n_eff(particles.weights)
    # print('n_eff: {}'.format(n_eff))
    if n_eff < c * len(particles.weights):
        indexes = systematic_resample(particles.weights)
        resample_from_index(particles, indexes)
        # print(np.unique(indexes, return_counts=True))
        print(colored('Resampled', 'red'))

    # centroid = spherical_centroid(particles.positions.squeeze().T, particles.weights)
    # normed_var = normalized_weighted_variance(particles, spherical_centroid=centroid)
    # print('weighted variance: {}'.format(normed_var))

    particles.entropy = calc_entropy(particles, constraints)
    print('entropy: {}'.format(particles.entropy))

    return particles

def IROS_demonstrations():
    prior = [np.array([[0, 0, -1]])]
    n_particles = 5000

    particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    particles = Particles(particle_positions)

    constraints_list = [prior]
    constraints_list.extend([[np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]])], [np.array([[0, 1, 2]])],
                   [np.array([[  0,  -1, -10]]), np.array([[ 0, -1, -4]])], [np.array([[1, 1, 0]])]])

    # todo: for playing around with what happens if you get a constraint wrong repeatedly
    # constraints_list.extend([[np.array([[0,  1,  2]]), np.array([[-1,  0,  2]])], [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])]
    #                          , [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])]])

    constraints_running = []
    for j, constraints in enumerate(constraints_list):
        print(j)

        # todo: for playing around with what happens if you get a constraint wrong repeatedly
        # if j == 2:
        #     del constraints_running[2]
        constraints_running.extend(constraints)
        constraints_running = BEC_helpers.remove_redundant_constraints(constraints_running, None, False)

        # print(calc_info_gain(particles, constraints))

        particles = update_particle_filter(particles, constraints)

        seen_dict = {}
        for pos in particles.positions:
            x = tuple(*pos)
            if x not in seen_dict.keys():
                seen_dict[x] = 1
            else:
                seen_dict[x] += 1

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.set_facecolor('white')
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        #
        # plot_particles(particles, fig=fig, ax=ax)
        # BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
        #
        # # visualize spherical polygon
        # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
        # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
        # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
        #
        # # visualize the ground truth constraint
        # w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
        # w_normalized = w / np.linalg.norm(w[0, :], ord=2)
        # ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='b', s=100)
        #
        # plt.show()

if __name__ == "__main__":
    from numpy.random import seed
    seed(2)

    IROS_demonstrations()