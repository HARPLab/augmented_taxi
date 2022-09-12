import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
from filterpy.monte_carlo import systematic_resample
from termcolor import colored
from spherical_geometry import great_circle_arc as gca

import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz

class Particles():
    def __init__(self, positions):
        self.positions = np.array(positions)
        self.weights = np.ones(len(positions)) / len(positions)

def reweight_particles(particles, mu, k=4):
    '''
    :param particles: particles
    :param mu: normal of constraints / mean direction of VMF
    :param k: concentration parameter of VMF
    :return: probability of x under this composite distribution (uniform + VMF)
    '''
    p = particles.positions[0].shape[1]
    mu = mu / np.linalg.norm(mu[0, :], ord=2)

    for j, x in enumerate(particles.positions):
        # normalize vectors to lie on the 2-sphere
        x = x / np.linalg.norm(x[0, :], ord=2)
        dot = mu.dot(x.T)

        if dot >= 0:
            # use the uniform dist
            prob = 0.12779
        else:
            # use the VMF dist
            prob = (k ** (p/2 - 1)) / (special.iv((p/2 - 1), k) * (2 * np.pi) ** (p/2)) * np.exp(k * dot)

        particles.weights[j] = particles.weights[j] * prob

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
        centroid = spherical_centroid(particles.positions.squeeze().T)
    ax.scatter(centroid[0], centroid[1], centroid[2], marker='o', c='r', s=100)

    if matplotlib.get_backend() == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


def calc_n_eff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, indexes, noise=0.1):
    particles.positions[:] = particles.positions[indexes]
    for j, position in enumerate(particles.positions):
        perturbed_position = position + np.random.random(position.shape) * noise
        particles.positions[j] = perturbed_position / np.linalg.norm(perturbed_position, ord=2)

    particles.weights = np.ones(len(particles.positions)) / len(particles.positions)


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
def improve_centroid(c, ps):
    # weight every particle equally
    # ans = (ps / np.sqrt(1 - np.power(c@ps, 2))).sum(axis=-1)

    # weight different particles differently
    ans = (ps / np.sqrt(1 - np.power(c@ps, 2))) * np.tile(particles.weights, (3, 1))
    ans = ans.sum(axis=-1)
    norm = np.sqrt(ans @ ans)
    return ans / norm

# def improve_centroid_explicit(c, ps):
#     ans = np.zeros(3)
#     for dim in range(ps.shape[0]):
#         sum = 0
#         for point_idx in range(ps.shape[1]):
#             sum += ps[dim, point_idx] / np.sqrt(1 - (c @ ps[:, point_idx].T) ** 2)
#         ans[dim] = sum
#
#     norm = np.sqrt(ans @ ans)
#     return ans / norm
#
def fixpoint(f, x0, eps=1e-5, maxiter=1000, **kwargs):
    for _ in range(maxiter):
        x = f(x0, **kwargs)
        # if distance(x, x0) < eps:
        if geodist(x, x0) < eps:
            return x
        x0 = x
    raise Exception("Did not converge")

def spherical_centroid(ps, eps=1e-5, maxiter=10000):
    return fixpoint(improve_centroid, np.zeros((3,)), ps=ps, eps=eps, maxiter=maxiter)

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

    return weighted_var


if __name__ == "__main__":
    from numpy.random import seed
    seed(2)

    prior = [np.array([[0, 0, -1]])]
    n_particles = 50
    c = 0.5

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

        for constraint in constraints:
            reweight_particles(particles, constraint)

        n_eff = calc_n_eff(particles.weights)
        print('n_eff: {}'.format(n_eff))
        if n_eff < c * n_particles:
            indexes = systematic_resample(particles.weights)
            resample_from_index(particles, indexes)
            print(colored('Resampled', 'red'))

        centroid = spherical_centroid(particles.positions.squeeze().T)
        normed_var = normalized_weighted_variance(particles, spherical_centroid=centroid)
        print('weighted variance: {}'.format(normed_var))
        print('weighted variance: {}'.format(np.sum(normed_var)))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plot_particles(particles, fig=fig, ax=ax)
        BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)

        # visualize spherical polygon
        ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
        poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
        BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)

        # visualize the ground truth constraint
        w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
        w_normalized = w / np.linalg.norm(w[0, :], ord=2)
        ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='b', s=100)

        plt.show()