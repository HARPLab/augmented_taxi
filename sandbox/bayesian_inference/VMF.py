import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy.matlib
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from policy_summarization import probability_utils as p_utils
import matplotlib
matplotlib.use('TkAgg')

fs = 16

def plot_3d_scatter(data, ax=None, colour='red', sz=30, el=20, az=50, sph=True, sph_colour="gray", sph_alpha=0.03,
                    eq_line=True, pol_line=True, grd=False):
    """
        plot_3d_scatter()
        =================

        Plots 3D samples on the surface of a sphere.

        INPUT:

            * data (array of floats of shape (N,3)) - samples of a spherical distribution such as von Mises-Fisher.
            * ax (axes) - axes on which the plot is constructed.
            * colour (string) - colour of the scatter plot.
            * sz (float) - size of points.
            * el (float) - elevation angle of the plot.
            * az (float) - azimuthal angle of the plot.
            * sph (boolean) - whether or not to inclde a sphere.
            * sph_colour (string) - colour of the sphere if included.
            * sph_alpha (float) - the opacity/alpha value of the sphere.
            * eq_line (boolean) - whether or not to include an equatorial line.
            * pol_line (boolean) - whether or not to include a polar line.
            * grd (boolean) - whether or not to include a grid.

        OUTPUT:

            * ax (axes) - axes on which the plot is contructed.
            * Plot of 3D samples on the surface of a sphere.

    """

    # The polar axis
    if ax is None:
        ax = plt.axes(projection='3d')

    # Check that data is 3D (data should be Nx3)
    d = np.shape(data)[1]
    if d != 3:
        raise Exception("data should be of shape Nx3, i.e., each data point should be 3D.")

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, c=colour)
    ax.view_init(el, az)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    # Add a shaded unit sphere
    if sph:
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:30j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color=sph_colour, alpha=sph_alpha)

    # Add an equitorial line
    if eq_line:
        # t = theta, p = phi
        eqt = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        eqp = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        eqx = 2 * np.sin(eqt) * np.cos(eqp)
        eqy = 2 * np.sin(eqt) * np.sin(eqp) - 1
        eqz = np.zeros(50)

        # Equator line
        ax.plot(eqx, eqy, eqz, color="k", lw=1)

    # Add a polar line
    if pol_line:
        # t = theta, p = phi
        eqt = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        eqp = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        eqx = 2 * np.sin(eqt) * np.cos(eqp)
        eqy = 2 * np.sin(eqt) * np.sin(eqp) - 1
        eqz = np.zeros(50)

        # Polar line
        ax.plot(eqx, eqz, eqy, color="k", lw=1)

    # Draw a centre point
    ax.scatter([0], [0], [0], color="k", s=sz)

    # Turn off grid
    # ax.grid(grd)

    # Ticks
    # ax.set_xticks([-1, 0, 1])
    # ax.set_yticks([-1, 0, 1])
    # ax.set_zticks([-1, 0, 1])

    return ax

# Drawing a fancy vector see Ref. [7]
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_arrow(point, ax, colour="red"):
    """
        plot_arrow(point,ax,colour="red")
        ==============================

        Plots a 3D arrow on the axes ax from the origin to the point mu.

        INPUT:

            * point (array of floats of shape (3,1)) - a 3D point.
            * ax (axes) - axes on which the plot is constructed.
            * colour (string) - colour of the arrow.

    """

    # Can use quiver for a simple arrow
    # ax.quiver(0,0,0,point[0],point[1],point[2],length=1.0,color=colour,pivot="tail")

    # Fancy arrow
    a = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color=colour)
    ax.add_artist(a)

    return ax

if __name__ == "__main__":

    # uniform distribution
    # data3D = p_utils.rand_uniform_hypersphere(N=1000, p=3)
    # fig = plt.figure(figsize=(10,8))
    # ax = plt.axes(projection='3d')
    # ax.scatter(data3D[:,0],data3D[:,1],data3D[:,2],s=5,c='mediumvioletred')
    # plt.show()

    # a) uniform VMF distribution
    Nsim = 500
    # mu_uniform = [0, 1, 0]
    # mu_uniform = mu_uniform / np.linalg.norm(mu_uniform)
    # kappa_uniform = 0
    # data_uniform = p_utils.rand_von_mises_fisher(mu_uniform, kappa=kappa_uniform, N=Nsim)
    #
    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes(projection='3d')
    # plot_3d_scatter(data_uniform,ax)
    # plot_arrow(mu_uniform,ax,colour="red")
    # ax.view_init(25, 15)
    # plt.show()

    # b) sample VMF distributions
    # All sets have the same number of data points

    # # Set 1
    # mu1 = [0, 1, 0]
    # mu1 = mu1 / np.linalg.norm(mu1)
    # kappa1 = 50
    # data1 = p_utils.rand_von_mises_fisher(mu1, kappa=kappa1, N=Nsim)
    #
    # # Set 2
    # mu2 = [0, 0, 1]
    # mu2 = mu2 / np.linalg.norm(mu2)
    # kappa2 = 20
    # data2 = p_utils.rand_von_mises_fisher(mu2, kappa=kappa2, N=Nsim)
    #
    # # Set 3
    # mu3 = [0, 0, -1]
    # mu3 = mu3 / np.linalg.norm(mu3)
    # kappa3 = 20
    # data3 = p_utils.rand_von_mises_fisher(mu3, kappa=kappa3, N=Nsim)
    #
    # # Set 4
    # mu4 = [-10, 0, -1]
    # mu4 = mu4 / np.linalg.norm(mu4)
    # kappa4 = 200
    # data4 = p_utils.rand_von_mises_fisher(mu4, kappa=kappa4, N=Nsim)
    #
    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes(projection='3d')
    #
    # # Set 1
    # plot_3d_scatter(data1,ax)
    # plot_arrow(mu1,ax,colour="red")
    # # Set 2
    # plot_3d_scatter(data2,ax,colour='orange')
    # plot_arrow(mu2,ax,colour="orange")
    # # Set 3
    # plot_3d_scatter(data3,ax,colour='blue')
    # plot_arrow(mu3,ax,colour="blue")
    # # Set 4
    # plot_3d_scatter(data4,ax,colour='green')
    # plot_arrow(mu4,ax,colour="green")
    #
    # # Labels
    # ax.set_xlabel('x',fontsize=fs)
    # ax.set_ylabel('y',fontsize=fs)
    # ax.set_zlabel('z',fontsize=fs)
    #
    # # Viewing angle
    # ax.view_init(20,135)
    #
    # plt.show()

    # c) VMF distribution that mocks a constraint
    Nsim = 200
    # mu_constraint = [0, 1, 0]
    mu_constraint = [1, 0, 2]
    mu_constraint = mu_constraint / np.linalg.norm(mu_constraint)
    kappa_constraint = 4   # 4 seems reasonable as a constraint approximation (k = 0 is uniform)
    data_constraint = p_utils.rand_von_mises_fisher(mu_constraint, kappa=kappa_constraint, N=Nsim, halfspace=False)

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    plot_3d_scatter(data_constraint,ax)
    plot_arrow(mu_constraint,ax,colour="red")
    ax.view_init(0, 0)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

    # # d) multiple constraints multiplied together
    # def multiply_VMF(mu_1, kappa_1, mu_2, kappa_2):
    #     weighted_sum = kappa_1 * mu_1 + kappa_2 * mu_2
    #     weighted_sum_norm = np.linalg.norm(weighted_sum)
    #     new_mu = weighted_sum / weighted_sum_norm
    #     new_kappa = weighted_sum_norm
    #
    #     return new_mu, new_kappa
    #
    # n_iter = 5
    #
    # mu = [0, 1, 0]
    # mu = mu / np.linalg.norm(mu)
    # kappa = 4
    #
    # mu_constraint = [0, 1, 0]
    # mu_constraint = mu_constraint / np.linalg.norm(mu_constraint)
    # kappa_constraint = 4
    #
    # for x in range(n_iter):
    #     mu, kappa = multiply_VMF(mu, kappa, mu_constraint, kappa_constraint)
    #
    # data = p_utils.rand_von_mises_fisher(mu, kappa=kappa, N=Nsim)
    #
    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes(projection='3d')
    # plot_3d_scatter(data,ax)
    # plot_arrow(mu,ax,colour="red")
    # ax.view_init(0, 0)
    # plt.show()