import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_normal(normal, X, Y, Z, ax=None):
    # set the normal vector to start at the center of the plane
    startX = np.mean([np.max(X), np.min(X)])
    startY = np.mean([np.max(Y), np.min(Y)])
    startZ = (-normal[0, 0] * startX - normal[0, 1] * startY) * 1. /normal[0, 2]
    delta = 0.2 * normal
    ax.quiver(startX, startY, startZ, delta[0, 0], delta[0, 1], delta[0, 2], arrow_length_ratio=0.15, linewidth=2, color='red')



def visualize_planes_team(constraints, fig=None, ax=None, alpha=0.5, color=None):
    '''
    Plot the planes associated with the normal vectors contained within 'constraints'
    '''
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot(projection='3d')

    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)

    X_xy, Y_xy, = np.meshgrid(x, y)


    

    for constraint in constraints:
        if constraint[0, 2] != 0:
            Z = (-constraint[0, 0] * X_xy - constraint[0, 1] * Y_xy) / constraint[0, 2]
            if color is not None:
                ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha, color=color)
            else:
                ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha)
            plot_normal(constraint, X_xy, Y_xy, Z, ax=ax)
        elif constraint[0, 1] != 0:
            X_xz, Z_xz, = np.meshgrid(x, z)
            Y = (-constraint[0, 0] * X_xz - constraint[0, 2] * Z_xz) / constraint[0, 1]
            if color is not None:
                ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha, color=color)
            else:
                ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha)
            plot_normal(constraint, X_xz, Y, Z_xz, ax=ax)
        else:
            Y_yz, Z_yz, = np.meshgrid(y, z)
            X = (-constraint[0, 1] * Y_yz - constraint[0, 2] * Z_yz) / constraint[0, 0]
            if color is not None:
                ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha, color=color)
            else:
                ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha)
            plot_normal(constraint, X, Y_yz, Z_yz, ax=ax)