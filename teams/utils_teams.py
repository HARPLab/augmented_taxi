import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools


def plot_normal(normal, X, Y, Z, ax=None):
    # set the normal vector to start at the center of the plane
    startX = np.mean([np.max(X), np.min(X)])
    startY = np.mean([np.max(Y), np.min(Y)])
    startZ = (-normal[0, 0] * startX - normal[0, 1] * startY) * 1. /normal[0, 2]
    delta = 0.2 * normal
    ax.quiver(startX, startY, startZ, delta[0, 0], delta[0, 1], delta[0, 2], arrow_length_ratio=0.15, linewidth=2, color='red')


def subset_surface_plot_data(X, Y, Z):


    # for [-3,0,2] constraint
    x_min, x_max = -10, 10
    y_min, y_max = -8, 8

    # # for [-2,0,1] constraint
    # x_min, x_max = -10, 10
    # y_min, y_max = -8, 8     

    # Slice the data arrays to the desired region
    X_subset = X[x_min:x_max, y_min:y_max]
    Y_subset = Y[x_min:x_max, y_min:y_max]
    Z_subset = Z[x_min:x_max, y_min:y_max]

    return X_subset, Y_subset, Z_subset



def visualize_planes_team(constraints, fig=None, ax=None, alpha=0.5, color=None):
    '''
    Plot the planes associated with the normal vectors contained within 'constraints'
    '''

    cnst_to_restrict = np.array([[-5,  0,  2]])

    plot_normal_flag = False

    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot(projection='3d')

    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)

    

    for constraint_id in range(len(constraints)):
        constraint = constraints[constraint_id]
        # print('constraint:', constraint)
        if constraint[0, 2] != 0:
            X_xy, Y_xy, = np.meshgrid(x, y)
            Z = (-constraint[0, 0] * X_xy - constraint[0, 1] * Y_xy) / constraint[0, 2]
            if (constraint == cnst_to_restrict).all():
                X_xy, Y_xy, Z = subset_surface_plot_data(X_xy, Y_xy, Z)
            if color is not None:
                if len(color) == len(constraints):
                    print('color:', color[constraint_id])
                    ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha, color=color[constraint_id])
                else:
                    print('color:', color)
                    ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha, color=color)
            else:
                ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha)
            if plot_normal_flag:
                plot_normal(constraint, X_xy, Y_xy, Z, ax=ax)
        elif constraint[0, 1] != 0:
            X_xz, Z_xz, = np.meshgrid(x, z)
            Y = (-constraint[0, 0] * X_xz - constraint[0, 2] * Z_xz) / constraint[0, 1]
            if (constraint == cnst_to_restrict).all():
                X_xz, Y, Z_xz = subset_surface_plot_data(X_xz, Y, Z_xz)
            if color is not None:
                if len(color) == len(constraints):
                    print('color:', color[constraint_id])
                    ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha, color=color[constraint_id])
                else:
                    print('color:', color)
                    ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha, color=color)
            else:
                ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha)
            if plot_normal_flag:
                plot_normal(constraint, X_xz, Y, Z_xz, ax=ax)
        else:
            Y_yz, Z_yz, = np.meshgrid(y, z)
            X = (-constraint[0, 1] * Y_yz - constraint[0, 2] * Z_yz) / constraint[0, 0]
            if (constraint == cnst_to_restrict).all():
                X, Y_yz, Z_yz = subset_surface_plot_data(X, Y_yz, Z_yz)
            if color is not None:
                if len(color) == len(constraints):
                    print('color:', color[constraint_id])
                    ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha, color=color[constraint_id])
                else:
                    print('color:', color)
                    ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha, color=color)
            else:
                ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha)
            if plot_normal_flag:
                plot_normal(constraint, X, Y_yz, Z_yz, ax=ax)


def flatten_list(nested_list):
    return list(itertools.chain(*nested_list))