from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

fs = 16

def gaussian():
    '''
    Reproduces Figure 2 of Adding Constraints to Bayesian Inverse Problems (wu_shadden_AAAI2019)
    '''
    x, y = np.mgrid[-10:10:.1, -10:10:.1]

    # prior
    pos = np.dstack((x, y))
    rv = multivariate_normal([0, 0], [[3., 0.], [0., 3.]])
    plt.subplot(2, 2, 1)
    plt.contourf(x, y, rv.pdf(pos))

    # data likelihood
    def data_likelihood(x, y):
        sigma_l = 1.
        HF = -1.5 * np.exp(-(x + 1) ** 2 - (y + 1) ** 2) - np.exp(-(x - 1) ** 2 - (y - 1) ** 2)
        return 1 / np.sqrt((2 * np.pi) ** 2 * sigma_l) * np.exp(-0.5 * (-1 - HF) * 1/sigma_l * (-1 - HF))
    plt.subplot(2, 2, 2)
    plt.contourf(x, y, data_likelihood(x, y))

    # constraint
    def constraint_likelihood(x, y):
        sigma_c = 0.1
        G = x + y - 2
        return 1 / np.sqrt((2 * np.pi) ** 2 * sigma_c) * np.exp(-0.5 * (-1 - G) * 1/sigma_c * (-1 - G))
    plt.subplot(2, 2, 3)
    plt.contourf(x, y, constraint_likelihood(x, y))

    # posterior
    def posterior(x, y):
        return data_likelihood(x, y) * constraint_likelihood(x, y) * rv.pdf(np.dstack((x, y)))
    plt.subplot(2, 2, 4)
    plt.contourf(x, y, posterior(x, y))

    plt.show()



if __name__ == "__main__":
    gaussian()

