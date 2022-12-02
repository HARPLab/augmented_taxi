import numpy as np
from scipy import special

def VMF_pdf(mu, k, p, x, dot=None):
    '''
    :param mu: mean direction
    :param k: concentration parameter
    :param p: dimensionality of the distribution (i.e. lies on the p-1 sphere)
    :param x: queried unit vectors
    :param dot: dot product between the mean direction and queried unit vector

    :return: probability density at queried unit vectors
    '''
    if dot is None:
        dot = x.dot(mu)

    return (k ** (p / 2 - 1)) / (special.iv((p / 2 - 1), k) * (2 * np.pi) ** (p / 2)) * np.exp(k * dot)
