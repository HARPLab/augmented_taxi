import numpy as np
from scipy import special
from numpy.random import random

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

def systematic_resample(weights, N=None):
    """ Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.

    Parameters
    ----------
    weights : list-like of float
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.

    Copyright 2015 Roger R Labbe Jr. FilterPy library. http://github.com/rlabbe/filterpy
    """

    if N is None:
        N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes