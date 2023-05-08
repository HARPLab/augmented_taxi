import numpy as np
from scipy import special
from numpy.random import random
from scipy.linalg import null_space

def rand_uniform_hypersphere(N, p):
    """
        rand_uniform_hypersphere(N,p)
        =============================

        Generate random samples from the uniform distribution on the (p-1)-dimensional
        hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$. We use the method by
        Muller [1], see also Ref. [2] for other methods.

        INPUT:

            * N (int) - Number of samples
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.

    References:

    [1] Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres."
    Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.

    [2] https://mathworld.wolfram.com/SpherePointPicking.html

    """

    if (p <= 0) or (not isinstance(p, int)):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or (not isinstance(N, int)):
        raise Exception("N must be a non-zero positive integer.")

    v = np.random.normal(0, 1, (N, p))

    #    for i in range(N):
    #        v[i,:] = v[i,:]/np.linalg.norm(v[i,:])

    v = np.divide(v, np.linalg.norm(v, axis=1, keepdims=True))

    return v

def rand_t_marginal(kappa, p, N=1, halfspace=False):
    """
        rand_t_marginal(kappa,p,N=1)
        ============================

        Samples the marginal distribution of t using rejection sampling of Wood [3].

        INPUT:

            * kappa (float) - concentration
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
            * N (int) - number of samples
            * halfspace (bool) - If true, only sample from the halfspace opposite the mean vector

        OUTPUT:

            * samples (array of floats of shape (N,1)) - samples of the marginal distribution of t
    """

    # Check kappa >= 0 is numeric
    if (kappa < 0) or (not isinstance(kappa, float) and (not isinstance(kappa, int))):
        raise Exception("kappa must be a non-negative number.")

    if (p <= 0) or (not isinstance(p, int)):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or (not isinstance(N, int)):
        raise Exception("N must be a non-zero positive integer.")

    # Start of algorithm
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa ** 2 + (p - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0 ** 2)

    samples = np.zeros((N, 1))

    # Loop over number of samples
    for i in range(N):

        # Continue unil you have an acceptable sample
        while True:

            # Sample Beta distribution
            Z = np.random.beta((p - 1.0) / 2.0, (p - 1.0) / 2.0)

            # W is essentially t
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)

            # only sample from the halfspace opposite the mean vector
            if halfspace:
                if W > 0:
                    continue

            # Sample Uniform distribution
            U = np.random.uniform(low=0.0, high=1.0)

            # Check whether to accept or reject
            if kappa * W + (p - 1.0) * np.log(1.0 - x0 * W) - c >= np.log(U):
                # Accept sample
                samples[i] = W
                break

    return samples


def rand_von_mises_fisher(mu, kappa, N=1, halfspace=False):
    """
        rand_von_mises_fisher(mu,kappa,N=1)
        ===================================

        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.

        INPUT:

            * mu (array of floats of shape (p,1)) - mean direction. This should be a unit vector.
            * kappa (float) - concentration.
            * N (int) - Number of samples.

        OUTPUT:

            * samples (array of floats of shape (N,p)) - samples of the von Mises-Fisher distribution
            with mean direction mu and concentration kappa.
    """

    # Check that mu is a unit vector
    eps = 10 ** (-8)  # Precision
    norm_mu = np.linalg.norm(mu)
    if abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")

    # Check kappa >= 0 is numeric

    if (kappa < 0) or (not isinstance(kappa, float) and (not isinstance(kappa, int))):
        raise Exception("kappa must be a non-negative number.")

    # Check N>0 and is an int
    if (N <= 0) or (not isinstance(N, int)):
        raise Exception("N must be a non-zero positive integer.")

    # Dimension p
    p = len(mu)

    # Make sure that mu has a shape of px1
    mu = np.reshape(mu, (p, 1))

    # Array to store samples
    samples = np.zeros((N, p))

    #  Component in the direction of mu (Nx1)
    t = rand_t_marginal(kappa, p, N, halfspace=halfspace)

    # Component orthogonal to mu (Nx(p-1))
    xi = rand_uniform_hypersphere(N, p - 1)

    # von-Mises-Fisher samples Nxp

    # Component in the direction of mu (Nx1).
    # Note that here we are choosing an
    # intermediate mu = [1, 0, 0, 0, ..., 0] later
    # we rotate to the desired mu below
    samples[:, [0]] = t

    # Component orthogonal to mu (Nx(p-1))
    samples[:, 1:] = np.tile(np.sqrt(1 - t ** 2), (1, p - 1)) * xi

    # Rotation of samples to desired mu
    O = null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    samples = np.dot(R, samples.T).T

    return samples

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
    offset is used to choose where to sample from for all divisions. This
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