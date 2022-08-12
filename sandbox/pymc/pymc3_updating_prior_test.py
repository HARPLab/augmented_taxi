# %matplotlib inline
import warnings

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from pymc3 import Model, Normal, Slice, sample
from pymc3.distributions import Interpolated
from scipy import stats

def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)


if __name__ == "__main__":

    plt.style.use("seaborn-darkgrid")
    print(f"Running on PyMC v{pm.__version__}")

    # Initialize random number generator
    np.random.seed(93457)

    # True parameter values
    alpha_true = 5
    beta0_true = 7
    beta1_true = 13

    # Size of dataset
    size = 100

    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2

    # Simulate outcome variable
    Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

    basic_model = Model()

    with basic_model:

        # Priors for unknown model parameters
        alpha = Normal("alpha", mu=0, sigma=1)
        beta0 = Normal("beta0", mu=12, sigma=1)
        beta1 = Normal("beta1", mu=18, sigma=1)

        # Expected value of outcome
        mu = alpha + beta0 * X1 + beta1 * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = Normal("Y_obs", mu=mu, sigma=1, observed=Y)

        # draw 1000 posterior samples
        trace = sample(1000)


    az.plot_trace(trace);
    plt.show()

    traces = [trace]

    for _ in range(3):

        # generate more data
        X1 = np.random.randn(size)
        X2 = np.random.randn(size) * 0.2
        Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

        model = Model()
        with model:
            # Priors are posteriors from previous iteration
            alpha = from_posterior("alpha", trace["alpha"])
            beta0 = from_posterior("beta0", trace["beta0"])
            beta1 = from_posterior("beta1", trace["beta1"])

            # Expected value of outcome
            mu = alpha + beta0 * X1 + beta1 * X2

            # Likelihood (sampling distribution) of observations
            Y_obs = Normal("Y_obs", mu=mu, sigma=1, observed=Y)

            # draw 10000 posterior samples
            trace = sample(1000)
            traces.append(trace)


    print("Posterior distributions after " + str(len(traces)) + " iterations.")
    cmap = mpl.cm.autumn
    for param in ["alpha", "beta0", "beta1"]:
        plt.figure(figsize=(8, 2))
        for update_i, trace in enumerate(traces):
            samples = trace[param]
            smin, smax = np.min(samples), np.max(samples)
            x = np.linspace(smin, smax, 100)
            y = stats.gaussian_kde(samples)(x)
            plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
        plt.axvline({"alpha": alpha_true, "beta0": beta0_true, "beta1": beta1_true}[param], c="k")
        plt.ylabel("Frequency")
        plt.title(param)

    plt.tight_layout();

    plt.show()

    # import seaborn as sns
    # import scipy
    # import scipy.stats
    # import numpy as np
    #
    # def bin_edges_to_center(edges):
    #     df = np.diff(edges)
    #
    #     return edges[:-1] + df / 2
    #
    # dist = scipy.stats.rayleigh(loc=2, scale=20)
    # rvs = dist.rvs(10000)
    # sns.distplot(rvs)
    #
    # # need to make a callable function of the histogram of this.
    # h, b = np.histogram(rvs, 30)
    # b = bin_edges_to_center(b)
    # plt.plot(b, h)
    #
    # f = scipy.interpolate.interp1d(b, h, bounds_error=False, fill_value=0)
    # XX = np.linspace(-1, 100, 1000)
    # plt.plot(XX, f(XX))
    #
    # with pm.Model() as model:
    #     pm.Interpolated('hist', b, h)
    #     pm.Interpolated('interp1d', XX, f(XX))
    #     trace = pm.sample(10000, target_accept=0.90)
    #
    #     pm.traceplot(trace);
    #
    #     ax = pm.plot_posterior(trace);
    #     ylim0 = ax[0].get_ylim()
    #     ax[0].plot(XX, f(XX) / f(XX).max() * ylim0[1], c='r')
    #     ax[1].plot(XX, f(XX) / f(XX).max() * ylim0[1], c='r');
    #
    #     plt.show()
    #

    # argus = scipy.stats.argus(3)
    # XX = np.linspace(0, 1, 1000)
    # YY = argus.pdf(XX)
    # plt.plot(XX, YY)
    #
    # with pm.Model() as model:
    #     pm.Interpolated('argus', XX, YY)
    #     trace = pm.sample(10000, target_accept=0.90)
    #
    #     pm.traceplot(trace);
    #     plt.show()
    #
    #     # ax = pm.plot_posterior(trace);
    #     # ylim0 = ax[0].get_ylim()
    #     # ax[0].plot(XX, YY / YY.max() * ylim0[1], c='r')