# %matplotlib inline
import warnings

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from pymc import Model, Normal, Slice, sample
from pymc.distributions import Interpolated
from pymc3.distributions import Interpolated as Intepolated3
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

    for test_idx in range(3):
        print(test_idx)

        # generate more data
        X1 = np.random.randn(size)
        X2 = np.random.randn(size) * 0.2
        Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

        model = Model()
        with model:
            # Priors are posteriors from previous iteration
            alpha = from_posterior("alpha", trace.posterior.alpha.values.flatten())
            beta0 = from_posterior("beta0", trace.posterior.beta0.values.flatten())
            beta1 = from_posterior("beta1", trace.posterior.beta1.values.flatten())

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
            # samples = trace[param]
            samples = trace.posterior[param].values.flatten()
            smin, smax = np.min(samples), np.max(samples)
            x = np.linspace(smin, smax, 100)
            y = stats.gaussian_kde(samples)(x)
            plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
        plt.axvline({"alpha": alpha_true, "beta0": beta0_true, "beta1": beta1_true}[param], c="k")
        plt.ylabel("Frequency")
        plt.title(param)

    plt.tight_layout();

    plt.show()

