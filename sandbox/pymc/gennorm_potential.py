import aesara.tensor as at
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from pymc.distributions.dist_math import check_parameters

from pymc.distributions import Interpolated
from scipy import stats

def from_posterior(param, samples):
    # smin, smax = np.min(samples), np.max(samples)
    # todo fixing the range for now so that the full width of the distribution is multipled
    smin = -15
    smax = 10
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])

    return Interpolated(param, x, y)

def from_posterior_modified(param, samples, pdf, beta, mu, sigma):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])

    # try and normalize the two likelihoods to allow for an approximation of a pdf
    y = y / sum(y)

    pdf_normalization = 0
    for idx, x_val in enumerate(x):
        pdf_normalization += pdf(x_val, beta, mu, sigma).eval()

    for idx, x_val in enumerate(x):
        y[idx] = (y[idx] * pdf(x_val, beta, mu, sigma).eval()) / pdf_normalization

    return Interpolated(param, x, y)


def logpdf_gennorm(value, beta, loc, scale):
    return check_parameters(
        at.log(beta / (2 * scale)) - at.gammaln(1.0 / beta) -
        (at.abs_(value - loc) / scale) ** beta, beta >= 0, scale >= 0)

def pdf_gennorm(value, beta, loc, scale):
    return check_parameters(
        beta / (2 * scale * at.gamma(1.0 / beta)) * at.exp(
        -(at.abs_(value - loc) / scale) ** beta), beta >= 0, scale >= 0)

beta = 2
sigma = 3

"""Create an MCMC trace."""
# model accumulates the objects defined within the proceeding
# context:
model = pm.Model()
with model:
    # Add random-variable x to model:
    rv_x = pm.Uniform(
        name="rv_x",
        lower=-15,
        upper=15
    )

    # Potential is a "potential term" defined as an "additional
    # tensor...to be added to the model logp"(PyMC3 developer
    # guide). In this instance, the potential is effectively
    # the model's log-likelihood.
    p = pm.Potential("sphere", logpdf_gennorm(rv_x, beta, 7.5, sigma))

    trace = pm.sample(
        5000,
        tune=1000,
        return_inferencedata=True,
        init="adapt_diag",
        progressbar=False,
        chains=4,
        # step=[pm.NUTS(target_accept=0.95)]
    )

# sns.displot(trace.posterior.rv_x.values[0], kind='kde')
# plt.show()
#
# az.plot_trace(trace)
# plt.show()

traces = [trace]

locs = [-7.5, -10]
for j in range(2):
    model = pm.Model()
    with model:
        rv_x = from_posterior("rv_x", trace.posterior.rv_x.values.flatten())
        p = pm.Potential("sphere", logpdf_gennorm(rv_x, beta, locs[j], sigma))

        # attempt A
        # rv_x = from_posterior_modified("rv_x", trace.posterior.rv_x.values.flatten(), pdf_gennorm, beta, locs[j], sigma)

        # attempt B
        # https://discourse.pymc.io/t/product-of-two-probabilities-distributions/1993 (attempt at following this suggestion)
        # p = pm.Potential("sphere", pm.logp(rv_x, trace.posterior.rv_x.values.flatten()) + logpdf_gennorm(
        #     trace.posterior.rv_x.values.flatten(), beta, locs[j], sigma))

        trace = pm.sample(
            5000,
            tune=1000,
            return_inferencedata=True,
            init="adapt_diag",
            progressbar=False,
            chains=4,
            # step=[pm.NUTS(target_accept=0.95)]
        )
        traces.append(trace)

    # sns.displot(trace.posterior.rv_x.values[0], kind='kde')
    # plt.show()
    #
    # az.plot_trace(trace)
    # plt.show()

print("Posterior distributions after " + str(len(traces)) + " iterations.")
cmap = mpl.cm.autumn
for param in ["rv_x"]:
    plt.figure(figsize=(8, 2))
    for update_i, trace in enumerate(traces):
        samples = trace.posterior[param].values.flatten()
        # smin, smax = np.min(samples), np.max(samples)
        smin = -15 # todo fixing the range for now so that the full width of the distribution is multipled
        smax = 10
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)
        plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
    # plt.axvline({"alpha": alpha_true, "beta0": beta0_true, "beta1": beta1_true}[param], c="k")
    plt.ylabel("Frequency")
    plt.legend(['likelihood1', 'posterior'])
    plt.title(param)

plt.tight_layout();
plt.show()

