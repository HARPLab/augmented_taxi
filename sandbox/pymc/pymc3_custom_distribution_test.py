import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

from pymc3.distributions.dist_math import bound, factln, logpow
from pymc3.distributions.distribution import draw_values, generate_samples
from pymc3.theanof import intX

def genpoisson_logp(theta, lam, value):
    theta_lam_value = theta + lam * value
    log_prob = np.log(theta) + logpow(theta_lam_value, value - 1) - theta_lam_value - factln(value)

    # Probability is 0 when value > m, where m is the largest positive integer for which
    # theta + m * lam > 0 (when lam < 0).
    log_prob = tt.switch(theta_lam_value <= 0, -np.inf, log_prob)

    return bound(log_prob, value >= 0, theta > 0, abs(lam) <= 1, -theta / 4 <= lam)

def genpoisson_rvs(theta, lam, size=None):
    if size is not None:
        assert size == theta.shape
    else:
        size = theta.shape
    lam = lam[0]
    omega = np.exp(-lam)
    X = np.full(size, 0)
    S = np.exp(-theta)
    P = np.copy(S)
    for i in range(size[0]):
        U = np.random.uniform()
        while U > S[i]:
            X[i] += 1
            C = theta[i] - lam + lam * X[i]
            P[i] = omega * C * (1 + lam / C) ** (X[i] - 1) * P[i] / X[i]
            S[i] += P[i]
    return X

class GenPoisson(pm.Discrete):
    def __init__(self, theta, lam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta
        self.lam = lam
        self.mode = intX(tt.floor(theta / (1 - lam)))

    def logp(self, value):
        theta = self.theta
        lam = self.lam
        return genpoisson_logp(theta, lam, value)

    def random(self, point=None, size=None):
        theta, lam = draw_values([self.theta, self.lam], point=point, size=size)
        return generate_samples(genpoisson_rvs, theta=theta, lam=lam, size=size)

std = pm.Poisson.dist(mu=5).random(size=5000)
equi = GenPoisson.dist(theta=np.full(5000, 5), lam=0).random()
under = GenPoisson.dist(theta=np.full(5000, 5), lam=-0.5).random()
over = GenPoisson.dist(theta=np.full(5000, 5), lam=0.3).random()

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 8))
plt.setp(ax, xlim=(0, 20))

ax[0][0].hist(std, bins=np.arange(21))
ax[0][0].set_title("Standard Poisson\n($\\mu=5$)")

ax[0][1].hist(equi, bins=np.arange(21))
ax[0][1].set_title("Generalized Poisson with Equidispersion\n($\\theta=5, \\lambda=0$)")

ax[1][0].hist(under, bins=np.arange(21))
ax[1][0].set_title("Generalized Poisson with Underdispersion\n($\\theta=5, \\lambda=-0.5$)")

ax[1][1].hist(over, bins=np.arange(21))
ax[1][1].set_title("Generalized Poisson with Overdispersion\n($\\theta=5, \\lambda=0.3$)");

plt.show()
#
# try:
#     df = pd.read_csv(
#         os.path.join("..", "data", "tufts_medical_center_2020-04-29_to_2020-07-06.csv")
#     )
# except FileNotFoundError:
#     df = pd.read_csv(pm.get_data("tufts_medical_center_2020-04-29_to_2020-07-06.csv"))
#
# dates = df["date"].values
# y = df["hospitalized_total_covid_patients_suspected_and_confirmed_including_icu"].astype(float)
#
# F = 14
# T = len(y) - F
# y_tr = y[:T]
# y_va = y[-F:]
#
# with pm.Model() as model:
#     bias = pm.Normal("beta[0]", mu=0, sigma=0.1)
#     beta_recent = pm.Normal("beta[1]", mu=1, sigma=0.1)
#     rho = [bias, beta_recent]
#     sigma = pm.HalfNormal("sigma", sigma=0.1)
#     f = pm.AR("f", rho, sigma=sigma, constant=True, shape=T + F)
#
#     lam = pm.TruncatedNormal("lam", mu=0, sigma=0.1, lower=-1, upper=1)
#
#     y_past = GenPoisson("y_past", theta=tt.exp(f[:T]), lam=lam, observed=y_tr)
#
# with model:
#     trace = pm.sample(
#         50,
#         tune=20,
#         target_accept=0.99,
#         max_treedepth=15,
#         # chains=2,
#         # cores=1,
#         init="adapt_diag",
#         random_seed=42,
#     )
#
# pm.traceplot(trace);
#
# with model:
#     y_future = GenPoisson("y_future", theta=tt.exp(f[-F:]), lam=lam, shape=F)
#     forecasts = pm.sample_posterior_predictive(trace, var_names=["y_future"], random_seed=42)
# samples = forecasts["y_future"]
#
# start = date.fromisoformat(dates[-1]) - timedelta(F - 1)  # start date of forecasts
#
# low = np.zeros(F)
# high = np.zeros(F)
# median = np.zeros(F)
#
# for i in range(F):
#     low[i] = np.percentile(samples[:, i], 2.5)
#     high[i] = np.percentile(samples[:, i], 97.5)
#     median[i] = np.percentile(samples[:, i], 50)
#
# x_future = np.arange(F)
# plt.errorbar(
#     x_future,
#     median,
#     yerr=[median - low, high - median],
#     capsize=2,
#     fmt="x",
#     linewidth=1,
#     label="2.5, 50, 97.5 percentiles",
# )
# x_past = np.arange(-30, 0)
#
# plt.plot(
#     np.concatenate((x_past, x_future)), np.concatenate((y_tr[-30:], y_va)), ".", label="observed"
# )
#
# plt.xticks([-30, 0, F - 1], [start + timedelta(-30), start, start + timedelta(F - 1)])
#
# plt.legend()
# plt.title("Predicted Counts of COVID-19 Patients at Tufts Medical Center")
# plt.ylabel("Count")
# plt.show()