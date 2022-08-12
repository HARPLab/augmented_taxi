import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
# import pymc as pm

from scipy.stats import gennorm
import seaborn as sns

class GenNormal(pm.Discrete):
    def __init__(self, mu, sigma, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.sigma = sigma
        self.beta = beta

    def logp(self, value):
        return gennorm.pdf(value, self.beta, loc=self.mu, scale=self.sigma)

    def random(self, point=None, size=None):
        # theta, lam = draw_values([self.theta, self.lam], point=point, size=size)
        # return generate_samples(genpoisson_rvs, theta=theta, lam=lam, size=size)
        return gennorm.rvs(self.beta, loc=self.mu, scale=self.sigma)


equi = GenNormal.dist(mu=np.full(5000, 0), sigma=1, beta=8).random()
# model = pm.Model()
# with model:
#     equi = pm.DensityDist('gennorm', logp=gennorm.pdf, random=gennorm.rvs)
#
sns.displot(equi, kind='kde')
plt.show()

# model = pm.Model()
# with model:
#
#     # gn1 = GenNormal.dist(mu=np.full(5000, 0), sigma=1, beta=8)
#     # gn2 = GenNormal.dist(mu=np.full(5000, 3), sigma=1, beta=8)
#
#     # y = gn1 * gn2
#
#     # trace = pm.sample(1000)
#
#     gn1 = GenNormal.dist(mu=0, sigma=1, beta=8)
#     gn2 = GenNormal.dist(mu=3, sigma=1, beta=8)
#
#
#     def joint(a, b):
#         return a * b
#
#
#     L = pm.DensityDist('L', joint, observed={'gn1': gn1, 'gn2': gn2})