# script to find likelihood of sampling an incorrect response for any constraint for kappa values of the von-mises pdf.

import numpy as np
from scipy.stats import vonmises_fisher
from scipy import integrate


mu = [1, 0, 0]
x = [0, 0, 1]
kappa = 4 # a positive float
vmf = vonmises_fisher(mu, kappa)

pdf_vmf = vmf.pdf(x)

phi_lim = [0, np.pi]
theta_lim = [np.pi/2, 3*np.pi/2]    # these limits are specific to the constraint mu = [1, 0, 0]


# def int_probability_vmf_func(theta, phi):

f = lambda phi, theta: 2*kappa*np.exp(kappa*[np.cos(theta)* np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)].dot(mu))*np.sin(phi)/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))

int_probability_vmf = integrate.dblquad(f, phi_lim[0], phi_lim[1], theta_lim[0], theta_lim[1])


k = 2
# For calculating the constants I should multiply uniform (x2) and VMF (x1 * x2) pdfs by to get a new valid pdf
# pdf_vmf = 0.0438822907955184         # value of pdf of VMF at edge / border with uniform
int_probability_vmf = 0.119203      # integration of VMF over hemisphere opposite to mean
int_probability_uniform = 0.5        # integration of uniform over hemisphere
x1 = 1 / (4 * np.pi * pdf_vmf)
x2 = 1 / (int_probability_vmf * x1 + 0.5)
print(x1)
print(x2)
print(int_probability_uniform * x2)
print(int_probability_vmf * x1 * x2)










