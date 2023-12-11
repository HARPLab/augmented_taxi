# script to find likelihood of sampling an incorrect response for any constraint for kappa values of the von-mises pdf.

import numpy as np
from scipy.stats import vonmises_fisher
from scipy import integrate
from scipy import special
from scipy.optimize import fsolve


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


def calc_kappa_from_cdf(self, VMF_cdf):

        # if using u_cdf to calculate kappa
        # if u_cdf == -1:
        #     LHS = 1-self.u_cdf
        # else:
        #     LHS = 1-u_cdf

        LHS = VMF_cdf
        
        def integrand(phi, theta, kappa):
            return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) ) #this equation is for mu = [1, 0, 0], but kappa would be same for any mu for a particular cdf.

        def func(kappa):
            y, err = integrate.dblquad(integrand, np.pi/2, 3*np.pi/2, 0, np.pi, args = (kappa, )) #these limits are for mu = [1, 0, 0]
            return LHS-y
        
        return fsolve(func, 0.0001)[0]



def solve_for_kappa(u_cdf_scaled):
     
    # fixed values
    mu = np.array([1, 0, 0])
    x = np.array([0, 0, 1])
    p = 3 # length of x
    dot = x.dot(mu)
    
    # uniform_pdf = 1 / (4 * np.pi)
    u_pdf = 1 / (4 * np.pi)

    def integrand(phi, theta, kappa):
        return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) ) #this equation is for mu = [1, 0, 0], but kappa would be same for any mu for a particular cdf.

    def func_solve(vars):
        kappa, x1 = vars
        eq1 = integrate.dblquad(integrand, np.pi/2, 3*np.pi/2, 0, np.pi, args = (kappa, ))[0] * x1 - (1-u_cdf_scaled)/(2*u_cdf_scaled)
        eq2 = ((kappa ** (p / 2 - 1)) / (special.iv((p / 2 - 1), kappa) * (2 * np.pi) ** (p / 2)) * np.exp(kappa * dot)) * x1 - u_pdf

        return [eq1, eq2]

    
    return fsolve(func_solve, [0.1, 1])


    
    
# Working but not needed
def calc_kappa_from_pdf(self, pdf):
    
    LHS = pdf
    
    def func(kappa):
        mu = np.array([1, 0, 0])
        x = np.array([0, 0, 1])
        p = 3 # length of x
        dot = x.dot(mu)
        y = (kappa ** (p / 2 - 1)) / (special.iv((p / 2 - 1), kappa) * (2 * np.pi) ** (p / 2)) * np.exp(kappa * dot)
        
        return LHS-y
    
    return fsolve(func, 0.0001)[0]




if __name__ == '__main__':
    

    #### when kappa is known
    # mu = np.array([1, 0, 0])
    # x = np.array([0, 0, 1])
    # phi_lim = [0, np.pi]
    # theta_lim = [np.pi/2, 3*np.pi/2]    # these limits are specific to the constraint mu = [1, 0, 0]
    # kappa = 4 # a positive float
    # vmf = vonmises_fisher(mu, kappa)
    # pdf_vmf = vmf.pdf(x)
    
    # f = lambda phi, theta: kappa*np.exp(kappa*np.array([np.cos(theta)* np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).dot(mu))*np.sin(phi)/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
    # int_probability_vmf = integrate.dblquad(f, phi_lim[0], phi_lim[1], theta_lim[0], theta_lim[1])
    # print(int_probability_vmf)
    # int_probability_uniform = 0.5
    ################################

    ### When kappa is unknown: calculate for Kappa, x1, and x2 jointly!
    u_cdf_scaled_input = 0.51
    kappa, x1 = solve_for_kappa(u_cdf_scaled_input)
    print('kappa: ', kappa)
    print('x1_sol: ', x1)

    # for verification
    mu = np.array([1, 0, 0])
    x = np.array([0, 0, 1])
    phi_lim = [0, np.pi]
    theta_lim = [np.pi/2, 3*np.pi/2]    # these limits are specific to the constraint mu = [1, 0, 0]
    vmf = vonmises_fisher(mu, kappa)
    pdf_vmf = vmf.pdf(x)

    f = lambda phi, theta: kappa*np.exp(kappa*np.array([np.cos(theta)* np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).dot(mu))*np.sin(phi)/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
    cdf_vmf, err = integrate.dblquad(f, theta_lim[0], theta_lim[1], phi_lim[0], phi_lim[1])
    cdf_uniform = 0.5

    x1_2 = 1 / (4 * np.pi * pdf_vmf)
    x2 = 1 / (cdf_vmf * x1 + 0.5)

    print('pdf_vmf: ', pdf_vmf)
    print('pdf_uniform: ', 1/(4*np.pi))
    print('x1_sol: ', x1, 'x1_2: ', x1_2)
    print('x2: ', x2)
    print('cdf_uniform: ', cdf_uniform)
    print('cdf_vmf: ', cdf_vmf)
    print('cdf_uniform_scaled: ', cdf_uniform * x2)
    print('cdf_vmf_scaled: ', cdf_vmf * x1 * x2)






    #### Manual calculation for specific kappa values
    # # k = 2
    # # For calculating the constants I should multiply uniform (x2) and VMF (x1 * x2) pdfs by to get a new valid pdf
    # pdf_vmf = 0.0438822907955184         # value of pdf of VMF at edge / border with uniform
    # int_probability_vmf = 0.119203      # integration of VMF over hemisphere opposite to mean
    # int_probability_uniform = 0.5        # integration of uniform over hemisphere
    # x1 = 1 / (4 * np.pi * pdf_vmf)
    # x2 = 1 / (int_probability_vmf * x1 + 0.5)
    # print(x1)
    # print(x2)
    # print(int_probability_uniform * x2)
    # print(int_probability_vmf * x1 * x2)


    # k = 4
    # For calculating the constants I should multiply uniform (x2) and VMF (x1 * x2) pdfs by to get a new valid pdf
    # pdf_vmf = 0.0116640                  # value of pdf of VMF at edge / border with uniform
    # int_probability_vmf = 0.0179862      # integration of VMF over hemisphere opposite to mean
    # int_probability_uniform = 0.5        # integration of uniform over hemisphere
    # kappa = 4

    # x1 = 1 / (4 * np.pi * pdf_vmf)
    # x2 = 1 / (int_probability_vmf * x1 + 0.5)
    # print(x1)
    # print(x2)

    # print(int_probability_uniform * x2)
    # print(int_probability_vmf * x1 * x2)

    # print(VMF_pdf(mu, kappa, 3, np.array([0, 0, 1])))

    ############################################################







