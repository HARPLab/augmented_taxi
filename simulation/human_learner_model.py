# This script contains the various human learner models

# TODO: Find the likelihood of sampling an incorrect response for any constraint for kappa values of the von-mises pdf. For now use a kappa, k = 2.

from scipy import stats
from policy_summarization import probability_utils as p_utils
import numpy as np
from scipy.stats import vonmises_fisher
from scipy import integrate
from scipy.integrate import simps
from scipy.interpolate import interpn
from policy_summarization import BEC_helpers
from scipy.spatial import KDTree
from scipy.optimize import fsolve




class cust_pdf_uniform(stats.rv_continuous):

    def __init__(self, mu, uniform_cdf):
        super().__init__()
        self.p = 3
        self.u_cdf = uniform_cdf  # probability of choosing right response
        self.mu = mu
        self.u_pdf = self.u_cdf/(2*np.pi)

        if uniform_cdf < 0.5:
            # print('Uniform_cdf: ', uniform_cdf)
            self.VMF_scaling_factor = 0.5/uniform_cdf
            self.kappa = self.calc_kappa(u_cdf=0.5) # limit VMF_cdf to 0.5 to find Kappa and appropriate scale the resulting pdf
        else:
            self.VMF_scaling_factor = 1
            self.kappa = self.calc_kappa()
        
        # print('Uniform_cdf: ', self.u_cdf)
        # print('Kappa: ', self.kappa)
        # print('VMF_scaling_factor: ', self.VMF_scaling_factor)

        self.xs_array, self.my_cdf = self.calc_cdf()



    def _pdf(self, x):

        dot = np.dot(x, self.mu)

        if dot >= 0:
            pdf = self.u_pdf
        else:
            pdf = p_utils.VMF_pdf(self.mu, self.kappa, self.p, x) * self.VMF_scaling_factor

        # print('pdf_scaled type:', type(pdf_scaled))
        return pdf



    def calc_kappa(self, u_cdf = -1):


        if u_cdf == -1:
            LHS = 1-self.u_cdf
        else:
            LHS = 1-u_cdf

        # print('LHS:', LHS)
        
        def integrand(phi, theta, kappa):
            return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) ) #this equation is for mu = [1, 0, 0], but kappa would be same for any mu for a particular cdf.


        def func(kappa):
            y, err = integrate.dblquad(integrand, np.pi/2, 3*np.pi/2, 0, np.pi, args = (kappa, )) #these limits are for mu = [1, 0, 0]
            return LHS-y
        
        return fsolve(func, 0.0001)[0]



    def calc_cdf(self):

        xs = BEC_helpers.sample_human_models_uniform([], 2500) # sample from uniform distribution
        xs = np.array(xs)

        my_pdfs = np.zeros(len(xs))
        xs_array = np.zeros([len(xs), 3])
        
        # print(my_pdfs.shape)
        # compute pdfs to be summed to form cdf
        for i in range(len(xs)):
            xs_array[i,:] = xs[i]
            my_pdfs[i] = self.pdf(xs[i])[0][0]


        # cumsum to form cdf
        my_cdf = np.cumsum(my_pdfs)
        # make sure cdf bounded on [0,1]
        # print('my_cdf: ', my_cdf)
        my_cdf = my_cdf / my_cdf[-1]
        
        return xs_array, my_cdf


    # cdf function
    def _cdf(self, x):
        return interpn(self.xs_array, self.my_cdf, x)
    
    # inverse cdf function
    def _ppf(self, query_cdf):

        nn_index = np.ones(len(query_cdf))*-1
        for i in range(len(query_cdf)):
            nn_index[i] = np.absolute(self.my_cdf - query_cdf[i]).argmin()
        
        query_x = np.zeros([len(query_cdf), 1, 3])
        for i in range(len(query_cdf)):
            query_x[i, 0, :] = self.xs_array[int(nn_index[i]), :]

        return query_x


######################################################
######################################################


class cust_pdf_kappa(stats.rv_continuous):
    
    def __init__(self, mu, kappa, p):
        super().__init__()
        self.mu = mu
        self.kappa = kappa
        self.p = p
        self.constraint = np.array([1, 0, 0])
        self.xs_array, self.my_cdf = self.calc_cdf()


    def _find_scaling_factors(self):

        # find scaling factor for the custom pdf for a specific kappa of VMF. Here we use a representative constraint and a point on the edge of the constraint to find the scaling factors.
        mu_rep = np.array([1,  0,  0])
        x_edge_rep = np.array([0,  0,  1])  # a point on the edge of the constraint mu
        kappa = self.kappa

        vmf = vonmises_fisher(mu_rep, kappa)
        pdf_vmf = vmf.pdf(x_edge_rep)

        phi_lim = [0, np.pi]
        theta_lim = [np.pi/2, 3*np.pi/2]    # these limits are specific to the constraint mu = [1, 0, 0]
        f1 = lambda phi, theta: kappa*np.exp(kappa*np.array([np.cos(theta)* np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).dot(mu_rep))*np.sin(phi)/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
        int_probability_vmf = integrate.dblquad(f1, theta_lim[0], theta_lim[1], phi_lim[0], phi_lim[1])
        x1 = 1 / (4 * np.pi * pdf_vmf)
        x2 = 1 / (int_probability_vmf[0] * x1 + 0.5)

        return x1, x2



    def _pdf(self, x):

        dot = np.dot(x, self.mu)
        x1, x2 = self._find_scaling_factors()

        if dot >= 0:
            pdf = 0.12779
            pdf_scaled = pdf * x2

        else:
            pdf = p_utils.VMF_pdf(self.mu, self.kappa, self.p, x)
            pdf_scaled = pdf * x1 * x2

        # print('pdf_scaled type:', type(pdf_scaled))
        return pdf_scaled
    

    def calc_cdf(self):
        # define domain as +/-25 sigma
        xs = BEC_helpers.sample_human_models_uniform([], 2500) # sample from uniform distribution
        xs = np.array(xs)

        my_pdfs = np.zeros(len(xs))
        xs_array = np.zeros([len(xs), 3])
        # print(my_pdfs.shape)
        # compute pdfs to be summed to form cdf
        for i in range(len(xs)):
            xs_array[i,:] = xs[i]
            my_pdfs[i] = self.pdf(xs[i])[0][0]

        # print('xs array: ', xs_array)
        # print('xs_array shape: ', xs_array.shape)

        # cumsum to form cdf
        my_cdf = np.cumsum(my_pdfs)
        # make sure cdf bounded on [0,1]
        print('my_cdf: ', my_cdf)
        my_cdf = my_cdf / my_cdf[-1]
        # print(my_cdf)
        # # create cdf and ppf
        # func_cdf = interpn(xs, my_cdf)
        # func_ppf = interpn(my_cdf, xs, fill_value='extrapolate')
        
        return xs_array, my_cdf
    
    
    # cdf function
    def _cdf(self, x):
        return interpn(self.xs_array, self.my_cdf, x)
    
    # inverse cdf function
    def _ppf(self, query_cdf):
        # return interpn(self.my_cdf, self.xs, x, fill_value='extrapolate')

        # nn_index = KDTree(self.my_cdf).query(cdf)
        # print(nn_index)
        nn_index = np.ones(len(query_cdf))*-1
        for i in range(len(query_cdf)):
            nn_index[i] = np.absolute(self.my_cdf - query_cdf[i]).argmin()
            # print(nn_index)
        
        query_x = np.zeros([len(query_cdf), 1, 3])
        for i in range(len(query_cdf)):
            query_x[i, 0, :] = self.xs_array[int(nn_index[i]), :]

        return query_x
        


    





