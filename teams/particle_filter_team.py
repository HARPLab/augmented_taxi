import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
from termcolor import colored
from spherical_geometry import great_circle_arc as gca
import copy
from numpy.random import uniform
from MeanShift import mean_shift as ms
from scipy.stats import norm
from scipy import integrate
from scipy import special
from scipy.optimize import fsolve


import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz
import policy_summarization.computational_geometry as cg
from policy_summarization import probability_utils as p_utils
from policy_summarization import particle_filter as pf
import teams.utils_teams as utils_teams
import params_team as params


# Used when particles_team_teacher class is derived from particles class
# class Particles_team(pf.Particles):
#     def __init__(self, positions):
#         super().__init__(positions)


# Particles_team is defined as a new separate class to avoid any confusion and relevant functions are copied over from the particles class
class Particles_team():
    def __init__(self, positions, learning_factor, eps=1e-5):
        self.positions = np.array(positions)
        self.weights = np.ones(len(positions)) / len(positions)
        self.eps = eps

        self.positions_prev = self.positions.copy()
        self.weights_prev = self.weights.copy()

        # parameters for the custom uniform + VMF distribution
        # fix sampling type
        sampling_flag = 'discontinuous' # 'continuous' or 'discontinuous'
        
        self.u_prob_mass_scaled = learning_factor
        self.u_pdf_scaled = self.u_prob_mass_scaled/(2*np.pi)

        self.VMF_prob_mass_scaled = 1-learning_factor  # scaled VMF probability mass/CDF
        self.VMF_kappa, self.x1  = self.solve_for_distribution_params(learning_factor, k_sol_flag = sampling_flag)
        self.VMF_pdf = self.calc_VMF_pdf()

        # for debugging purposes
        self.particles_prob_correct = 0
        self.clusters_prob_correct = 0

        

        # calculate scaling factors
        if sampling_flag == 'continuous':
            self.x2 = learning_factor/0.5
        else:
            self.x2 = 1   # only scale the VMF (x1) and not the entire VMF+uniform distribution




        # # fix the scaling factor and vary kappa according to the uniform cdf.
        # self.x1 = 6.8224855577801495
        # self.x2 = 1.6058824379694276
        # self.integral_prob_uniform = 0.8029412189847138   # the total probability on the uniform half of the custom uniform + VMF distribution
        # self.integral_prob_VMF = 0.19705878101528612      # the total probability on the VMF half of the custom uniform + VMF distribution
        # self.VMF_kappa = 4                                # the concentration parameter of the VMF distribution
        

        print('u_pdf_scaled: ', self.u_pdf_scaled, 'u_prob_mass_scaled: ', self.u_prob_mass_scaled, 'VMF_prob_mass_scaled: ', self.VMF_prob_mass_scaled, 'VMF_kappa: ', self.VMF_kappa, 'VMF_pdf: ', self.VMF_pdf, 'x1: ', self.x1, 'x2: ', self.x2) 



        self.knowledge_constraints = []

        # Spherical discretization taken from : A new method to subdivide a spherical surface into equal-area cells,
        # Malkin 2019 https://arxiv.org/pdf/1612.03467.pdf
        self.ele_bin_edges = np.array([0, 0.1721331, 0.34555774, 0.52165622, 0.69434434, 0.86288729,
                                  1.04435092, 1.20822512, 1.39251443, 1.57079633, 1.74907822,
                                  1.93336753, 2.09724173, 2.27870536, 2.44724832, 2.61993643,
                                  2.79603491, 2.96945956, 3.1416])
        self.azi_bin_edges = \
            {0: np.array([0., 2.0943951, 4.1887902, 6.28318531]),
             1: np.array([0., 0.6981317, 1.3962634, 2.0943951, 2.7925268,
                          3.4906585, 4.1887902, 4.88692191, 5.58505361, 6.28318531]),
             2: np.array([0., 0.41887902, 0.83775804, 1.25663706, 1.67551608,
                          2.0943951, 2.51327412, 2.93215314, 3.35103216, 3.76991118,
                          4.1887902, 4.60766923, 5.02654825, 5.44542727, 5.86430629,
                          6.28318531]),
             3: np.array([0., 0.31415927, 0.62831853, 0.9424778, 1.25663706,
                          1.57079633, 1.88495559, 2.19911486, 2.51327412, 2.82743339,
                          3.14159265, 3.45575192, 3.76991118, 4.08407045, 4.39822972,
                          4.71238898, 5.02654825, 5.34070751, 5.65486678, 5.96902604,
                          6.28318531]),
             4: np.array([0., 0.26179939, 0.52359878, 0.78539816, 1.04719755,
                          1.30899694, 1.57079633, 1.83259571, 2.0943951, 2.35619449,
                          2.61799388, 2.87979327, 3.14159265, 3.40339204, 3.66519143,
                          3.92699082, 4.1887902, 4.45058959, 4.71238898, 4.97418837,
                          5.23598776, 5.49778714, 5.75958653, 6.02138592, 6.28318531]),
             5: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                          1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                          2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                          3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                          4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                          5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                          6.28318531]),
             6: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                          1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                          2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                          3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                          4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                          5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                          6.28318531]),
             7: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                          0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                          1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                          2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                          3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                          4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                          5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                          6.10865238, 6.28318531]),
             8: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                          0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                          1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                          2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                          3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                          4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                          5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                          6.10865238, 6.28318531]),
             9: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                          0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                          1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                          2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                          3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                          4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                          5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                          6.10865238, 6.28318531]),
             10: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                           0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                           1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                           2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                           3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                           4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                           5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                           6.10865238, 6.28318531]),
             11: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                           1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                           2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                           3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                           4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                           5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                           6.28318531]),
             12: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                           1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                           2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                           3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                           4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                           5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                           6.28318531]),
             13: np.array([0., 0.26179939, 0.52359878, 0.78539816, 1.04719755,
                           1.30899694, 1.57079633, 1.83259571, 2.0943951, 2.35619449,
                           2.61799388, 2.87979327, 3.14159265, 3.40339204, 3.66519143,
                           3.92699082, 4.1887902, 4.45058959, 4.71238898, 4.97418837,
                           5.23598776, 5.49778714, 5.75958653, 6.02138592, 6.28318531]),
             14: np.array([0., 0.31415927, 0.62831853, 0.9424778, 1.25663706,
                           1.57079633, 1.88495559, 2.19911486, 2.51327412, 2.82743339,
                           3.14159265, 3.45575192, 3.76991118, 4.08407045, 4.39822972,
                           4.71238898, 5.02654825, 5.34070751, 5.65486678, 5.96902604,
                           6.28318531]),
             15: np.array([0., 0.41887902, 0.83775804, 1.25663706, 1.67551608,
                           2.0943951, 2.51327412, 2.93215314, 3.35103216, 3.76991118,
                           4.1887902, 4.60766923, 5.02654825, 5.44542727, 5.86430629,
                           6.28318531]),
             16: np.array([0., 0.6981317, 1.3962634, 2.0943951, 2.7925268,
                           3.4906585, 4.1887902, 4.88692191, 5.58505361, 6.28318531]),
             17: np.array([0., 2.0943951, 4.1887902, 6.28318531])}

        self.ele_bin_edges_20 = np.array([0, np.pi/4, np.pi/2, 3 * np.pi/4, np.pi + eps])

        self.azi_bin_edges_20 = \
            {0: np.array([0, 2 * np.pi/3, 4 * np.pi/3, 2 * np.pi + eps]),
             1: np.array([0, 2 * np.pi/7, 4 * np.pi/7, 6 * np.pi/7, 8 * np.pi/7, 10 * np.pi/7, 12 * np.pi/7, 2 * np.pi + eps]),
             2: np.array([0, 2 * np.pi/7, 4 * np.pi/7, 6 * np.pi/7, 8 * np.pi/7, 10 * np.pi/7, 12 * np.pi/7, 2 * np.pi + eps]),
             3: np.array([0, 2 * np.pi/3, 4 * np.pi/3, 2 * np.pi + eps])}

        self.binned = False

        self.cluster_centers = None
        self.cluster_weights = None
        self.cluster_assignments = None

        self.bin_neighbor_mapping = self.initialize_bin_neighbor_mapping()
        self.bin_neighbor_mapping_20 = self.initialize_bin_neighbor_mapping_20()
        self.bin_particle_mapping = None
        self.bin_weight_mapping = None

        # newly added to check reset count
        self.pf_reset_count = 0
        self.reset_constraint = []

    def calc_kappa_from_cdf(self, VMF_cdf):
        phi_lim = [0, np.pi]
        theta_lim = [np.pi/2, 3*np.pi/2]

        LHS = VMF_cdf
        
        def integrand(phi, theta, kappa):
            return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) ) #this equation is for mu = [1, 0, 0], but kappa would be same for any mu for a particular cdf.

        def func(kappa):
            y, err = integrate.dblquad(integrand, theta_lim[0], theta_lim[1], phi_lim[0], phi_lim[1], args = (kappa, )) #these limits are for mu = [1, 0, 0]; limits for 2nd variable first
            return LHS-y
        
        return fsolve(func, 0.0001)[0]


    def solve_for_distribution_params(self, u_cdf_scaled, k_sol_flag = 'discontinuous'):
        
        if k_sol_flag == 'continuous':
            kappa, x1 =  self.solve_for_kappa_continuous(u_cdf_scaled)
        else:
            kappa, x1 =  self.solve_for_kappa_discontinous(u_cdf_scaled)

        return kappa, x1
        



    def solve_for_kappa_continuous(self, u_cdf_scaled):
        
        # fixed values
        mu = np.array([1, 0, 0])
        x = np.array([0, 0, 1])
        p = 3 # length of x
        dot = x.dot(mu)
        phi_lim = [0, np.pi]
        theta_lim = [np.pi/2, 3*np.pi/2]

        if u_cdf_scaled > 0.5:
            u_pdf = 1 / (4 * np.pi)

            def integrand(phi, theta, kappa):
                return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) ) #this equation is for mu = [1, 0, 0], but kappa would be same for any mu for a particular cdf.

            def func_solve(vars):
                kappa, x1 = vars
                eq1 = integrate.dblquad(integrand, theta_lim[0], theta_lim[1], phi_lim[0], phi_lim[1], args = (kappa, ))[0] * x1 - (1-u_cdf_scaled)/(2*u_cdf_scaled)  # limits for 2nd variable first
                eq2 = ((kappa ** (p / 2 - 1)) / (special.iv((p / 2 - 1), kappa) * (2 * np.pi) ** (p / 2)) * np.exp(kappa * dot)) * x1 - u_pdf

                return [eq1, eq2]
            
            return fsolve(func_solve, [0.1, 1])
            
        else:
            print('u_cdf_scaled should be greater than 0.5')
            return [None, None]
            

            
    def solve_for_kappa_discontinous(self, u_cdf_scaled):
        

        # fixed values
        mu = np.array([1, 0, 0])
        x = np.array([0, 0, 1])
        phi_lim = [0, np.pi]
        theta_lim = [np.pi/2, 3*np.pi/2]
        
        if u_cdf_scaled > 0.5:
            kappa = self.calc_kappa_from_cdf(1-u_cdf_scaled)
            x1 = 1
        else:
            kappa = self.calc_kappa_from_cdf(0.49)
            f = lambda phi, theta: kappa*np.exp(kappa*np.array([np.cos(theta)* np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).dot(mu))*np.sin(phi)/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
            VMF_cdf, err = integrate.dblquad(f, theta_lim[0], theta_lim[1], phi_lim[0], phi_lim[1])  # limits for 2nd argument first; https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html
            print('VMF_cdf: ', VMF_cdf)
            x1 = (1-u_cdf_scaled)/VMF_cdf

        
        return kappa, x1


    def calc_VMF_pdf(self):
        mu = np.array([1, 0, 0])
        x = np.array([0, 0, 1])
        dot = x.dot(mu)
        p = 3 # length of x
        k = self.VMF_kappa
        
        return (k ** (p / 2 - 1)) / (special.iv((p / 2 - 1), k) * (2 * np.pi) ** (p / 2)) * np.exp(k * dot)



    # def solve_for_distribution_params(self, u_cdf_scaled, k_sol_flag = 'discontinuous'):
        
    #     if k_sol_flag == 'continuous':
    #         kappa, x1 =  self.solve_for_kappa_continuous(u_cdf_scaled)
    #     else:
    #         kappa, x1 =  self.solve_for_kappa_discontinous(u_cdf_scaled)


    #     return kappa, x1



    # def solve_for_kappa_continuous(self, u_cdf_scaled):
        
    #     # fixed values
    #     mu = np.array([1, 0, 0])
    #     x = np.array([0, 0, 1])
    #     p = 3 # length of x
    #     dot = x.dot(mu)
        

    #     if u_cdf_scaled > 0.5:
    #         # uniform_pdf = 1 / (4 * np.pi)
    #         u_pdf = 1 / (4 * np.pi)

    #         def integrand(phi, theta, kappa):
    #             return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) ) #this equation is for mu = [1, 0, 0], but kappa would be same for any mu for a particular cdf.

    #         def func_solve(vars):
    #             kappa, x1 = vars
    #             eq1 = integrate.dblquad(integrand, np.pi/2, 3*np.pi/2, 0, np.pi, args = (kappa, ))[0] * x1 - (1-u_cdf_scaled)/(2*u_cdf_scaled)
    #             eq2 = ((kappa ** (p / 2 - 1)) / (special.iv((p / 2 - 1), kappa) * (2 * np.pi) ** (p / 2)) * np.exp(kappa * dot)) * x1 - u_pdf

    #             return [eq1, eq2]
            

    #         return fsolve(func_solve, [0.1, 1])
            
    #     else:
    #         print('u_cdf_scaled should be greater than 0.5')
    #         return None, None
            

        
    # def solve_for_kappa_discontinous(self, u_cdf_scaled):
        

    #     # fixed values
    #     mu = np.array([1, 0, 0])
    #     x = np.array([0, 0, 1])
    #     p = 3 # length of x
    #     dot = x.dot(mu)
    #     phi_lim = [0, np.pi]
    #     theta_lim = [np.pi/2, 3*np.pi/2]
        
        
    #     if u_cdf_scaled > 0.5:
    #         kappa = self.calc_kappa_from_cdf(1-u_cdf_scaled)
    #         x1 = 1
    #     else:
    #         prob_mass = 0.49
    #         kappa = self.calc_kappa_from_cdf(prob_mass)  # use a probability mass that is close to uniform. This suggests that there is no learning (everything is equally bad on this side of the constraint) and then scale it according to the required prob mass
    #         # f = lambda phi, theta: kappa*np.exp(kappa*np.array([np.cos(theta)* np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).dot(mu))*np.sin(phi)/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
    #         f = lambda phi, theta: ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) )
    #         VMF_cdf, err = integrate.dblquad(f, phi_lim[0], phi_lim[1], theta_lim[0], theta_lim[1])
    #         print('VMF_cdf: ', VMF_cdf)
    #         x1 = (1-u_cdf_scaled)/prob_mass

        
    #     return kappa, x1
    #######################################################################

    # # Calculates Kappa with variable scaling factors - most accurate
    # def solve_for_kappa(self, u_cdf_scaled):
    #     # Note u_cdf_scaled > 0.5; u_cdf_scaled = 0.5 results in kappa = 0, fully uniform on both sides of the sphere.
        
    #     # fixed values
    #     mu = np.array([1, 0, 0])
    #     x = np.array([0, 0, 1])
    #     p = 3 # length of x
    #     dot = x.dot(mu)
        
    #     # uniform_pdf = 1 / (4 * np.pi)
    #     u_pdf = 1 / (4 * np.pi)

    #     def integrand(phi, theta, kappa):
    #         return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) ) #this equation is for mu = [1, 0, 0], but kappa would be same for any mu for a particular cdf.

    #     def func_solve(vars):
    #         kappa, x1 = vars
    #         eq1 = integrate.dblquad(integrand, np.pi/2, 3*np.pi/2, 0, np.pi, args = (kappa, ))[0] * x1 - (1-u_cdf_scaled)/(2*u_cdf_scaled)
    #         eq2 = ((kappa ** (p / 2 - 1)) / (special.iv((p / 2 - 1), kappa) * (2 * np.pi) ** (p / 2)) * np.exp(kappa * dot)) * x1 - u_pdf

    #         return [eq1, eq2]

        
    #     return fsolve(func_solve, [0.1, 1])

    
    # # Working for fixed scaling factors, but not needed   
    # def calc_kappa_from_pdf(self, pdf):
        
    #     LHS = pdf
        
    #     def func(kappa):
    #         mu = np.array([1, 0, 0])
    #         x = np.array([0, 0, 1])
    #         p = 3 # length of x
    #         dot = x.dot(mu)
    #         y = (kappa ** (p / 2 - 1)) / (special.iv((p / 2 - 1), kappa) * (2 * np.pi) ** (p / 2)) * np.exp(kappa * dot)
            
    #         return LHS-y
        
    #     return fsolve(func, 0.0001)[0]
    

    def reinitialize(self, positions):
        self.positions = np.array(positions)
        self.weights = np.ones(len(positions)) / len(positions)

    def initialize_bin_neighbor_mapping(self):
        '''
        Calculate and store the neighboring bins in a 406-cell discretization of the 2-sphere
        Discretization provided by A new method to subdivide a spherical surface into equal-area cells
        https://arxiv.org/pdf/1612.03467.pdf
        '''
        # elements are (elevation_bin, azimuth_bin) pairs
        bin_neighbor_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(9)], 2: [[] for _ in range(15)], 3: [[] for _ in range(20)], 4: [[] for _ in range(24)],
                       5: [[] for _ in range(30)],
                       6: [[] for _ in range(30)], 7: [[] for _ in range(36)], 8: [[] for _ in range(36)], 9: [[] for _ in range(36)], 10: [[] for _ in range(36)],
                       11: [[] for _ in range(30)], 12: [[] for _ in range(30)], 13: [[] for _ in range(24)], 14: [[] for _ in range(20)], 15: [[] for _ in range(15)],
                       16: [[] for _ in range(9)], 17: [[] for _ in range(3)]}

        for elevation_bin_idx in bin_neighbor_mapping.keys():
            if elevation_bin_idx == 0 or elevation_bin_idx == 17:
                continue

            n_azimuth_bins = len(bin_neighbor_mapping[elevation_bin_idx])
            for azimuth_bin_idx, _ in enumerate(bin_neighbor_mapping[elevation_bin_idx]):
                if azimuth_bin_idx == 0:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, n_azimuth_bins - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])
                    # top left & bottom left
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1), (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])

                    # considering top & top right
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < self.azi_bin_edges[elevation_bin_idx - 1][azimuth_bin_idx + 1]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0)])
                    else:
                        # else add the top and top right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0), (elevation_bin_idx - 1, 1)])

                    # considering bottom & bottom right
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < self.azi_bin_edges[elevation_bin_idx + 1][azimuth_bin_idx + 1]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx + 1, 0)])
                    else:
                        # else add the bottom and bottom right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, 0), (elevation_bin_idx + 1, 1)])
                elif azimuth_bin_idx == n_azimuth_bins - 1:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, 0)])

                    # top right & bottom right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0), (elevation_bin_idx + 1, 0)])

                    # considering top left & top left
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx] - self.eps > self.azi_bin_edges[elevation_bin_idx - 1][-2]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1)])
                    else:
                        # else add the top and top left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1), (elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 2)])

                    # considering bottom & bottom left
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx] - self.eps > self.azi_bin_edges[elevation_bin_idx + 1][-2]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])
                    else:
                        # else add the bottom and bottom left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1), (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 2)])
                else:
                    # left and right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])

                    left_edge = self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx]
                    right_edge = self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx + 1]
                    # top
                    top_bin_left_edge = np.digitize(left_edge - self.eps, self.azi_bin_edges[elevation_bin_idx - 1])
                    top_bin_right_edge = np.digitize(right_edge + self.eps, self.azi_bin_edges[elevation_bin_idx - 1])
                    if top_bin_left_edge == top_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, top_bin_right_edge - 1)])
                    else:
                        for x in range(top_bin_left_edge, top_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx - 1, x - 1)])
                    # bottom
                    bottom_bin_left_edge = np.digitize(left_edge - self.eps,
                                self.azi_bin_edges[elevation_bin_idx + 1])
                    bottom_bin_right_edge = np.digitize(right_edge + self.eps,
                                      self.azi_bin_edges[elevation_bin_idx + 1])
                    if bottom_bin_left_edge == bottom_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, bottom_bin_right_edge - 1)])
                    else:
                        for x in range(bottom_bin_left_edge, bottom_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx + 1, x - 1)])

        # handle corner cases
        bin_neighbor_mapping[0][0].extend([(1, 0), (1, 1), (1, 2), (0, 1), (0, 2), (1, 3), (1, 8)])
        bin_neighbor_mapping[0][1].extend([(1, 3), (1, 4), (1, 5), (0, 0), (0, 2), (1, 2), (1, 6)])
        bin_neighbor_mapping[0][2].extend([(1, 6), (1, 7), (1, 8), (0, 0), (0, 1), (1, 5), (1, 0)])

        bin_neighbor_mapping[17][0].extend([(16, 0), (16, 1), (16, 2), (17, 1), (17, 2), (16, 8), (16, 3)])
        bin_neighbor_mapping[17][1].extend([(16, 3), (16, 4), (16, 5), (17, 0), (17, 2), (16, 2), (16, 6)])
        bin_neighbor_mapping[17][2].extend([(16, 6), (16, 7), (16, 8), (17, 0), (17, 1), (16, 5), (16, 0)])

        return bin_neighbor_mapping

    def initialize_bin_neighbor_mapping_20(self):
        '''
        Calculate and store the neighboring bins in a 20-cell discretization of the 2-sphere
        Discretization provided by A New Equal-area Isolatitudinal Grid on a Spherical Surface
        https://iopscience.iop.org/article/10.3847/1538-3881/ab3a44/pdf
        '''

        # elements are (elevation_bin, azimuth_bin) pairs
        bin_neighbor_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(7)], 2: [[] for _ in range(7)],
                                3: [[] for _ in range(3)]}

        for elevation_bin_idx in bin_neighbor_mapping.keys():
            if elevation_bin_idx == 0 or elevation_bin_idx == 3:
                continue

            n_azimuth_bins = len(bin_neighbor_mapping[elevation_bin_idx])
            for azimuth_bin_idx, _ in enumerate(bin_neighbor_mapping[elevation_bin_idx]):
                if azimuth_bin_idx == 0:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, n_azimuth_bins - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])
                    # top left & bottom left
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1),
                         (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])

                    # considering top & top right
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < \
                            self.azi_bin_edges_20[elevation_bin_idx - 1][azimuth_bin_idx + 1]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0)])
                    else:
                        # else add the top and top right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, 0), (elevation_bin_idx - 1, 1)])

                    # considering bottom & bottom right
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < \
                            self.azi_bin_edges_20[elevation_bin_idx + 1][azimuth_bin_idx + 1]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx + 1, 0)])
                    else:
                        # else add the bottom and bottom right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, 0), (elevation_bin_idx + 1, 1)])
                elif azimuth_bin_idx == n_azimuth_bins - 1:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, 0)])

                    # top right & bottom right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx - 1, 0), (elevation_bin_idx + 1, 0)])

                    # considering top left & top left
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx] - self.eps > \
                            self.azi_bin_edges_20[elevation_bin_idx - 1][-2]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1)])
                    else:
                        # else add the top and top left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1),
                             (elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 2)])

                    # considering bottom & bottom left
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx] - self.eps > \
                            self.azi_bin_edges_20[elevation_bin_idx + 1][-2]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])
                    else:
                        # else add the bottom and bottom left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1),
                             (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 2)])
                else:
                    # left and right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])

                    left_edge = self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx]
                    right_edge = self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx + 1]
                    # top
                    top_bin_left_edge = np.digitize(left_edge - self.eps, self.azi_bin_edges_20[elevation_bin_idx - 1])
                    top_bin_right_edge = np.digitize(right_edge + self.eps, self.azi_bin_edges_20[elevation_bin_idx - 1])
                    if top_bin_left_edge == top_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, top_bin_right_edge - 1)])
                    else:
                        for x in range(top_bin_left_edge, top_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx - 1, x - 1)])
                    # bottom
                    bottom_bin_left_edge = np.digitize(left_edge - self.eps,
                                                       self.azi_bin_edges_20[elevation_bin_idx + 1])
                    bottom_bin_right_edge = np.digitize(right_edge + self.eps,
                                                        self.azi_bin_edges_20[elevation_bin_idx + 1])
                    if bottom_bin_left_edge == bottom_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, bottom_bin_right_edge - 1)])
                    else:
                        for x in range(bottom_bin_left_edge, bottom_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx + 1, x - 1)])

        # handle corner cases
        bin_neighbor_mapping[0][0].extend([(1, 0), (1, 1), (1, 2), (0, 1), (0, 2), (1, 6)])
        bin_neighbor_mapping[0][1].extend([(1, 2), (1, 3), (1, 4), (0, 0), (0, 2)])
        bin_neighbor_mapping[0][2].extend([(1, 4), (1, 5), (1, 6), (1, 0), (0, 1), (0, 0)])

        bin_neighbor_mapping[3][0].extend([(2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (2, 6)])
        bin_neighbor_mapping[3][1].extend([(2, 2), (2, 3), (2, 4), (3, 0), (3, 2)])
        bin_neighbor_mapping[3][2].extend([(2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (2, 0)])

        return bin_neighbor_mapping

    def bin_particles(self):
        '''
        Bin a particle into one of the 406-cell discretization of the 2-sphere
        Discretization provided by A new method to subdivide a spherical surface into equal-area cells
        https://arxiv.org/pdf/1612.03467.pdf
        '''

        # sort particles into bins
        positions_spherical = cg.cart2sph(self.positions.squeeze())

        # dictionary with keys based on elevation, and values based on associated number of azimuth bins
        bin_weight_mapping = {0: np.zeros(3), 1: np.zeros(9), 2: np.zeros(15), 3: np.zeros(20), 4: np.zeros(24),
                       5: np.zeros(30),
                       6: np.zeros(30), 7: np.zeros(36), 8: np.zeros(36), 9: np.zeros(36), 10: np.zeros(36),
                       11: np.zeros(30), 12: np.zeros(30), 13: np.zeros(24), 14: np.zeros(20), 15: np.zeros(15),
                       16: np.zeros(9), 17: np.zeros(3)}

        # contains the indices of points contained within each bin
        bin_particle_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(9)], 2: [[] for _ in range(15)], 3: [[] for _ in range(20)], 4: [[] for _ in range(24)],
                       5: [[] for _ in range(30)],
                       6: [[] for _ in range(30)], 7: [[] for _ in range(36)], 8: [[] for _ in range(36)], 9: [[] for _ in range(36)], 10: [[] for _ in range(36)],
                       11: [[] for _ in range(30)], 12: [[] for _ in range(30)], 13: [[] for _ in range(24)], 14: [[] for _ in range(20)], 15: [[] for _ in range(15)],
                       16: [[] for _ in range(9)], 17: [[] for _ in range(3)]}

        elevations, azimuths = zip(*positions_spherical)

        # sort the points into elevation bins
        elevation_bins = np.digitize(elevations, self.ele_bin_edges)

        # for each point, sort into the correct azimuth bin
        for j, elevation_bin in enumerate(elevation_bins):
            azimuth_bin = np.digitize(azimuths[j], self.azi_bin_edges[elevation_bin - 1])
            bin_weight_mapping[elevation_bin - 1][azimuth_bin - 1] += self.weights[j]

            bin_particle_mapping[elevation_bin - 1][azimuth_bin - 1].append(j)

        return bin_particle_mapping, bin_weight_mapping

    def bin_particles_20(self):
        '''
        Bin a particle into one of the 20-cell discretization of the 2-sphere
        Discretization provided by A New Equal-area Isolatitudinal Grid on a Spherical Surface
        https://iopscience.iop.org/article/10.3847/1538-3881/ab3a44/pdf
        '''

        # sort particles into bins
        positions_spherical = cg.cart2sph(self.positions.squeeze())

        # dictionary with keys based on elevation, and values based on associated number of azimuth bins
        bin_weight_mapping = {0: np.zeros(3), 1: np.zeros(7), 2: np.zeros(7), 3: np.zeros(3)}

        # contains the indices of points contained within each bin
        bin_particle_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(7)], 2: [[] for _ in range(7)], 3: [[] for _ in range(3)]}

        elevations, azimuths = zip(*positions_spherical)

        # sort the points into elevation bins
        elevation_bins = np.digitize(elevations, self.ele_bin_edges_20)

        # for each point, sort into the correct azimuth bin
        for j, elevation_bin in enumerate(elevation_bins):
            azimuth_bin = np.digitize(azimuths[j], self.azi_bin_edges_20[elevation_bin - 1])
            bin_weight_mapping[elevation_bin - 1][azimuth_bin - 1] += self.weights[j]

            # record indices of points belonging to each bin
            bin_particle_mapping[elevation_bin - 1][azimuth_bin - 1].append(j)

        return bin_particle_mapping, bin_weight_mapping

    def meanshift_plusplus_neighbors(self, query_point, points, weights):
        '''
        Implement "MeanShift++: Extremely Fast Mode-Seeking With Applications to Segmentation and Object Tracking" by
        Jang et al. from CVPR 2021 on 2-sphere
        '''
        # bin this point
        query_point_sph = cg.cart2sph(query_point)
        query_elevation_bin = np.digitize(query_point_sph[0][0], self.ele_bin_edges)
        query_azimuth_bin = np.digitize(query_point_sph[0][1], self.azi_bin_edges[query_elevation_bin - 1])

        # find neighbors
        neighbor_bins = self.bin_neighbor_mapping[query_elevation_bin - 1][query_azimuth_bin - 1]

        # obtain points of neighbors
        neighboring_particles = []
        neighboring_weights = []
        for neighbor_bin in neighbor_bins:
            elevation_bin, azimuth_bin = neighbor_bin
            particle_idxs = self.bin_particle_mapping[elevation_bin][azimuth_bin]
            for idx in particle_idxs:
                neighboring_particles.append(points[idx])
                neighboring_weights.append(weights[idx])

        # obtain points in this bin as well
        particle_idxs = self.bin_particle_mapping[query_elevation_bin - 1][query_azimuth_bin - 1]
        for idx in particle_idxs:
            neighboring_particles.append(points[idx])
            neighboring_weights.append(weights[idx])

        return np.array(neighboring_particles), np.array(neighboring_weights)

    def cluster(self):
        if self.binned == False:
            bin_particle_mapping, bin_weight_mapping = self.bin_particles()
            self.bin_particle_mapping = bin_particle_mapping
            self.bin_weight_mapping = bin_weight_mapping
            self.binned = True

        # cluster particles using mean-shift and store the cluster centers
        mean_shifter = ms.MeanShift()
        # only use a subset of neighboring points to perform meanshift clustering
        # mean_shift_result = mean_shifter.cluster(self.positions.squeeze(), weights=self.weights, downselect_points=self.meanshift_plusplus_neighbors)
        # use all points to perform meanshift clustering
        mean_shift_result = mean_shifter.cluster(self.positions.squeeze(), weights=self.weights)
        self.cluster_centers = mean_shift_result.cluster_centers
        self.cluster_assignments = mean_shift_result.cluster_assignments

        # assign weights to cluster centers by summing up the weights of constituent particles
        cluster_weights = []
        for j, cluster_center in enumerate(self.cluster_centers):
            cluster_weights.append(sum(self.weights[np.where(mean_shift_result.cluster_assignments == j)[0]]))
        self.cluster_weights = cluster_weights

    
    def plot(self, centroid=None, fig=None, ax=None, cluster_centers=None, cluster_weights=None,
                       cluster_assignments=None, plot_prev=False):
        if plot_prev:
            particle_positions = self.positions_prev
            particle_weights = self.weights_prev
        else:
            particle_positions = self.positions
            particle_weights = self.weights

        vis_scale_factor = 10 * len(particle_positions) * (1/sum(particle_weights))
        if fig == None:
            fig = plt.figure()
        if ax == None:
            ax = fig.add_subplot(projection='3d')

        if cluster_centers is not None:
            for cluster_id, cluster_center in enumerate(cluster_centers):
                ax.scatter(cluster_center[0][0], cluster_center[0][1], cluster_center[0][2],
                           s=500 * cluster_weights[cluster_id], c='red', marker='+')
                        #    s=50, c='red', marker='+')

            # print("# of clusters: {}".format(len(np.unique(cluster_assignments))))

        if cluster_assignments is not None:
            # color the particles according to their cluster assignments if provided
            plt.set_cmap("gist_rainbow")
            ax.scatter(particle_positions[:, 0, 0], particle_positions[:, 0, 1], particle_positions[:, 0, 2],
                       s=particle_weights * vis_scale_factor, c=cluster_assignments)
        else:

            ax.scatter(particle_positions[:, 0, 0], particle_positions[:, 0, 1], particle_positions[:, 0, 2],
                       s=particle_weights * vis_scale_factor, color='tab:blue')

        # if centroid == None:
        #     centroid = cg.spherical_centroid(particle_positions.squeeze().T, particle_weights)
        # ax.scatter(centroid[0], centroid[1], centroid[2], marker='o', c='r', s=100)

        if matplotlib.get_backend() == 'TkAgg':
            ax.set_xlabel('$\mathregular{w_0}$: Mud')
            ax.set_ylabel('$\mathregular{w_1}$: Recharge')
            ax.set_zlabel('$\mathregular{w_2}$: Action')



    def resample_from_index(self, indexes, K=1):
        
        original_positions = copy.deepcopy(self.positions)

        # resample
        self.positions = self.positions[indexes]

        # sort particles into bins
        positions_spherical = cg.cart2sph(self.positions.squeeze())
        elevations = positions_spherical[:, 0]
        azimuths = positions_spherical[:, 1]

        ############## Calculate maximum noise
        # # Method 1: use the maximum distance between any two particles in the current set of particles
        # max_ele_dist = max(elevations) - min(elevations)
        # max_azi_dist = max(azimuths) - min(azimuths)  # 12/10/23. Added to increase noise more equally in both elevation and azimuth directions
        # ###
        
        # Method 2: use the maximum distance between any two CONSECUTIVE particles in the original set of particles
        azimuths_sorted = np.sort(azimuths)
        azi_dists = np.empty(len(azimuths))
        azi_dists[0:-1] = np.diff(azimuths_sorted)
        azi_dists[-1] = min(2 * np.pi - (max(azimuths_sorted) - min(azimuths_sorted)), max(azimuths_sorted) - min(azimuths_sorted))

        if np.std(azi_dists[azi_dists > self.eps]) < 0.01 and np.std(azimuths_sorted) > 1:
            # the particles are relatively evenly spaced out across the full range of azimuth
            max_azi_dist = 2 * np.pi
        else:
            # take the largest gap/azimuth distance between two consecutive particles
            max_azi_dist = max(azi_dists)

        ######## 12/10/23. added to make noise more isotropic, based on elevation differences between points similar to azimuth
        elevations_sorted = np.sort(elevations)
        ele_dists = np.empty(len(elevations))
        ele_dists[0:-1] = np.diff(elevations_sorted)
        ele_dists[-1] = min(np.pi - (max(elevations_sorted) - min(elevations_sorted)), max(elevations_sorted) - min(elevations_sorted))

        if np.std(ele_dists[ele_dists > self.eps]) < 0.01 and np.std(elevations_sorted) > 1:
            # the particles are relatively evenly spaced out across the full range of elevation
            max_ele_dist = np.pi
        else:
            # take the largest gap/elevation distance between two consecutive particles
            max_ele_dist = max(ele_dists)
        ###

        ##### Method 2b 12/17/23. Added to make noise more isotropic.
        max_azi_dist = max(max_azi_dist, max_ele_dist)  # make azimuth atleast as wide as elevation
        
        ########################################

        # noise suggested by "Novel approach to nonlinear/non-Gaussian Bayesian state estimation" by Gordon et al.
        print(colored("max_ele_dist: " + str(max_ele_dist) + ". max_azi_dist: " + str(max_azi_dist), "red"))

        noise = np.array([np.random.normal(scale=max_ele_dist, size=len(positions_spherical)),
                          np.random.normal(scale=max_azi_dist, size=len(positions_spherical))]).T
        
        noise_before_transform = copy.deepcopy(noise)

        noise *= K * positions_spherical.shape[0] ** (-1/positions_spherical.shape[1])

        ########### added for debugging purposes - 12/10
        positions_before_noise = np.empty_like(self.positions)
        positions_spherical_before_noise = copy.deepcopy(positions_spherical)
        for aa in range(0, len(positions_spherical)):
            positions_before_noise[aa, :] = np.array(cg.sph2cart(positions_spherical_before_noise[aa, :])).reshape(1, -1)
        #################


        positions_spherical += noise

        for j in range(0, len(positions_spherical)):
            self.positions[j, :] = np.array(cg.sph2cart(positions_spherical[j, :])).reshape(1, -1)

        # reset the weights
        self.weights = np.ones(len(self.positions)) / len(self.positions)


        # ###### plots for debugging - 12/10/23 #######
        # # plot particles
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        # ax2 = fig.add_subplot(1, 3, 2, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
        # ax3 = fig.add_subplot(1, 3, 3, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
        # ax1.title.set_text('Original_positions')
        # ax2.title.set_text('resampled particles before noise')
        # ax3.title.set_text('Resampled particles after noise')
        # ax1.scatter(original_positions[:, 0, 0], original_positions[:, 0, 1], original_positions[:, 0, 2])
        # ax2.scatter(positions_before_noise[:, 0, 0], positions_before_noise[:, 0, 1], positions_before_noise[:, 0, 2])
        # ax3.scatter(self.positions[:, 0, 0], self.positions[:, 0, 1], self.positions[:, 0, 2])


        # # plt.figure()
        # # plt.scatter(noise[:,0], noise[:,1])

        # fig2, axn = plt.subplots(1, 2)
        # axn[0].scatter(noise_before_transform[:, 0], noise_before_transform[:, 1])
        # axn[1].scatter(noise[:, 0], noise[:, 1])
        # axn[0].title.set_text('noise before transform')
        # axn[1].title.set_text('noise after transform')

        
        # # fig3, axn2 = plt.subplots(1, 2)
        # # axn2[0].plot(np.arange(1, len(elevations_sorted)+1), elevations_sorted)
        # # axn2[1].plot(np.arange(1, len(azimuths_sorted)+1), azimuths_sorted)
        # # axn2[0].title.set_text('elevations_sorted')
        # # axn2[1].title.set_text('azimuths_sorted')
        # # # print("ele_dists: " + str(ele_dists))


        # plt.show()







    # http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    def normalized_weighted_variance(self, spherical_centroid=None):
        sum_weights = np.sum(self.weights)

        eff_dof = (sum_weights ** 2) / (np.sum(self.weights ** 2))

        if spherical_centroid is None:
            spherical_centroid = cg.spherical_centroid(self.positions.squeeze().T)

        # take the trace of the covariance matrix as the variance measure
        weighted_var = 0
        for j, particle_position in enumerate(self.positions):
            # geodesic distance
            weighted_var += self.weights[j] * gca.length(spherical_centroid, particle_position,
                                                              degrees=False) ** 2 / sum_weights * (
                                        eff_dof / (eff_dof - 1))

        # normalize
        weighted_var = weighted_var / len(self.positions)

        return weighted_var[0]

    def calc_entropy(self):
        '''
        Implement entropy measure for particle filter as described in Learning Human Ergonomic Preferences for Handovers (Bestick et al. 2018)

        Spherical discretization taken from : A new method to subdivide a spherical surface into equal-area cells,
        Malkin 2019 https://arxiv.org/pdf/1612.03467.pdf
        '''

        if self.binned == False:
            bin_particle_mapping, bin_weight_mapping = self.bin_particles()
            self.bin_particle_mapping = bin_particle_mapping
            self.bin_weight_mapping = bin_weight_mapping
            self.binned = True
        else:
            bin_weight_mapping = self.bin_weight_mapping

        entropy = 0
        for ele_bin in bin_weight_mapping.keys():
            for azi_prob in bin_weight_mapping[ele_bin]:
                if azi_prob > 0:
                    entropy += azi_prob * np.log(azi_prob)
        entropy = entropy * -1

        # perform some basic checks (e.g. that the weights of the particles sum to one)
        sum = 0
        for item in list(bin_weight_mapping.values()):
            sum += np.sum(np.array(item))
        assert np.isclose(sum, 1)

        entropy = entropy.round(4) # for numerical stability

        assert entropy >= 0, "Entropy shouldn't be negative!"

        return entropy

    def calc_info_gain(self, new_constraints):
        new_particles = copy.deepcopy(self)
        new_particles.update(new_constraints)

        return self.calc_entropy() - new_particles.calc_entropy()

    def KLD_resampling(self, k=0, epsilon=.15, N_min=20, N_max=1000, delta=0.01):
        '''
        An implementation of 'Adapting sample size in particle filters through KLD-resampling' (2013) by Li et al.
        :return: A list of particle indexes to resample
        '''

        z = norm.ppf(1 - delta)
        candidate_indexes = []
        resample_indexes = []
        N = N_min

        # todo: I should probably move this somewhere else and get rid of the _20 versions
        # dictionary with keys based on elevation, and values based on associated number of azimuth bins
        bin_occupancy = {0: np.zeros(3), 1: np.zeros(9), 2: np.zeros(15), 3: np.zeros(20), 4: np.zeros(24),
                         5: np.zeros(30),
                         6: np.zeros(30), 7: np.zeros(36), 8: np.zeros(36), 9: np.zeros(36), 10: np.zeros(36),
                         11: np.zeros(30), 12: np.zeros(30), 13: np.zeros(24), 14: np.zeros(20),
                         15: np.zeros(15),
                         16: np.zeros(9), 17: np.zeros(3)}

        while len(resample_indexes) <= N and len(resample_indexes) <= N_max or len(resample_indexes) < N_min:
            if len(candidate_indexes) > 1:
                index = candidate_indexes.pop()
            else:
                # get another set of candidate indexes using systematic resampling
                candidate_indexes = p_utils.systematic_resample(self.weights)
                np.random.shuffle(candidate_indexes)
                candidate_indexes = list(candidate_indexes)
                index = candidate_indexes.pop()

            resample_indexes.append(index)

            position_spherical = cg.cart2sph(self.positions[index])
            elevation = position_spherical[0][0]
            azimuth = position_spherical[0][1]

            elevation_bin = np.digitize(elevation, self.ele_bin_edges)
            azimuth_bin = np.digitize(azimuth, self.azi_bin_edges[elevation_bin - 1])

            if bin_occupancy[elevation_bin - 1][azimuth_bin - 1] == 0:
                k += 1
                bin_occupancy[elevation_bin - 1][azimuth_bin - 1] = 1
                if k > 1:
                    N = (k - 1) / (2 * epsilon) * (1 - 2 / (9 * (k - 1)) + np.sqrt(2 / (9 * (k - 1))) * z) ** 3

        return np.array(resample_indexes)


    def reset(self, constraint):
        '''
        Reset the particle filter to conservative include a set of new particles that are consistent with the specified constraint
        '''
        new_particle_positions = BEC_helpers.sample_human_models_uniform(constraint, 50)  # 50 is a heuristic number of particles to seed KLD resampling for calculating optimal number of resampled particles
        joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions))
        self.reinitialize(joint_particle_positions)
        resample_indexes = self.KLD_resampling()
        n_desired_reset_particles_informed = len(resample_indexes)
        print('informed number: {}'.format(n_desired_reset_particles_informed))

        # sample from the VMF + uniform distribution
        new_particle_positions_uniform = BEC_helpers.sample_human_models_random(constraint, int(np.ceil(
            n_desired_reset_particles_informed * self.u_prob_mass_scaled)))

        mu_constraint = constraint[0] / np.linalg.norm(constraint[0])
        new_particle_positions_VMF = p_utils.rand_von_mises_fisher(mu_constraint, kappa=self.VMF_kappa, N=int(
            np.ceil(n_desired_reset_particles_informed * self.VMF_prob_mass_scaled)),
                                                                   halfspace=True)
        new_particle_positions = np.vstack(
            (np.array(new_particle_positions_uniform), np.expand_dims(new_particle_positions_VMF, 1)))

        joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions_prev))
        self.reinitialize(joint_particle_positions)

        print(colored('Performed a reset', 'red'))
        self.pf_reset_count += 1
        self.reset_constraint = [constraint[0]]

    
    def knowledge_update(self, knowledge_constraints):
        self.knowledge_constraints = copy.deepcopy(knowledge_constraints)


    ################ These functions are primarily for debugging purposes. To calculate what proportion of particles are in the correct halfspace
    def calc_particles_probability(self, constraints):
        N_particles = len(self.positions)
        uniform_particles_id = []
        for particle_id in range(N_particles):
            pos  = self.positions[particle_id]
            in_BEC_area_flag = True
            for constraint in constraints:
                dot = constraint.dot(pos.T)
                if dot < 0:
                    in_BEC_area_flag = False
            
            if in_BEC_area_flag:
                uniform_particles_id.append(particle_id)
        
        self.particles_prob_correct = sum([self.weights[i] for i in uniform_particles_id])



    def calc_clusters_probability(self, constraints):
        if self.cluster_centers is not None:
            N_clusters = len(self.cluster_centers)
            uniform_clusters_id = []
            for cluster_id in range(N_clusters):
                cluster  = self.cluster_centers[cluster_id]
                in_BEC_area_flag = True
                for constraint in constraints:
                    dot = constraint.dot(cluster.T)
                    if dot < 0:
                        in_BEC_area_flag = False
                
                if in_BEC_area_flag:
                    uniform_clusters_id.append(cluster_id)
        
            self.clusters_prob_correct = sum([self.cluster_weights[i] for i in uniform_clusters_id])
        else:
            self.clusters_prob_correct = []
            


    ################################################################################################


    def update(self, constraints, c=0.5, reset_threshold_prob=0.001, learning_factor = None, plot_title = None):
        self.weights_prev = self.weights.copy()
        self.positions_prev = self.positions.copy()
        print(colored('constraints: ' + str(constraints), 'red'))


        # plot particles functions
        def label_axes(ax, mdp_class, weights=None, view_params = None):
            fs = 12
            ax.set_facecolor('white')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            if weights is not None:
                ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100/2)
            if mdp_class == 'augmented_taxi2':
                ax.set_xlabel(r'$\mathregular{w_0}$: Mud', fontsize = fs)
                ax.set_ylabel(r'$\mathregular{w_1}$: Recharge', fontsize = fs)
            elif mdp_class == 'colored_tiles':
                ax.set_xlabel('X: Tile A (brown)')
                ax.set_ylabel('Y: Tile B (green)')
            else:
                ax.set_xlabel('X: Goal')
                ax.set_ylabel('Y: Skateboard')
            ax.set_zlabel('$\mathregular{w_2}$: Action', fontsize = fs)

            # ax.view_init(elev=16, azim=-160)
            if not view_params is None:
                elev = view_params[0]
                azim = view_params[1]
            else:
                elev=16
                azim = -160
            ax.view_init(elev=elev, azim=azim)
        

        # fig = plt.figure()
        # ax = []
        # plt_id = 1
        # N = len(constraints)
        cnst_id = 0
        ##########################################


        for constraint in constraints:
            # print('constraint: {}'.format(constraint))
            
            self.reweight(constraint, learning_factor)

            

            ############ plot
            # if len(constraint) > 0:
            #     ax.append(fig.add_subplot(N, 3, plt_id, projection='3d'))
            #     plt_id += 1
            #     ax.append(fig.add_subplot(N, 3, plt_id, projection='3d', sharex=ax[cnst_id*3], sharey=ax[cnst_id*3], sharez=ax[cnst_id*3]))
            #     plt_id += 1
            #     ax.append(fig.add_subplot(N, 3, plt_id, projection='3d', sharex=ax[cnst_id*3], sharey=ax[cnst_id*3], sharez=ax[cnst_id*3]))
            #     plt_id += 1
            #     ax[cnst_id*3].title.set_text('Particles before reweighting')
            #     ax[cnst_id*3 + 1].title.set_text('Particles after reweighting')
            #     ax[cnst_id*3 + 2].title.set_text('Particles after resampling')

            #     self.plot(fig=fig, ax=ax[cnst_id*3], plot_prev=True)
            #     BEC_viz.visualize_planes([constraint], fig=fig, ax=ax[cnst_id*3])
            #     self.plot(fig=fig, ax=ax[cnst_id*3 + 1])
            #     BEC_viz.visualize_planes([constraint], fig=fig, ax=ax[cnst_id*3 + 1])

            #     # plot the spherical polygon corresponding to the constraints
            #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraint)
            #     poly = Polyhedron.Polyhedron(ieqs=ieqs)
            #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax[cnst_id*3 + 1], plot_ref_sphere=False, alpha=0.75)

            #     if constraint[0][0] == 0:
            #         view_params = [16, -160]
            #     elif constraint[0][1] == 0:
            #         view_params = [2, -100]
            #     elif constraint[0][2] == 0:
            #         view_params = [2, -60]
            #     else:
            #         view_params = [16, -160]

            #     label_axes(ax[cnst_id*3], params.mdp_class, params.weights['val'], view_params = view_params)
            #     label_axes(ax[cnst_id*3 + 1], params.mdp_class, params.weights['val'], view_params = view_params)
            #     label_axes(ax[cnst_id*3 + 2], params.mdp_class, params.weights['val'], view_params = view_params)

            #     if plot_title is not None:
            #         fig.suptitle(plot_title, fontsize=16)

            ##########################################


            if sum(self.weights) < reset_threshold_prob:
                print('Resetting... Constraint: {}'.format(constraint))
                self.reset(constraint)
            else:
                # normalize weights and update particles
                # print(colored('Normalizing particle weights...', 'red'))
                self.weights /= sum(self.weights)

                # print('weights sum: ', sum(self.weights))

                n_eff = self.calc_n_eff(self.weights)
                # print('n_eff: {}'.format(n_eff))
                if n_eff < c * len(self.weights):
                    # a) use systematic resampling
                    # resample_indexes = p_utils.systematic_resample(self.weights)
                    # self.resample_from_index(resample_indexes)

                    # b) use KLD resampling
                    # print('Plotting before resampling..')
                    # self.plot()
                    print(colored('Resampling...', 'red'))
                    # for debugging purposes
                    self.calc_particles_probability([constraint])
                    # self.cluster()
                    self.calc_clusters_probability([constraint])
                    print(colored('Particles prob before resampling: ' + str(self.particles_prob_correct) + '. Clusters prob before resampling: ' + str(self.clusters_prob_correct), 'red'))
                    # self.plot(fig=fig, ax=ax[cnst_id*3 + 1], cluster_centers=self.cluster_centers, cluster_weights=self.cluster_weights)

                    resample_indexes = self.KLD_resampling()
                    self.resample_from_index(np.array(resample_indexes))

                    # # For debugging purposes
                    self.calc_particles_probability([constraint])
                    # self.cluster()
                    self.calc_clusters_probability([constraint])
                    print(colored('Particles prob after resampling: ' + str(self.particles_prob_correct) + '. Clusters prob after resampling: ' + str(self.clusters_prob_correct), 'red'))
                    # self.plot(fig=fig, ax=ax[cnst_id*3 + 2], cluster_centers=self.cluster_centers, cluster_weights=self.cluster_weights)
                    # BEC_viz.visualize_planes([constraint], fig=fig, ax=ax[cnst_id*3 + 2])


            # # https://stackoverflow.com/questions/41167196/using-matplotlib-3d-axes-how-to-drag-two-axes-at-once
            # # link the pan of the three axes together
            # def on_move(event):
            #     if event.inaxes == ax[cnst_id*3]:
            #         ax[cnst_id*3+1].view_init(elev=ax[cnst_id*3].elev, azim=ax[cnst_id*3].azim)
            #         ax[cnst_id*3+2].view_init(elev=ax[cnst_id*3].elev, azim=ax[cnst_id*3].azim)
            #     elif event.inaxes == ax[cnst_id*3+1]:
            #         ax[cnst_id*3].view_init(elev=ax[cnst_id*3+1].elev, azim=ax[cnst_id*3+1].azim)
            #         ax[cnst_id*3+2].view_init(elev=ax[cnst_id*3+1].elev, azim=ax[cnst_id*3+1].azim)
            #     elif event.inaxes == ax[cnst_id*3+2]:
            #         ax[cnst_id*3].view_init(elev=ax[cnst_id*3+2].elev, azim=ax[cnst_id*3+2].azim)
            #         ax[cnst_id*3+1].view_init(elev=ax[cnst_id*3+2].elev, azim=ax[cnst_id*3+2].azim)
            #         return
            #     fig.canvas.draw_idle()
            
            # fig.canvas.mpl_connect('motion_notify_event', on_move)

            cnst_id += 1

        # plt.show()

        self.binned = False



    def reweight(self, constraint, learning_factor = None):
        '''
        :param constraints: normal of constraints / mean direction of VMF
        :param k: concentration parameter of VMF
        :return: probability of x under this composite distribution (uniform + VMF)
        '''

        if learning_factor is None:
            u_pdf_scaled = self.u_pdf_scaled
            VMF_kappa = self.VMF_kappa
            x1 = self.x1
            x2 = self.x2
        else:
            u_pdf_scaled = learning_factor/(2*np.pi)
            VMF_kappa, x1 = self.solve_for_distribution_params(learning_factor)
            x2 = learning_factor/0.5

        for j, x in enumerate(self.positions):
            obs_prob = self.observation_probability(u_pdf_scaled, VMF_kappa, x, constraint, x1, x2)
            
            # print('Old particle weight:', str(self.weights[j]), 'Observation probability:', str(obs_prob))
            self.weights[j] = self.weights[j] * obs_prob
            # print('New particle weight:', str(self.weights[j]))

            # self.plot_particles(x, constraints)

            


    @staticmethod
    def observation_probability(u_pdf_scaled, VMF_kappa, x, constraint, x1, x2):
        prob = 1
        p = x.shape[1]

        # the shape of constraints should be (1, p)

        # for constraint in constraints:

        dot = constraint.dot(x.T)

        if dot >= 0:
            # use the uniform dist
            prob *= u_pdf_scaled
        else:
            # use the VMF dist
            # prob *= p_utils.VMF_pdf(constraint, VMF_kappa, p, x, dot=dot)
            # prob *= p_utils.VMF_pdf(constraint, VMF_kappa, p, x, dot=dot) * x1 * x2
            prob *= p_utils.VMF_pdf(constraint[0], VMF_kappa, p, x, dot=dot) * x1 * x2   # only when for loop is not used.

        return prob


    @staticmethod
    def calc_n_eff(weights):
        return 1. / np.sum(np.square(weights))


    ## Altered functions for the team case




    ## Unique functions for the team case

    def update_jk(self, joint_knowledge, c=0.5, reset_threshold_prob=0.001):
        
        joint_constraints = joint_knowledge[0]

        print('Update JK with constraints: {}'.format(joint_constraints))
        
        self.weights_prev = self.weights.copy()
        self.positions_prev = self.positions.copy()

        self.reweight_jk(joint_constraints) # particles reweighted even if one of the original constraints is satisfied.

        # Joint particles reset temporarily removed. To be fixed and added back!

        # if sum(self.weights) < reset_threshold_prob:
        #     self.reset_jk(joint_constraints) # TODO: Check if the same reset procedure applies to the joint knowledge case. Reset particles based on the first constraint for now.
        #     print(colored('Resetting weights JK. The reset process is currently incomplete.', 'red'))
        # else:
        #     # normalize weights and update particles
        #     self.weights /= sum(self.weights)

        #     n_eff = self.calc_n_eff(self.weights)
        #     # print('n_eff: {}'.format(n_eff))
        #     if n_eff < c * len(self.weights):
        #         # a) use systematic resampling
        #         # resample_indexes = p_utils.systematic_resample(self.weights)
        #         # self.resample_from_index(resample_indexes)

        #         # b) use KLD resampling
        #         resample_indexes = self.KLD_resampling()
        #         print('Resampling...')
        #         self.resample_from_index(np.array(resample_indexes))

        self.binned = False

    
    def reset_jk(self, joint_constraints):
        
        # TODO: Currently, just copied over the reset function from the single constraint case. Need to modify this to work for the joint knowledge case.
        
        '''
        Reset the particle filter to conservative include a set of new particles that are consistent with the specified constraint
        '''
        new_particle_positions = []
        N_team = len(joint_constraints[0])
        for individual_constraints in joint_constraints:
            new_particle_positions.extend(BEC_helpers.sample_human_models_uniform(individual_constraints, int(50/N_team))) # sample from feasible region of each person on the team        
        
        joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions))
        self.reinitialize(joint_particle_positions)
        resample_indexes = self.KLD_resampling()
        n_desired_reset_particles_informed = len(resample_indexes)
        print('informed number: {}'.format(n_desired_reset_particles_informed))

        # sample from the VMF + uniform distribution
        i = 0
        for individual_constraints in joint_constraints:
            for constraint in individual_constraints:
                new_particle_positions_uniform = BEC_helpers.sample_human_models_random(constraint, int(np.ceil(
                n_desired_reset_particles_informed * self.u_prob_mass_scaled)))

                mu_constraint = constraint[0] / np.linalg.norm(constraint[0])
                new_particle_positions_VMF = p_utils.rand_von_mises_fisher(mu_constraint, kappa=self.VMF_kappa, N=int(
                np.ceil(n_desired_reset_particles_informed * self.VMF_prob_mass_scaled)),
                                                                   halfspace=True)
                
                if i == 0:
                    new_particle_positions = np.vstack(
                        (np.array(new_particle_positions_uniform), np.expand_dims(new_particle_positions_VMF, 1)))
                else:
                    # print('Using new particle update method...')
                    new_particle_positions = np.vstack(
                        (new_particle_positions, np.array(new_particle_positions_uniform), np.expand_dims(new_particle_positions_VMF, 1)))
                    
                i += 1

        
        joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions_prev))
        self.reinitialize(joint_particle_positions)

        print(colored('Performed a reset for joint knowledge', 'red'))
        self.pf_reset_count += 1






    def reweight_jk(self, constraints, plot_particles_flag = False):
        '''
        :param constraints: normal of constraints / mean direction of VMF
        :param k: concentration parameter of VMF
        :return: probability of x under this composite distribution (uniform + VMF)
        '''

        # plot_particles_input = input('Plot particles? (y/n)')
        # if plot_particles_input == 'y':
        #     plot_particles_flag = True
        
        plot_particles_flag = False
        # cnt = 0
        
        for j, x in enumerate(self.positions):
            
            prob = self.observation_probability_jk(self.u_pdf_scaled, self.VMF_kappa, x, constraints, plot_particles_flag, self.x1, self.x2)
            self.weights[j] = self.weights[j] * prob
            
            # debug (only possible for high von-mises probabilities. should not normllay happen)            
            if prob > 0.5 or plot_particles_flag:
                self.plot_particles(x, constraints)
                
            # print('Particle ', j, 'with weights ', x)
            # print('Current weight: ', self.weights_prev[j])
            # print('Probability: ', prob)
            # print('Updated weight: ', self.weights[j])


    # this function is primarily for debugging particle update
    def plot_particles(self, x, constraints):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        ax1.title.set_text('Particles before update')
        self.plot(fig=fig, ax=ax1, plot_prev=True)
        ax1.scatter(x[0, 0], x[0, 1], x[0, 2], marker='o', c='r', s=100/2)
        # for constraint in constraints:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)
        # utils_`teams.visualize_planes_team(constraints, fig=fig, ax=ax1)
        # utils_teams.visualize_planes_team(constraints[0], fig=fig, ax=ax1)

        ax2.title.set_text('Particles after update')
        self.plot(fig=fig, ax=ax2)
        ax2.scatter(x[0, 0], x[0, 1], x[0, 2], marker='o', c='r', s=100/2)
        # utils_teams.visualize_planes_team(constraints, fig=fig, ax=ax2)
        # for constraint in constraints:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax2)

        plt.show()


    @staticmethod
    def observation_probability_jk(u_pdf_scaled, VMF_kappa, x, joint_constraints, plot_particles_flag, x1, x2):
        prob = 1
        p = x.shape[1]

        # check if atleast one of the constraints is satisfied, i.e. none of the inverse constraints are satisfied
        team_constraint_satisfied_flag = []
        prob_individual = np.ones([1, len(joint_constraints)])
        
        ind_id = 0
        for individual_constraint in joint_constraints:
            # print('Individual constraints: ', individual_constraint)
            team_constraint_satisfied_flag.append(True)
            
            # check if the constraints for each individual member are satisfied
            for constraint in individual_constraint:
                dot = constraint.dot(x.T)

                if dot < 0:
                    team_constraint_satisfied_flag[-1] = False
                    # prob_individual[0, ind_id] *= p_utils.VMF_pdf(constraint, VMF_kappa, p, x, dot=dot) # use the von Mises dist
                    prob_individual[0, ind_id] *= p_utils.VMF_pdf(constraint, VMF_kappa, p, x, dot=dot) * x1 * x2 # use the von Mises dist
                    # prob_individual[0, ind_id] *= p_utils.VMF_pdf(constraint, 4, p, x, dot=dot) * x1 * x2 # use the von Mises dist
                else:
                    prob_individual[0, ind_id] *= u_pdf_scaled # use the uniform dist

            ind_id += 1


        
        ## Update the pdf even if constraints are satisfied for one of the members; 
        
        # # Method 1: Use product of pdfs; this is from Mike's work which is applicable for half-spaces. Here the distribution should be for the union of constraints and thus pdf for uniform distribution ould vary.
        # if sum(np.array(team_constraint_satisfied_flag)) > 0  :
          
        #     # use the uniform dist
        #     prob *=  0.12779
        # else:
        #     # use the VMF dist for each of the constraints for all individuals (Design choice).
            
        #     # vm = []
        #     for individual_constraint in joint_constraints:
        #         for i in range(len(individual_constraint)):
        #             dot = individual_constraint[i].dot(x.T)
        #             # vm.append(p_utils.VMF_pdf(individual_constraint[i], k, p, x, dot=dot))
        #             prob *= p_utils.VMF_pdf(individual_constraint[i], k, p, x, dot=dot)
            
        #     # print('Von mises probabilities for each constraint: ', vm)
        #     # debug
        #     if prob > 0.13 or plot_particles_flag:
        #         print('All constraints: ', joint_constraints)
        #         print('Von mises joint prob: ', prob)
        #         print('final constraint_satisfied_flag: ', team_constraint_satisfied_flag)



        # Method 2: Use maximum probability distribution
        prob = np.max(prob_individual)

        if plot_particles_flag:
            print('Individual probabilities: ', prob_individual)
            print('All constraints: ', joint_constraints)
            print('final constraint_satisfied_flag: ', team_constraint_satisfied_flag)
            print('Prob: ', prob)


        return prob
    

    # def update_ck():







        