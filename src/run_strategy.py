

import sys
sys.path.insert(0, "../SingleSample")
import argparse
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import time

try:
    import cupy as cp
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            np = cp
        else:
            raise ImportError("No CUDA-capable GPU found.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        raise ImportError(f"CUDA runtime error: {e}")
except ImportError as e:
    print(f"Falling back to numpy due to error: {e}")
    import numpy as np


import json

from data_generation import *
from models import *
from visualization import plot_data_2d, generate_grid
from example_functions import *


#--------------------------------------------------------------------------
# Terminal Input
#--------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Use setting dictionary to run 2sls with finite basis function extension.')
parser.add_argument('--output_dir', type=str, default=None, help='Name of output directory.')
parser.add_argument("--experiment_name", type=str, default="ActiveLearning", help="Name of experiment.")
parser.add_argument("--seed", type=int, default=4, help="Seed for the random number generator.")  # 8282
parser.add_argument('--n_seeds', type=int, default=50, help='Number of seeds.')

parser.add_argument('--dx', type=int, default=2, help='Number of treatment variables.')
parser.add_argument('--dz', type=int, default=2, help='Number of instrument variables.')
parser.add_argument('--data_scenario', type=str, default="standard", help='Data scenario.')

parser.add_argument('--functional_type', type=str, default="gradient", help='Functional type.')
parser.add_argument('--xbar_type', type=str, default="mean", help='Type of xbar.')
parser.add_argument("--strategy_type", type=str, default="continuous_exploration", help="Do exploration phase.")
parser.add_argument('--n_exploration', type=int, default=250, help='Number of samples.') # 100
parser.add_argument('--n_exploitation', type=int, default=250, help='Number of exploitation samples.') # 250
parser.add_argument('--T', type=int, default=16, help='Number of experiments.') # 16
parser.add_argument('--T_exploration', type=int, default=10, help='Number of exploration samples.') # 10
parser.add_argument('--K_exploitation', type=int, default=250, help='Number of exploration samples.') # 100
parser.add_argument("--sample_sigma", type=float, default=0.001, help="Regularization parameter for the kernel.")
parser.add_argument("--gaussian_K", type=int, default=3, help="Parameter for Gaussian Mixture Model.")
parser.add_argument("--do_sigma_update", type=str, default="Y", help="Parameter for Gaussian Mixture Model.")

parser.add_argument("--lam_c", type=float, default=0.03, help="Regularization parameter for feasibility.")
parser.add_argument("--lam_first", type=float, default=0.01, help="Regularization parameter for smoothness.")
parser.add_argument("--lam_second", type=float, default=1.0, help="Regularization parameter for smoothness.")
parser.add_argument("--gamma_kernel", type=float, default=1.0, help="Regularization parameter for the kernel.")
parser.add_argument("--step_size_list", type=list, default=[4.0, 1.0, 0.1, 0.01, 0.001], help="Regularization parameter for the adaptive strategy.")



def main(experiment_dir, args):

    #--------------------------------------------------------------------------
    # Logging
    #--------------------------------------------------------------------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level
    # Create a file handler
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'logfile.log'), delay=True)  # Replace 'logfile.log' with your filename
    file_handler.setLevel(logging.INFO)  # Set the logging level
    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Set the logging level

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

     # Log the command line arguments
    logger.info('Command line arguments: %s', args)

    #--------------------------------------------------------------------------
    # Data Parameter Definition
    #--------------------------------------------------------------------------

    # save parameter to file
    args = parser.parse_args()
    args_dict = vars(args)
    with open(os.path.join(experiment_dir, 'parameters.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    T = args.T
    seed = args.seed
    n_exploitation = args.n_exploitation
    T_exploration = args.T_exploration
    K_exploitation = args.K_exploitation
    n_exploration = args.n_exploration
    sigma = args.sample_sigma
    strategy_type = args.strategy_type
    n_seeds = args.n_seeds
    dz = args.dz
    dx = args.dx
    step_size_list = args.step_size_list
    data_scenario = args.data_scenario
    functional_type = args.functional_type
    xbar_type = args.xbar_type
    do_sigma_update = args.do_sigma_update

    # Set predefined seed
    np.random.seed(seed)
    
    #--------------------------------------------------------------------------
    # Experiment Details
    #--------------------------------------------------------------------------  
    #-----------------------------------
    # Data Scenario
    #-----------------------------------
    if data_scenario == "standard":
        #alpha = np.array([[0.5, -0.25], [-0.25, 0.5]])
        alpha = np.random.randn(dz, dx)
        h = lambda z, e_x: h_nonlinear(z, e_x, coef=alpha)
        
        f = lambda x: 20*f_nonlinear(x, interaction=False)

    elif data_scenario == "highdim":
        dz = 5
        dx = 20
        alpha = np.random.randn(dz, dx) 
        alpha = np.zeros((dz, dx))
        alpha[:dz, :dz] = np.identity(dz)
        h = lambda z, e_x: h_nonlinear(z, e_x, coef=alpha)
        f = lambda x: 20*f_nonlinear(x, interaction=False)
        
    
    elif data_scenario == "highdim2":
        dz = 20
        dx = 20
        alpha = np.random.randn(dz, dx) 
        alpha = np.zeros((dz, dx))
        alpha[:dz, :dz] = np.identity(dz)
        h = lambda z, e_x: h_nonlinear(z, e_x, coef=alpha)
        f = lambda x: 20*f_nonlinear(x, interaction=False)
        
    #-----------------------------------
    # Type of Functional
    #-----------------------------------
    if (functional_type == "gradient") & (data_scenario in ["highdim", "highdim2", "standard"]):
        comp = 0
        grad_f_exact = lambda x, comp: 20*grad_f_nonlinear(x, comp)

    elif functional_type == "integral":
        grad_f_exact = lambda x, comp: 20*int_f_nonlinear(x, comp)
    
    
    #-----------------------------------
    # Type of local point
    #-----------------------------------
    if xbar_type == "mean":
        def xbar_position(x_data):
            return np.mean(x_data, axis=0)
        
    elif xbar_type == "quantile":
        def xbar_position(x_data):
            return np.quantile(x_data, 0.3, axis=0)
    
    elif xbar_type == "fix":
        def xbar_position(x_data):
            return np.array([0.2, 0.2])
        

    #--------------------------------------------------------------------------
    # Data Generation
    #--------------------------------------------------------------------------
    # Confounded Data (only for checking purpose)
    n = 1000
    z_data = np.random.uniform(-2, 2, (n, dz))
    x_data, y_data = generate_data(z_data, dx, h, f)

    assert x_data.shape == (n, dx), "Shape of x should be (n, dx)"
    assert y_data.shape == (n, 1), "Shape of y should be (n, 1)"

    # Value we want to get the gradient at
    xbar = xbar_position(x_data)
    xbar = xbar[..., np.newaxis].T
  
    
    #--------------------------------------------------------------------------
    # Optimization Parameter Definition
    #-------------------------------------------------------------------------- 
    lam_c = args.lam_c
    lam_first = args.lam_first
    gamma_kernel = args.gamma_kernel
    sigma = args.sample_sigma
    lam_second = args.lam_second

    # TODO : check if this still applies ?!
    #lam_second = lam_c


    #--------------------------------------------------------------------------
    # Optimization Routine
    #--------------------------------------------------------------------------
    res_seed = {}
    res_seed["Grad_Min_Exact"] = []
    res_seed["Grad_Max_Exact"] = []
    res_seed["Time"] = []
    res_seed["Gamma"] = []
    res_seed["Mu"] = []
    res_seed["Sigma"] = []
    # set seed list for confidence bounds
    list_seed = np.random.randint(0, 10000, n_seeds)

    for jter in range(n_seeds):

        logger.info(f"------------------------------------------------ Seed {jter} of {n_seeds} ------------------------------------------------")
        # set seed for confidence bounds
        seed_jter = list_seed[jter]
        np.random.seed(seed_jter)
        res = {}
        
        X1, X2, _, f_grid = generate_grid(x_data, f)
        res["X1"] = X1
        res["X2"] = X2
        res["f_grid"] = f_grid
        res["xbar"] = xbar
        res["x_data"] = x_data 

        grad_true = np.zeros((dx,))
        grad_comp = grad_f_exact(xbar, comp)
        grad_true[comp] = grad_comp.squeeze()
               
        Z_Exp_runs = []
        X_Exp_runs = []
        Y_Exp_runs = []

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Random Sampling
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if strategy_type=="random_sampling":
            # initate results
            Grad_Min_Exact = np.zeros((1, T))
            Grad_Max_Exact = np.zeros((1, T))
            Time = np.zeros((1, T))
            Grad = np.zeros((1, T))

            for iter in range(T):
                logger.info(f"------------------------------------------------ Run {iter} of {T} ------------------------------------------------")
                start = time.time()
                # ----------------------------------------------------------------------------------------
                # ---------------------------------- Exploration Phase -----------------------------------
                # ----------------------------------------------------------------------------------------
                # Sample random scenario
                mu_z = np.random.randn(dz)
                cov = np.eye(dz) * sigma
                # -----------------------------
                # Perform experiment
                z_samples = np.random.multivariate_normal(mu_z, cov, n_exploitation)
                x_samples, y_samples = generate_data(z_samples, dx, h, f)
                # -----------------------------
                # -----------------------------------------------------------------------------------------
                # ---------------------------------- Exploitation Phase -----------------------------------
                # -----------------------------------------------------------------------------------------
                # add the samples to the exploitation data
                X_Exp_runs.append(x_samples)
                Y_Exp_runs.append(y_samples)
                Z_Exp_runs.append(z_samples)
                X_Samples = np.concatenate(X_Exp_runs)
                Y_Samples = np.concatenate(Y_Exp_runs)
                Z_Samples = np.concatenate(Z_Exp_runs)
                
                # ------------------------------------------------------------------
                # Minimization Problem
                # ------------------------------------------------------------------
                # Minimal Solution Exact
                theta_min_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=True)
                theta_max_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=False)
                Kxxbar = linear_functional_rbf_kernel(X_Samples, xbar, gamma_kernel, comp, type=functional_type)
                grad_min_exact = Kxxbar@theta_min_exact
                grad_max_exact = Kxxbar@theta_max_exact
                
                # ---------------------------
                # save results
                Grad_Min_Exact[:, iter] = grad_min_exact.squeeze()
                Grad_Max_Exact[:, iter] = grad_max_exact.squeeze()
                Grad[:, iter] = grad_true.squeeze()[comp]
                Time[:, iter] = time.time() - start

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Explore Then Exploit
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if strategy_type=="explore_then_exploit":
            Z_Exp = []
            X_Exp = []
            Y_Exp = []
            X_Mu_extend = []

            if T_exploration > T:
                T_exploration = T - 1

            Grad_Min_Exact = np.zeros((1, int(T - T_exploration)))
            Grad_Max_Exact = np.zeros((1, int(T - T_exploration)))
            Time = np.zeros((1, int(T - T_exploration)))
            Grad = np.zeros((1, int(T - T_exploration)))


            # ----------------------------------------------------------------------------------------
            # ---------------------------------- Exploration Phase -----------------------------------
            # ----------------------------------------------------------------------------------------
            start = time.time()
            for _ in range(T_exploration):
                # random scenario
                mu_z = np.random.randn(dz)
                cov = np.eye(dz) * sigma

                # generate the data
                z_candidate = np.random.multivariate_normal(mu_z, cov, (n_exploration))
                x_candidate, y_candidate = generate_data(z_candidate, dx, h, f)

                Z_Exp.append(z_candidate)
                X_Exp.append(x_candidate)
                Y_Exp.append(y_candidate)
            
            Z_Exp = np.concatenate(Z_Exp)
            X_Exp = np.concatenate(X_Exp)
            Y_Exp = np.concatenate(Y_Exp)


            # -----------------------------------------------------------------------------------------
            # ---------------------------------- Exploitation Phase -----------------------------------
            # -----------------------------------------------------------------------------------------
            for iter in range(int(T - T_exploration)):
                # compute the distance
                Dist = np.linalg.norm(X_Exp - xbar, axis=1)
                X_Mu_extend = Dist.squeeze()

                # select the best samples
                indices_smallest = np.argsort(X_Mu_extend)[:K_exploitation]
                Z_Exp = Z_Exp[indices_smallest, :]
                X_Exp = X_Exp[indices_smallest, :]
                Y_Exp = Y_Exp[indices_smallest]

                mu_z_exp = np.mean(Z_Exp, axis=0)

                # Compute the variance of each column of Z_Exp
                var_z_exp = np.var(Z_Exp, axis=0)
                cov_z_exp = np.diag(var_z_exp)

                # fit distribution over remaining Z
                # Generate the experiment from the data
                z_samples = np.random.multivariate_normal(mu_z_exp, cov_z_exp, n_exploitation)
                x_samples, y_samples = generate_data(z_samples, dx, h, f)
                
                # add the samples to the exploitation data
                X_Exp_runs.append(x_samples)
                Y_Exp_runs.append(y_samples)
                Z_Exp_runs.append(z_samples)

                X_Samples = np.concatenate(X_Exp_runs)
                Y_Samples = np.concatenate(Y_Exp_runs)
                Z_Samples = np.concatenate(Z_Exp_runs)

                theta_min_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=True)
                theta_max_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=False)
                Kxxbar = linear_functional_rbf_kernel(X_Samples, xbar, gamma_kernel, comp, type=functional_type)
                grad_min_exact = Kxxbar@theta_min_exact
                grad_max_exact = Kxxbar@theta_max_exact
                
                Grad_Min_Exact[:, iter] = grad_min_exact.squeeze()
                Grad_Max_Exact[:, iter] = grad_max_exact.squeeze()
                
                Grad[:, iter] = grad_true.squeeze()[comp]
                Time[:, iter] = time.time() - start
                start = time.time()


        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Continuous Exploration
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if strategy_type=="continuous_exploration":
            
            Grad_Min_Exact = np.zeros((1, int(T / 2)))
            Grad_Max_Exact = np.zeros((1, int(T / 2)))
            Grad = np.zeros((1, int(T / 2)))
            Time = np.zeros((1, int(T / 2)))

            # save all exploration data
            Z_Exp_exploration = []
            X_Exp_exploration = []
            Y_Exp_exploration = []
            
            for iter in range(int(T / 2)):
                logger.info(f"------------------------------------------------ Run {iter} of {T} ------------------------------------------------")
                start = time.time()


                # ----------------------------------------------------------------------------------------
                # ---------------------------------- Exploration Phase -----------------------------------
                # ----------------------------------------------------------------------------------------
                # random scenario
                mu_z = np.random.randn(dz)
                cov = np.eye(dz) * sigma

                # generate the data
                z_candidate = np.random.multivariate_normal(mu_z, cov, (n_exploration))
                x_candidate, y_candidate = generate_data(z_candidate, dx, h, f)

                # save all samples from the exploration
                Z_Exp_exploration.append(z_candidate)
                X_Exp_exploration.append(x_candidate)
                Y_Exp_exploration.append(y_candidate)
            
                Z_Exp = np.concatenate(Z_Exp_exploration)
                X_Exp = np.concatenate(X_Exp_exploration)
                Y_Exp = np.concatenate(Y_Exp_exploration)


                # -----------------------------------------------------------------------------------------
                # ---------------------------------- Exploitation Phase -----------------------------------
                # -----------------------------------------------------------------------------------------
                # compute the distance
                Dist = np.linalg.norm(X_Exp - xbar, axis=1)
                X_Mu_extend = Dist.squeeze()
                # select the data overall with the smallest distance
                indices_smallest = np.argsort(X_Mu_extend)[:K_exploitation]
                Z_Exp = Z_Exp[indices_smallest, :]
                X_Exp = X_Exp[indices_smallest, :]
                Y_Exp = Y_Exp[indices_smallest]

                mu_z_exp = np.mean(Z_Exp, axis=0)

                # Compute the variance of each column of Z_Exp
                var_z_exp = np.var(Z_Exp, axis=0)
                cov_z_exp = np.diag(var_z_exp)

                # fit distribution over remaining Z
                # Generate the experiment from the data
                z_samples = np.random.multivariate_normal(mu_z_exp, cov_z_exp, n_exploitation)
                x_samples, y_samples = generate_data(z_samples, dx, h, f)
                
                # add the samples to the exploitation data
                X_Exp_runs.append(x_samples)
                Y_Exp_runs.append(y_samples)
                Z_Exp_runs.append(z_samples)

                X_Samples = np.concatenate(X_Exp_runs)
                Y_Samples = np.concatenate(Y_Exp_runs)
                Z_Samples = np.concatenate(Z_Exp_runs)

                theta_min_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=True)
                theta_max_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=False)
                Kxxbar = linear_functional_rbf_kernel(X_Samples, xbar, gamma_kernel, comp, type=functional_type)
                grad_min_exact = Kxxbar@theta_min_exact
                grad_max_exact = Kxxbar@theta_max_exact
                
                Grad_Min_Exact[:, iter] = grad_min_exact.squeeze()
                Grad_Max_Exact[:, iter] = grad_max_exact.squeeze()
                
                Grad[:, iter] = grad_true.squeeze()[comp]
                Time[:, iter] = time.time() - start


        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Adaptive Sampling
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if strategy_type=="adaptive_sampling":
                 
            ns = len(step_size_list)
            K = args.gaussian_K
            Grad_Min_Exact = np.zeros((ns, T))
            Grad_Max_Exact = np.zeros((ns, T))
            Grad = np.zeros((ns, 1))
            Time = np.zeros((ns, T))
            Gamma_Update = np.zeros((ns, T, K))
            Mu_Update = np.zeros((ns, T, K * dz))
            Sigma_Update = np.zeros((ns, T, K * dz))

            Z_iteration_samples = dict()
            X_iteration_samples = dict()
            Y_iteration_samples = dict()
            
            for i in range(ns):
                start = time.time()
                Gamma = np.ones((K, ))/ K
                Mu = np.random.randn(K, dz)

                # initialize just diagonal matrix
                Sigma = np.ones((K, dz))

                Z_Exp_runs = []
                X_Exp_runs = []
                Y_Exp_runs = []
                Z_Exp = []
                X_Exp = []
                Y_Exp = []
            

                # ----------------------------------------------------------------------------------------
                # ---------------------------------- Learning Step ---------------------------------------
                # ----------------------------------------------------------------------------------------     
                step_size = step_size_list[i]
                logger.info(f"--------- Step Size {step_size} ------------")
                for t in range(int(T)):
                    if t < T_exploration:
                        logger.info(f"--------- Learning {t} of {T_exploration} ------------")
                        
                        Gamma_Update[i, t, :] = Gamma.reshape(-1,1).squeeze()
                        Mu_Update[i, t, :] = Mu.reshape(-1,1).squeeze()
                        Sigma_Update[i, t, :] = Sigma.reshape(-1,1).squeeze()

                        z_candidate = sample_gmm_sigma(n_exploration, Gamma, Mu, Sigma)
                        x_candidate, y_candidate = generate_data(z_candidate, dx, h, f)
                        # Compute the objective function
                        
                        Z_Exp.append(z_candidate)
                        X_Exp.append(x_candidate)
                        Y_Exp.append(y_candidate)

                        def create_batches(z_candidate, x_candidate, y_candidate, num_batches=10):
                            # Get the number of samples
                            num_samples = z_candidate.shape[0]

                            # Create an array of indices and shuffle it
                            indices = np.arange(num_samples)
                            np.random.shuffle(indices)

                            # Use the shuffled indices to shuffle the z, x, and y arrays
                            z_candidate = z_candidate[indices]
                            x_candidate = x_candidate[indices]
                            y_candidate = y_candidate[indices]

                            # Split the indices into batches
                            index_batches = np.array_split(indices, num_batches)

                            # Use the index batches to create z, x, and y batches
                            z_batches = [z_candidate[index_batch] for index_batch in index_batches]
                            x_batches = [x_candidate[index_batch] for index_batch in index_batches]
                            y_batches = [y_candidate[index_batch] for index_batch in index_batches]

                            return z_batches, x_batches, y_batches
                        
                        num_batches = 10
                        obj = np.zeros(num_batches)
                        gamma_update = np.zeros((num_batches, K))
                        mean_update = np.zeros((num_batches, K, dz))
                        sigma_update = np.zeros((num_batches, K, dz))
                        
                        z_batch, x_batch, y_batch = create_batches(z_candidate, x_candidate, y_candidate, num_batches)
                        
                        for j in range(num_batches):
                            obj[j] = objective_bounds_kernel(z_batch[j], x_batch[j], y_batch[j], xbar, comp, gamma_kernel, lam_c, lam_first, lam_second)
                            gamma_update[j, :], mean_update[j, :, :], sigma_update[j, :, :] = log_likelihood_derivatives(z_batch[j], Gamma, Mu, Sigma)
                            gamma_update[j, :] *=  obj[j]
                            mean_update[j, :, :] *=  obj[j]
                            sigma_update[j, :, :] *=  obj[j]

                        # Update the parameters
                        Gamma = Gamma - step_size * np.mean(gamma_update, axis=0).squeeze()
                        # normalization of gamma
                        if (Gamma > 0).sum() < K: 
                            logger.info(f"Gamma should be positive")
                            Gamma[Gamma < 0] = 0.0001
                        Gamma = Gamma / np.sum(Gamma)
                        Mu = Mu - step_size * np.mean(mean_update, axis=0).squeeze()
                        
                        if do_sigma_update == "Y":
                            Sigma = Sigma - step_size * np.mean(sigma_update, axis=0).squeeze()
                        else:
                            Sigma = Sigma
                        
                        # Check if Sigma is updated to rigourously 
                        eps = 1e-6
                        Sigma[Sigma < eps] = 1 / 100

                        theta_min_exact = kernel_minmax_functional(xbar, x_candidate, y_candidate, z_candidate, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=True)
                        theta_max_exact = kernel_minmax_functional(xbar, x_candidate, y_candidate, z_candidate, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=False)
                        Kxxbar = linear_functional_rbf_kernel(x_candidate, xbar, gamma_kernel, comp, type=functional_type)
                        grad_min_exact = Kxxbar@theta_min_exact
                        grad_max_exact = Kxxbar@theta_max_exact
                        Time[i, t] = time.time() - start
                        # save for each step size
                        Grad_Min_Exact[i, t] = grad_min_exact.squeeze()#[comp]
                        Grad_Max_Exact[i, t] = grad_max_exact.squeeze()#[comp]
                        
                    else:
                        start = time.time()


                        # ----------------------------------------------------------------------------------------
                        # ---------------------------------- Exploitation Step -----------------------------------
                        # ----------------------------------------------------------------------------------------
                        logger.info(f"--------- Exploit {jter} of {n_seeds} ------------")
                        # Compute the final sample set
                        z_samples = sample_gmm_sigma(n_exploitation, Gamma, Mu, Sigma)
                        x_samples, y_samples = generate_data(z_samples, dx, h, f)

                        # add the samples to the exploitation data
                        X_Exp_runs.append(x_samples)
                        Y_Exp_runs.append(y_samples)
                        Z_Exp_runs.append(z_samples)

                        X_Samples = np.concatenate(X_Exp_runs)
                        Y_Samples = np.concatenate(Y_Exp_runs)
                        Z_Samples = np.concatenate(Z_Exp_runs)

                        Z_Exp.append(z_samples)
                        X_Exp.append(x_samples)
                        Y_Exp.append(y_samples)
  
                        theta_min_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=True)
                        theta_max_exact = kernel_minmax_functional(xbar, X_Samples, Y_Samples, Z_Samples, gamma_kernel, comp, lam_c, lam_first, lam_second, minimum=False)
                        Kxxbar = linear_functional_rbf_kernel(X_Samples, xbar, gamma_kernel, comp, type=functional_type)
                        grad_min_exact = Kxxbar@theta_min_exact
                        grad_max_exact = Kxxbar@theta_max_exact
                        
                        Time[i, t] = time.time() - start
                        # save for each step size
                        Grad_Min_Exact[i, t] = grad_min_exact.squeeze()#[comp]
                        Grad_Max_Exact[i, t] = grad_max_exact.squeeze()#[comp]
                        
                        Grad[i, :] = grad_true.squeeze()[comp]
                        
                Z_iteration_samples.update({step_size: Z_Exp})
                X_iteration_samples.update({step_size: X_Exp})
                Y_iteration_samples.update({step_size: Y_Exp})

        res["Grad_Min_Exact"] = Grad_Min_Exact
        res["Grad_Max_Exact"] = Grad_Max_Exact
        res["Grad"] = Grad
        res["Z_Samples"] = Z_Samples
        res["X_Samples"] = X_Samples
        res["Y_Samples"] = Y_Samples
        res["Time"] = Time
         
    
        # ------------------------------------------------------------------------------------------------------------------------------------
        # save results over different seeds
        # ------------------------------------------------------------------------------------------------------------------------------------
        res_seed["Grad_Min_Exact"].append(Grad_Min_Exact)
        res_seed["Grad_Max_Exact"].append(Grad_Max_Exact)
        res_seed["Time"].append(Time)
        if strategy_type == "adaptive_sampling":
            res_seed["Gamma"].append(Gamma_Update)
            res_seed["Mu"].append(Mu_Update)
            res_seed["Sigma"].append(Sigma_Update)
            res["Z_Samples"] = Z_iteration_samples
            res["X_Samples"] = X_iteration_samples
            res["Y_Samples"] = Y_iteration_samples
        # save result as npy file
        np.save(os.path.join(experiment_dir, f"results_{jter}.npy"), res)
        np.save(os.path.join(experiment_dir, f"results_seed.npy"), res_seed)



if __name__ == "__main__":

    args = parser.parse_args()
    
    if args.output_dir is None:
        output_dir = "/Users/elisabeth.ailer/Projects/P14_NonlinearSampleIV/Output/ActiveLearning"
        now = datetime.now()

        # Format as a string
        now_str = now.strftime("%Y%m%d_%H%M")

        # Append to experiment name
        experiment_name = f"{args.experiment_name}_{args.strategy_type}_{args.data_scenario}_{args.functional_type}_{args.xbar_type}_{now_str}"
        experiment_dir = os.path.join(output_dir, experiment_name)

    else: 
        output_dir = args.output_dir
        experiment_name = args.experiment_name
        experiment_dir = output_dir
    
    os.makedirs(experiment_dir, exist_ok=True)
    main(experiment_dir, args)
            
        

    