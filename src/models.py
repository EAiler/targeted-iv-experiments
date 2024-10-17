import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import quad
from sklearn.metrics.pairwise import rbf_kernel


def log_likelihood_derivatives(Z_Samples, Gamma, Mu, Sigma):
    """
    Compute the derivatives of the log-likelihood of the GMM w.r.t. the parameters.

    Parameters
    ----------
    Z_Samples : np.ndarray
        instrument variable samples
    Gamma : np.ndarray
        weight parameter
    Mu : np.ndarray
        mean parameter
    Sigma : np.ndarray
        variance parameter

    Returns
    -------
    tuple (np.ndarray, np.ndarray, np.ndarray)
        derivatives of the log-likelihood w.r.t. the parameters
    """

    N, _ = Z_Samples.shape
    K = len(Gamma)  # Number of components
    
    # Storage for gradients
    dL_dGamma = np.zeros_like(Gamma)
    dL_dMu = np.zeros_like(Mu)
    dL_dSigma = np.zeros_like(Sigma)
    
    # Storage for responsibilities
    responsibilities = np.zeros((N, K))
    
    # Calculate responsibilities
    for k in range(K):
        cov = np.diag(Sigma[k])  # cov matrix
        pdf_values = multivariate_normal.pdf(Z_Samples, mean=Mu[k], cov=cov)
        responsibilities[:, k] = Gamma[k] * pdf_values
    
    # Make sure they sum to one
    responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= responsibilities_sum
    
    # Calculate gradients
    for k in range(K):
        # Gradient w.r.t. the mixture weights
        dL_dGamma[k] = np.mean(responsibilities[:, k] / Gamma[k])

        # Gradient w.r.t means
        diff = Z_Samples - Mu[k]  # Shape (n, d)
        Sigma_Inv = 1 / Sigma[k]  # Inverse of diagonal elements
        weighted_diff = responsibilities[:, k, np.newaxis] * diff * Sigma_Inv  # Shape (n, d)
        dL_dMu[k] = np.mean(weighted_diff, axis=0)  # average over n samples
        
        # Gradient w.r.t. the diagonal covariance entries
        sq_diff = diff ** 2  # Element-wise square
        weighted_sq_diff = responsibilities[:, k, np.newaxis] * sq_diff * Sigma_Inv**2  # Element-wise inverse squared (n, d)
        dL_dSigma[k] = -0.5 * (Sigma_Inv - np.mean(weighted_sq_diff, axis=0))
        
    # Normalizing the gradient of the mixture weights
    dL_dGamma -= np.mean(dL_dGamma)  # Ensure the sum of updates is zero to maintain the constraint sum(weights) = 1
    
    return dL_dGamma, dL_dMu, dL_dSigma


def sample_gmm_sigma(N, Gamma, Mu, Sigma):
    """
    sample from a Gaussian Mixture Model with diagonal covariance matrices.

    Parameters
    ----------
    N : int
        number of samples
    Gamma : np.ndarray
        weight parameter
    Mu : np.ndarray
        mean parameter
    Sigma : np.ndarray
        covariance parameter

    Returns
    -------
    np.ndarray
        samples from the GMM
    """
    
    # Number of components
    K, _ = Mu.shape

    # Sample component indices
    indices = np.random.choice(K, size=N, p=Gamma)

    # Sample from multivariate Gaussian
    samples = np.array([np.random.multivariate_normal(Mu[i], np.diag(Sigma[i])) for i in indices])

    return samples


def objective_bounds_kernel(z, x, y, xbar, comp, gamma_kernel, lam_feasibility, lam_first, lam_second, type="gradient"):
    """
    value of objective function which computes the upper and the lower bounds of the objective function.

    Parameters
    ----------
    z : np.ndarray
        instrument samples values
    x : np.ndarray
        treatment samples values
    y : np.ndarray
        outcome sample values
    xbar : np.ndarray
        local treatment point
    comp : int
        component index of local treatment point
    gamma_kernel : float
        hyperparameter of the RBF kernel
    lam_feasibility : float
        hyperparameter of the feasibility constraint
    lam_first : float
        hyperparameter for regularization of the first stage constraint
    lam_second : float
        hyperparameter for the regularization of the second stage constraint
    type : str, optional
        which functional we are looking at, by default "gradient"

    Returns
    -------
    float
        value of objective function
    """
     
    # Kernel matrices for existing data
    Kzz = rbf_kernel(z, z, gamma_kernel)
    Kxx = rbf_kernel(x, x, gamma_kernel)

    #Kxxbar_grad = gradient_rbf_kernel(x, xbar, gamma_kernel, comp)[..., np.newaxis]
    Kxxbar_grad = linear_functional_rbf_kernel(x, xbar, gamma_kernel, comp, type="gradient")[..., np.newaxis]
    #M = np.sqrt(Kzz)@np.linalg.solve(U / (n * delta **2) * Kzz + np.identity(n), np.sqrt(Kzz))
    M = Kzz
    
    def compute_pinv(Kxx, M, Kxxbar_grad, lam_first, lam_second, regularization=1e-8):
        # Add regularization to the diagonal elements
        regularized_matrix = Kxx @ M @ Kxx + 4 * (lam_first * lam_second) * Kxx + regularization * np.eye(Kxx.shape[0])
        
        # Compute the pseudo-inverse
        pinv = np.linalg.pinv(regularized_matrix) @ Kxxbar_grad
    
        return pinv
    
    pinv = np.linalg.solve(Kxx@M@Kxx + 4 * (lam_first * lam_second )  * Kxx, Kxxbar_grad)
    #pinv = compute_pinv(Kxx, M, Kxxbar_grad, lam_first, lam_second)
    obj = - 2 / lam_feasibility * pinv.T@Kxxbar_grad

    obj_abs = np.linalg.norm(obj, 2)

    return obj_abs.squeeze()


def kernel_minmax_functional(xbar, x, y, z, gamma_kernel, comp, lam_feasibility, lam_first, lam_second, minimum=True, type="gradient"):
    """
    Value of upper resp. lower bound of the objective function.

    Parameters
    ----------
    xbar : np.ndarray
        treatment point
    x : np.ndarray
        treatment sample values
    y : np.ndarray
        outcome sample values
    z : np.ndarray
        instrument sample values
    gamma_kernel : float
        hyperparameter of the RBF kernel
    comp : int
        component index of the treatment point
    lam_feasibility : float
        hyperparameter of the feasibility constraint
    lam_first : float
        hyperparameter of the first stage constraint
    lam_second : float
        hyperparameter of the second stage constraint
    minimum : bool, optional
        whether to compute the upper or lower bound, by default True
    type : str, optional
        which functional is relevant, by default "gradient"

    Returns
    -------
    np.ndarray
        theta parameter for lower resp. upper bound, theta@Kxxbar
    """
    
    Kxx = rbf_kernel(x, x, gamma_kernel)
    Kzz = rbf_kernel(z, z, gamma_kernel)
    Kxxbar_grad = linear_functional_rbf_kernel(x, xbar, gamma_kernel, comp, type="gradient")[..., np.newaxis]
    M = Kzz
    regularization = 4 * (lam_first * lam_second)  * Kxx
    mat = Kxx @ M @ Kxx + regularization
    pinv = np.linalg.pinv(mat)
    
    grad_var = Kxxbar_grad if minimum else - Kxxbar_grad
    theta_hat = pinv@(Kxx@M@y + 1 / lam_feasibility * grad_var)

    return theta_hat


def gradient_rbf_kernel(x, xbar, gamma, comp):
    """
    Gradient of the RBF kernel with respect to the comp^th dimension of xbar.

    Parameters
    ----------
    x : np.ndarray
        treatment samples
    xbar : np.ndarray
        treatment point
    gamma : np.float
        hyperparameter of the RBF kernel
    comp : int
        component index of the treatment point

    Returns
    -------
    np.ndarray
        Gradient kernel matrix
    """
    
    # Compute the RBF kernel value using sklearn (note: rbf_kernel expects 2D array inputs)
    K = rbf_kernel(x, xbar.reshape(1, -1), gamma=gamma)
    
    # Calculate the derivative of K with respect to the comp^th dimension
    # First, reshape xbar to align with the shape expected by sklearn (1, dx)
    xbar_reshaped = xbar.reshape(1, -1)
    diff = x - xbar_reshaped
    dK_dcomp = diff[:, comp] * K.flatten() * gamma * 2
    
    return dK_dcomp


def integral_rbf_kernel(x, xbar, gamma, comp):
    """
    Integral of the RBF kernel with respect to the comp^th dimension of xbar.

    Parameters
    ----------
    x : np.ndarray
        treatment samples
    xbar : np.ndarray
        treatment point
    gamma : np.float
        hyperparameter of the RBF kernel
    comp : int
        component index of the treatment point

    Returns
    -------
    np.ndarray
        Integral kernel matrix
    """

    results = []

    # Define the 1D function to integrate
    def integrand(x_i, x_sample):
        # Construct the full x vector with x_i in the d-th position
        xbar_ = xbar.copy()
        xbar_[0][comp] = x_i
        return rbf_kernel(x_sample, xbar_, gamma)

    # Perform the integral for each sample
    for x_sample in x:
        x_sample = x_sample[..., np.newaxis].T
        result, _ = quad(integrand, xbar.squeeze()[comp] - 0.5, xbar.squeeze()[comp] + 0.5, args=(x_sample,))
        results.append(result)
    
    return np.array(results)
 

def linear_functional_rbf_kernel(x, xbar, gamma, comp, type="gradient"):
    """
    Wrapper function for the gradient and integral of the RBF kernel.

    Parameters
    ----------
    x : np.ndarray
        treatment samples
    xbar : np.ndarray
        treatment point
    gamma : np.float
        hyperparameter of the RBF kernel
    comp : int
        component index of the treatment point
    type : str, optional
        type of function, by default "gradient"

    Returns
    -------
    np.ndarray
        Kernel matrix of functional defined in type

    Raises
    ------
    NotImplementedError
        _description_
    """
    if type == "gradient":
        dK_dcomp = gradient_rbf_kernel(x, xbar, gamma, comp)
    elif type == "integral":
        dK_dcomp = integral_rbf_kernel(x, xbar, gamma, comp)
    else:
        raise NotImplementedError("Only gradient is implemented for now.")
    
    return dK_dcomp
