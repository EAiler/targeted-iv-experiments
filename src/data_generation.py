import numpy as np
from typing import Tuple

def generate_covariance_matrix(dx):
    """
    Generates Covariance matrix for the data generation process.

    Parameters
    ----------
    dx : int
        dimensionality of the data

    Returns
    -------
    np.matrix
        covariance matrix of shape (dx+1, dx+1)
    """

    # Generate a random matrix
    A = np.random.randn(dx+1, dx+1)

    # Compute the covariance matrix
    cov_matrix = A.mm(A.t())

    # Make the covariance matrix symmetric
    cov_matrix = (cov_matrix + cov_matrix.t()) / 2

    # Add a small value to the diagonal for numerical stability
    cov_matrix += np.eye(dx+1) * 1e-6

    return cov_matrix


def generate_data(z, dx, h, f, confounding_intensity=0.5) -> Tuple[np.array, np.array]:
    """
    Generates data with confounding affecting both x and y, ensuring that the confounding
    on x and y is correlated.
    

    Parameters
    ----------
    z : np.array
        instrument  
    dx : int
        dimensionality of the treatment
    h : function
        first stage function
    f : function
        second stage function
    confounding_intensity : float, optional
        intensity of the confounding, by default 0.5

    Returns
    -------
    Tuple[np.array, np.array]
       experimental data
    """
    
    n_samples = z.shape[0]

    # Generate a single confounder for each sample to affect both x and y; dimensions (n_samples,)
    # The confounder is generated using a normal distribution with mean 0 and standard deviation 1 and then scaled by confounding_intensity
    confounder = np.random.normal(scale=confounding_intensity, size=(n_samples, 1))

    # Get confounder_x by adding confounder to each row of e_x
    confounder_x = confounder.squeeze()[:, None]  # Broadcasting confounder to match x's shape, i. e. the confounder is added to each row of e_x
    
    # Generate x with confounding via h(z, confounder_x)
    x = h(z, confounder_x)
    
    # Apply the same confounder to y, ensuring correlated confounding effect
    y = f(x) + confounder  # Using confounder directly since y is 1-dimensional
    
    return x, y
