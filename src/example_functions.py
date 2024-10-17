import numpy as np


def h_nonlinear(z, confounder, coef):
    """
    First stage function for the nonlinear data generation process.

    Parameters
    ----------
    z : np.ndarray or list
        instrument values
    confounder : np.ndarray or list
        confounder values
    
    Returns
    -------
    np.ndarray
        
    """

    x_base = np.sin(z)  # Apply the sine function element-wise to 'z', results in (n_samples, dz)

    if coef is not None:
        x_base = np.dot(x_base, coef)
    
    dz, dx = coef.shape
    if dx > dz:
        x_base = np.concatenate([x_base[:, :dz], np.ones((z.shape[0], dx - dz))], axis=1)

    # Apply confounding effect, the shape of 1 + confounder[:, None] is (n_samples, 1), the multiplication results in (n_samples, dz) via broadcasting, i. e. the confounder is multiplied to each column of x_base
    x_confounded = x_base * (1 + confounder)

    return x_confounded


def f_nonlinear(x, interaction=False) -> np.array:
    """
    Generates the variable 'y' from the input variable 'x' using a nonlinear relationship.

    If the 'interaction' parameter is True, interaction effects are included in the relationship.

    Parameters
    ----------
    x : np.array
        The input variable with shape (n_samples, dx).
    interaction : bool
        Whether to include interaction effects.

    Returns
    -------
    np.array
        The generated variable 'y' with shape (n_samples,).
    """
    
    # Apply the exponential and sine functions element-wise to 'x'
    x_transformed = np.exp(-x) * np.sin(x)
    
    if interaction and x.shape[1] >= 2:
        # If 'interaction' is True and 'x' has at least 2 columns, include interaction effects via the product of the first two columns of 'x'
        inter_effect = np.prod(x[:, :2], axis=1)
        y = np.sum(x_transformed * inter_effect[:, None], axis=1, keepdims=True)
    else:
        # If 'interaction' is False or 'x' has less than 2 columns, do not include interaction effects
        y = np.sum(x_transformed, axis=1, keepdims=True)
    
    return y


def grad_f_nonlinear(x, comp) -> np.array:
    """
    Gradient of `f_nonlinear` with respect to the `comp`-th component of `x`.

    Parameters
    ----------
    x : np.ndarray
        treatment values
    comp : int
        component index

    Returns
    -------
    np.array
        value of the gradient of `f_nonlinear` with respect to the `comp`-th component of `x`
    """
    x_i = x[:, comp]
    
    # Compute the partial derivative with respect to x_i
    partial_derivatives = np.exp(-x_i) * (np.cos(x_i) - np.sin(x_i))
    
    # Return the partial derivatives, reshaping for consistency with f_nonlinear's output#    
    return partial_derivatives.reshape(-1, 1)


def int_f_nonlinear(x, comp):
    """
    Integral of `f_nonlinear` over the interval [x - 0.5, x + 0.5] with respect to x.

    Parameters
    ----------
    x : np.ndarray
        treatment values
    comp : int
        component index

    Returns
    -------
    np.ndarray
        integral of `f_nonlinear` over the interval [x - 0.5, x + 0.5] with respect to x in the `comp`-th component of `x`
    """
    
    x_lower_i = x[:, comp] - 0.5
    x_upper_i = x[:, comp] + 0.5
    
    # Compute the integral over the interval [x_lower_i, x_upper_i] of the function f(x) = exp(-x) * sin(x) with respect to x
    integral = - 1/2 * np.exp(-x_upper_i) * (np.cos(x_upper_i) + np.sin(x_upper_i)) + 1/2 * np.exp(-x_lower_i) * (np.cos(x_lower_i) + np.sin(x_lower_i))

    return integral







