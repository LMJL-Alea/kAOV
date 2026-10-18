import warnings

import torch as to

from .linalg import distances


def _median(x, y=None):
    if y is None:
        dtot = distances(x)
    else:
        dtot = distances(to.cat((x, y)))
    median = dtot.median()
    if median == 0:
        warnings.warn(
            "The median is null. To avoid a kernel with zero bandwidth, " +
            "the median is replaced by the mean."
        )
        mean = dtot.mean()
        if mean == 0:
            warnings.warn('The whole dataset is null.')
        return mean
    else:
        return median


def linear_kernel(x, y=None):
    """
    Computes the standard linear kernel k(x,y)= <x,y>.

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.

    Returns
    -------
    K : torch.Tensor
        Kernel matrix (gram).

    """
    if y is None:
        y = x.clone()
    K = to.matmul(x, y.T)
    return K


def gauss_kernel(x, y=None, sigma=1):
    """
    Computes the standard Gaussian kernel

    k(x,y) = exp(- ||x-y||^2 / (2 * sigma^2)).

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.
    sigma : int
        Standard deviation for Gaussian kernel, 1 by default.

    Returns
    -------
    K : torch.Tensor
        Kernel matrix (gram).

    """
    d = distances(x, y)   # [sq_dists]_ij = ||X_j - Y_i||^2
    K = to.exp(-d / (2 * sigma**2))  # Gram matrix
    return K


def gauss_kernel_median(x, y=None, bandwidth='median', median_coef=1,
                        return_bandwidth=False):
    """
    Computes the gaussian kernel with bandwidth set as the median of the
    distances between pairs of observations (`bandwidth='median'`).

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.
    bandwidth : 'median' or float
        If 'median' (default), the bandwidth is calculated with the median
        method. If float, the value is assigned to the bandwidth.
    median_coef : float
        Coefficient in the badwidth calculation, 1 by default.
    return_bandwidth : bool
        Bandwidth, calculated with the median method. False by default.

    Returns
    -------
    kernel : callable
        Kernel function.
    if return_bandwidth=True:
    computed_bandwidth : float
        Bandwidth, calculated with the median method.
    """
    if bandwidth == 'median':
        computed_bandwidth = to.sqrt(_median(x, y) * median_coef)
    else:
        computed_bandwidth = bandwidth
    kernel = lambda a, b=None: gauss_kernel(x=a, y=b, sigma=computed_bandwidth)
    if return_bandwidth:
        return (kernel, computed_bandwidth)
    else:
        return kernel
