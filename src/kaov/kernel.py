import warnings

import torch as to

from .linalg import distances


def linear_kernel(x: to.Tensor, y: to.Tensor | None = None):
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
        y = x
    K = to.matmul(x, y.T)
    return K


def gauss_kernel(x: to.Tensor, y: to.Tensor | None = None, sigma: float = 1):
    """
    Computes the standard Gaussian kernel

    k(x,y) = exp(- ||x-y||^2 / (2 * sigma^2)).

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.
    sigma : float
        Standard deviation for Gaussian kernel, 1 by default.

    Returns
    -------
    K : torch.Tensor
        Kernel matrix (gram).

    """
    d = distances(x, y)   # [sq_dists]_ij = ||X_j - Y_i||^2
    K = to.exp(-d / (2 * sigma**2))  # Gram matrix
    return K


def _median_dist(x: to.Tensor, y: to.Tensor | None = None):
    """
    Compute the median of pairwise squared distances.

    Computes the median of all pairwise squared distances between rows of
    `x` (and `y` if provided). This is commonly used as a heuristic for
    setting the bandwidth parameter in kernel methods (e.g., Gaussian
    kernel).

    Notes
    -----
    - This is an internal function.
    - Per `torch.median` documentation: "The median is not unique for input
    tensors with an even number of elements. In this case the lower of the
    two medians is returned. To compute the mean of both medians, use
    `torch.quantile()` with `q=0.5` instead."

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data points.
    y : None or torch.Tensor, optional
        2-d tensor containing additional data points. If None, distances
        are computed within `x` only. Defaults to None.

    Returns
    -------
    float
        The median of the pairwise squared distances. If the median is
        zero, the mean is returned instead (with a warning).

    Warns
    -----
    UserWarning
        If the median is zero, warning that the median is replaced by
        the mean to avoid a kernel with zero bandwidth.
    UserWarning
        If both the median and mean are zero, warning that the whole
        dataset is null.

    Examples
    --------
    >>> x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> _median_dist(x)
    tensor(2.)

    >>> x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> y = to.tensor([[2.0, 2.0], [3.0, 3.0]])
    >>> _median_dist(x, y)
    tensor(4.)
    """
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


def gauss_kernel_median(
    x: to.Tensor, y: to.Tensor | None = None,
    bandwidth: str | float = 'median', median_coef: float = 1,
    return_bandwidth: bool = False
):
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
        Flag to enable/disable returned bandwidth value calculated with the
        median method. False by default.

    Returns
    -------
    kernel : callable
        Kernel function.
    computed_bandwidth : float or None
        Bandwidth if `return_bandwidth=True`, calculated with the median
        method, else `None`.
    """
    if bandwidth == 'median':
        computed_bandwidth = to.sqrt(_median_dist(x, y) * median_coef)
    else:
        computed_bandwidth = bandwidth
    kernel = lambda a, b=None: gauss_kernel(x=a, y=b, sigma=computed_bandwidth)
    if return_bandwidth:
        return kernel, computed_bandwidth
    else:
        return kernel, None
