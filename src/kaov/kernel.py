from functools import partial
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
    kernel = partial(gauss_kernel, sigma=computed_bandwidth)
    if return_bandwidth:
        return kernel, computed_bandwidth
    else:
        return kernel, None


def _quantile_dist(
    x: to.Tensor, y: to.Tensor | None = None,
    q: float = 0.5, exclude_zero: bool = True
):
    """
    Compute the quantile of pairwise squared distances.

    Computes the specified quantile of all pairwise squared distances
    between rows of `x` (and `y` if provided). This is commonly used as
    a heuristic for setting the bandwidth parameter in kernel methods
    (e.g., Gaussian kernel).

    Notes
    -----
    - This is an internal function.
    - The default quantile `q=0.5` corresponds to the median.
    - Per `torch.quantile` documentation: "If the requested quantile is
      between two data points, linear interpolation is used."
    - When `exclude_zero=True`, only non-zero distances are used to
      compute the quantile.

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data points.
    y : None or torch.Tensor, optional
        2-d tensor containing additional data points. If None, distances
        are computed within `x` only. Defaults to None.
    q : float, optional
        Quantile to compute, between 0 and 1. Defaults to 0.5 (median).
    exclude_zero : bool, optional
        If True, only non-zero distances are used for quantile
        computation. Defaults to True.

    Returns
    -------
    float
        The quantile of the pairwise squared distances as a scalar
        torch tensor.

    Warns
    -----
    UserWarning
        If all distances are zero (and `exclude_zero=True`), warning
        that the dataset contains only null distances.

    Examples
    --------
    >>> x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> _quantile_dist(x)
    tensor(2.)

    >>> x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> y = to.tensor([[2.0, 2.0], [3.0, 3.0]])
    >>> _quantile_dist(x, y, q=0.25)
    tensor(2.)
    """
    if y is None:
        dtot = distances(x)
    else:
        dtot = distances(to.cat((x, y)))

    if exclude_zero:
        # Filter out zero distances
        dtot = dtot[dtot > 0]

    if dtot.numel() == 0:
        warnings.warn("All distances between observations are zero.")
        return to.tensor(0.0)

    quantile = to.quantile(dtot, q=q)

    return quantile


def gauss_kernel_quantile(
    x: to.Tensor,
    y: to.Tensor | None = None,
    bandwidth: str | float = 'quantile',
    scaling_coef: float = 1,
    return_bandwidth: bool = False,
    q: float = 0.5,
    exclude_zero: bool = True,
):
    """
    Defines the gaussian kernel function with bandwidth set as a chosen
    empirical quantile of the distances between pairs of observations
    (`bandwidth='quantile'`) or with a custom bandwidth.

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.
    bandwidth : 'median' or float
        If 'median' (default), the bandwidth is calculated with the quantile
        method. If float, the value is assigned to the bandwidth.
    scaling_coef : float
        Coefficient to scale the quantile bandwidth, 1 by default.
    return_bandwidth : bool
        Flag to enable/disable returned bandwidth value calculated with the
        median method. False by default.
    q : float, optional
        Quantile to use for bandwidth computation when `bandwidth='median'`.
        Defaults to 0.5 (median).
    exclude_zero : bool, optional
        If True, only non-zero distances are used for bandwidth computation.
        Defaults to True.

    Returns
    -------
    kernel : callable
        Kernel function.
    computed_bandwidth : float or None
        Bandwidth if `return_bandwidth=True`, calculated with the quantile
        method, else `None`.

    Examples
    --------
    >>> x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> kernel, bw = gauss_kernel_median(x, return_bandwidth=True)
    >>> bw
    tensor(1.4142)  # sqrt(2)
    >>> kernel(x).shape
    torch.Size([3, 3])
    """
    if bandwidth == 'quantile':
        computed_bandwidth = to.sqrt(
            _quantile_dist(x, y, q=q, exclude_zero=exclude_zero)
            * scaling_coef
        )
    else:
        computed_bandwidth = bandwidth
    kernel = partial(gauss_kernel, sigma=computed_bandwidth)
    if return_bandwidth:
        return kernel, computed_bandwidth
    else:
        return kernel, None
