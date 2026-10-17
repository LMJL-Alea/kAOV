import warnings

import torch as to


def ordered_eigsy(
    matrix: to.Tensor, eps: float | None = None, clip: bool = True
):
    """
    Calculate the eigendecomposition of a symmetric matrix.

    Calculates the eigendecomposition of the input symmetric matrix, using a
    PyTorch `eigh <https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html>`_
    solver.

    Eigenvalues and eigenvectors are stored in decreasing eigenvalue order.

    Notes
    -----
    - Only positive non-null eigenvalues and corresponding eigenvectors
      are returned. Eigenvalues lower than `eps` threshold are clipped to
      zeros and removed.
    - Input matrix is assumed to be symmetrical and positive semi-definite,
      such that its eigenvalues are positive or null. It may happen due to
      numerical issues that the eigen decomposition finds negative
      eigenvalues for such matrix anyway. These are also clipped to 0.

    Parameters
    ----------
    matrix : torch.Tensor
        2-d array storing a matrix to decompose.
    eps : float or None, optional
        Minimum threshold value to clip lower eigenvalues to zeros.
        If None (default), then machine precision (given by
        ``torch.finfo()``) for matrix dtype is used as threshold.
    clip : bool, optional
        Flag to enable/disable eigenvalue clipping. Default is True.

    Returns
    -------
    sp : torch.Tensor
        Vector of eigenvalues.
    ev : torch.Tensor
        Matrix of eigenvectors.

    Raises
    ------
    ValueError
        If the input array is not a matrix (i.e. not a 2-dimensional array).
    """
    # eigen decomposition gives sp (eigen values) and ev (eigen vectors)
    # following ascending eigen value order (no need for sorting)
    sp, ev = to.linalg.eigh(matrix)
    # revert to get decreasing order for eigen values
    sp = sp.flip(dims=(0,))
    ev = ev.flip(dims=(1,))
    # clipping?
    if clip:
        # eigen value clipping threshold
        if eps is None:
            eps = to.finfo(matrix.dtype).eps
        # select non null eigen val
        sp_mask = sp >= eps
        # reduc dimension
        reduc_dim = sp_mask.sum()
        # warning
        if reduc_dim < len(sp):
            warnings.warn(
                f"Clipping last {len(sp) - reduc_dim} eigen values " +
                f"out of {len(sp)} dimensions, " +
                f"that are lower than threshold {eps}."
            )

        # output
        return sp[sp_mask], ev[:, sp_mask]
    else:
        # no clipping
        # output
        return sp, ev
