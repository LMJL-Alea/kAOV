import warnings

import torch as to


def ordered_eigsy(
    matrix: to.Tensor, eps: float | None = None, clip: bool = True
):
    """
    Calculates the eigendecomposition of the input symmetric matrix, using a
    PyTorch [`eigh`](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html)
    solver.

    Eigen values and eigen vectors are stored by decreasing eigen value order.

    Note 1: only positive non-null eigen values and corresponding eigen
    vectors are returned. Eigen values lower than `eps` threshold are
    clipped to zeros and removed.

    Note 2: input matrix is assumed to be symmetrical and positive
    semi-definite, such that its eigen values are positive or null. It
    may happen due to numerical issues that the eigen decomposition finds
    negative eigen values for such matrix anyway. These are also clipped
    to 0.

    Args:
        matrix (torch.Tensor): 2-d array stroting a matrix to decompose.
        eps (float | None): minimum threshold value to clip lower eigen
            values to zeros.
            If `None` (default), then machine precision (given by
            `torch.finfo()`) for matrix dtype is used as threshold.
        clip (bool): flag to enable/disable eigen value clipping. Default is
            `True`.

    Returns:
        sp (torch.Tensor): vector of eigenvalues.
        ev (torch.Tensor): matrix of eigenvectors.
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
