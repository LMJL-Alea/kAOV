import numpy as np
import pandas as pd
import pandas.testing as pdt
import torch as to
import torch.testing as tot

from kaov import AOV, Data


def dummy_data(n_row=100, n_col=100, n_factor=1, n_group=[2,], seed=None):
    """
    Generate a synthetic data set for testing statistical models.

    The function creates a `pandas.DataFrame` filled with random Gaussian
    values (mean = 0, std = 1) and adds one‑hot‑encoded categorical factor
    columns.  It also returns the formula strings that can be fed directly
    to modelling libraries such as `statsmodels` or `patsy`.

    Note: there is no separation between groups defined by factors.

    Args:
        n_row (int, optional): Number of rows (observations) in the dataset.
            Defaults to 100.
        n_col (int, optional): Number of columns (numeric features) in the
            dataset. Defaults to 100.
        n_factor (int, optional): Number of categorical factor columns to
            add to the dataset. Defaults to `1`.
        n_group (list of int, optional): List specifying the number of
            categories (levels) for each factor column. The length of this
            list must equal `n_factor`. Defaults to `[2,]`.
        seed (int, optional): Random seed for reproducibility. If None,
            uses system entropy. Defaults to None.

    Returns:
        data (pandas.DataFrame): The generated data set. It contains
            `n_col` numeric columns named `col1` … `col{n_col}` and
            `n_factor` categorical columns named `factor1` … `factor{n_factor}`.
        input_formula (str): A formula string that encodes the factor
            columns using one‑hot encoding, e.g.
            `"C(factor1, OneHot) + C(factor2, OneHot)"`.
        output_formula (str): A formula string that lists all numeric
            columns, e.g. `"col1 + col2 + … + col{ncol}"`.

    Raises:
        ValueError: If the length of `n_group` does not equal `n_factor`.

    Example:
        >>> data, input_formula, output_formula = dummy_data(
        ...     nrow=10,
        ...     ncol=5,
        ...     n_factor=2,
        ...     n_group=[3, 2],
        ...     seed=42,
        ... )
        >>> data.shape
        (50, 7)
        >>> data.head()
           col1  col2  col3  col4  col5 factor1 factor2
        0  0.33 -0.12  1.07 -0.44  0.55       c2       c1
        1 -0.23  0.78 -0.91  0.12 -0.34       c1       c2
        ...
        >>> input_formula
        'C(factor1, OneHot) + C(factor2, OneHot)'
        >>> output_formula
        'col1 + col2 + col3 + col4 + col5'
    """
    # check input
    if n_factor != len(n_group):
        raise ValueError(
            "`n_group` list length should be equal to `n_factor`."
        )
    # generate random data (under H0, no separation)
    rng = np.random.default_rng(seed=seed)
    data_array = rng.normal(loc=0, scale=1, size=(n_row, n_col))

    # create a data frame from random gaussian data
    data = pd.DataFrame(
        data=data_array,
        columns=[f"col{i+1}" for i in range(n_col)]
    )

    # create meta data (add factor columns)
    for i in range(n_factor):

        group_ind = np.sort(
            np.tile(
                np.array([f"c{i+1}" for i in range(n_group[i])]),
                (n_row+n_group[i]) // n_group[i]
            )[:n_row]
        )

        rng.shuffle(group_ind)

        data[f"factor{i+1}"] = group_ind

    # formulae
    output_formula = " + ".join([f"col{i+1}" for i in range(n_col)])
    input_formula = " + ".join([f"C(factor{i+1}, OneHot)" for i in range(n_factor)])

    # output
    return data, input_formula, output_formula


def assert_equal_data_obj(
    data1: Data, data2: Data, rtol: float = 0, atol: float = 1e-5
):
    """
    Function to compare two Data objects and assert that they are
    (almost) equal or not.

    See <https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close>
    for `atol` and `rtol` definitions.

    Args:
        data1 (kaov.Data): first Data object to compare.
        data2 (kaov.Data): second Data object to compare.
        atol (float): absolute tolerance for comparison.
        rtol (float): relative tolerance for comparison.

    Returns: no return value.

    Raises:
        AssertError: If one check fails.
    """

    # check torch.Tensor attribute equality
    for attr in ['_ProjImX', '_ProjImXorthogonal', '_XXinv', 'endog', 'exog']:
        tot.assert_close(
            getattr(data1, attr), getattr(data2, attr), rtol=rtol, atol=atol
        )

    # check pandas.Index attribute equality
    for attr in ['endog_names', 'exog_names', 'index']:
        pdt.assert_index_equal(getattr(data1, attr), getattr(data2, attr))

    # check pandas.DataFrame attribute equality
    pdt.assert_frame_equal(data1.meta, data2.meta)

    # check scalar attribute equality
    for attr in ['nlvl', 'nobs', 'nvar', 'nystrom']:
        assert getattr(data1, attr) == getattr(data2, attr)


def assert_equal_aov_obj(
    kfit1: AOV, kfit2: AOV, trunc: int | None = None,
    rtol: float = 0, atol: float = 1e-5
):
    """
    Function to compare two AOV objects and assert that they are
    (almost) equal or not.

    See <https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close>
    or <https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html>
    for `atol` and `rtol` definitions.

    Note: partial comparison to check data input and result output (test
    statistic and p-values) if any.

    Args:
        kfit1 (kaov.AOV): first AOV object to compare.
        kfit2 (kaov.AOV): second AOV object to compare.
        trunc (int | None): maximum truncation level (i.e. maximum dimension
            in embedded space to consider). `None` by default and no
            truncation is done.
        atol (float): absolute tolerance for comparison.
        rtol (float): relative tolerance for comparison.

    Returns: no return value.

    Raises:
        AssertError: If one check fails.
    """
    # TODO: test attribute values after running kaov.AOV.test()

    # factor info (dict)
    assert kfit1._factor_info == kfit2._factor_info
    # bandwidth
    tot.assert_close(
        kfit1.computed_bandwidth, kfit2.computed_bandwidth,
        rtol=rtol, atol=atol
    )
    # data
    assert_equal_data_obj(kfit1.data, kfit2.data)
    # data nystrom
    if kfit1.data_nystrom is None or kfit2.data_nystrom is None:
        assert kfit1.data_nystrom == kfit2.data_nystrom
    else:
        assert_equal_data_obj(kfit1.data_nystrom, kfit2.data_nystrom)

    # diagnostics
    assert kfit1.diagnostics == kfit2.diagnostics

    # formula
    assert kfit1.formula == kfit2.formula

    # kernel (function)
    tot.assert_close(
        kfit1.kernel(kfit1.data.endog[:5], kfit1.data.endog[:5]),
        kfit2.kernel(kfit2.data.endog[:5], kfit2.data.endog[:5]),
        rtol=rtol, atol=atol
    )

    # kernel parameter (scalar)
    for attr in ['kernel_bandwidth', 'kernel_function', 'kernel_median_coef']:
        assert getattr(kfit1, attr) == getattr(kfit2, attr)


def assert_vec_equal_abs_one(
    vec: np.array | to.Tensor, rtol: float = 0, atol: float = 1e-11
):
    """Check that a vector is equal to a vector of 1 in absolute value.

    This function asserts that all elements of the input vector are equal
    to 1 (in absolute value) within the specified tolerances. It supports
    both NumPy arrays and PyTorch tensors.

    Args:
        vec (np.array | torch.Tensor): Input vector to check. Must be a
            NumPy array or a PyTorch tensor.
        rtol (float, optional): Relative tolerance for comparison.
            Defaults to 0.
        atol (float, optional): Absolute tolerance for comparison.
            Defaults to 1e-11.

    Returns:
        None: This function performs assertions and returns nothing.

    Raises:
        TypeError: If `vec` is neither a NumPy array nor a PyTorch tensor.
        AssertionError: If the vector elements are not equal to 1 (in
            absolute value) within the specified tolerances.

    Example:
        >>> import numpy as np
        >>> vec = np.array([1.0, -1.0, 1.0])
        >>> assert_vec_equal_abs_one(vec)
        # Passes since all elements are ±1
        >>> vec = np.array([1.0, 0.5, 1.0])
        >>> assert_vec_equal_abs_one(vec)
        # Raises AssertionError since 0.5 is not ±1

    Notes:
        - The function checks that the absolute value of each element is
          equal to 1, meaning both 1 and -1 are acceptable values.
        - The tolerance parameters follow the same conventions as
          `numpy.testing.assert_allclose` and
          `torch.testing.assert_close`.
    """
    if isinstance(vec, np.ndarray):
        np.testing.assert_allclose(
            np.abs(vec), np.ones(vec.shape, dtype=vec.dtype),
            atol=atol, rtol=rtol
        )
    elif isinstance(vec, to.Tensor):
        to.testing.assert_close(
            to.abs(vec), to.ones(vec.shape, dtype=vec.dtype),
            atol=atol, rtol=rtol
        )
    else:
        raise TypeError(
            "Input `vec` should be a Numpy array or a Torch tensor."
        )


def assert_same_eigenvectors(
    a: np.ndarray | to.Tensor, b: np.ndarray | to.Tensor,
    rtol: float = 0, atol: float = 1e-11
):
    """Assert that two sets of ordered eigenvectors are identical up
    to a rotation of angle pi.

    Assert that two sets of ordered eigenvectors (by increasing or decreasing
    eigenvalue order) are identical up to a rotation of angle pi, i.e.
    assert that for any column index j, we have `a[:, j] == b[:, j]` or
    `a[:, j] == -b[:, j]`, meaning that scalar product
    <a[:, j], b[:, j]> = 1 or -1.

    Args:
        a (np.ndarray | torch.Tensor): First set of eigenvectors. Must be a
            NumPy array or a PyTorch tensor.
        b (np.ndarray | torch.Tensor): Second set of eigenvectors. Must be a
            NumPy array or a PyTorch tensor.
        rtol (float, optional): Relative tolerance for comparison.
            Defaults to 0.
        atol (float, optional): Absolute tolerance for comparison.
            Defaults to 1e-11.

    Returns:
        None: This function performs assertions and returns nothing.

    Raises:
        AssertionError: If the input types are invalid, shapes don't match,
            or the eigenvectors are not identical up to sign.
        AssertionError: If the scalar products are not equal to 1 or -1
            within the specified tolerances.

    Example:
        >>> import numpy as np
        >>> a = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([[1.0, 0.0], [0.0, -1.0]])
        >>> assert_equal_eigenvectors(a, b)
        # Passes since columns are either identical or negated
        >>> c = np.array([[1.0, 0.0], [0.0, 0.5]])
        >>> assert_equal_eigenvectors(a, c)
        # Raises AssertionError since second column differs

    Notes:
        - Both input arrays must be of the same type (both NumPy arrays or
          both PyTorch tensors).
        - The function checks that for each column j, the scalar product
          <a[:, j], b[:, j]> is either 1 or -1, meaning the eigenvectors
          are identical up to a sign flip.
        - The tolerance parameters follow the same conventions as
          `numpy.testing.assert_allclose` and
          `torch.testing.assert_close`.
        - This function is useful for comparing eigenvectors obtained from
          different eigendecomposition implementations, where eigenvectors
          may differ by a sign due to numerical considerations.
    """

    # check type
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray) or \
        isinstance(a, to.Tensor) and isinstance(b, to.Tensor)

    # check dimension
    assert a.shape == b.shape

    # setup required functions
    einsum_fun = np.einsum if isinstance(a, np.ndarray) else to.einsum

    abs_fun = np.abs if isinstance(a, np.ndarray) else to.abs

    # compute column-wise scalar product
    col_scal_prod = einsum_fun('ij,ij->j', a, b)

    # check results (assert that all scalar product result are 1 or -1)
    assert_vec_equal_abs_one(abs_fun(col_scal_prod), rtol=rtol, atol=atol)
