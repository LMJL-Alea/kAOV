import warnings

import pytest
import numpy as np
import torch as to

from kaov.linalg import distances, ordered_eigsy, _calculate_XXinv_and_ProjImX
from kaov.testing import dummy_data, assert_same_eigenvectors


class Test_distances:
    """Test suite for distances function."""

    # Output
    def test_returns_tensor(self):
        """Test that return type is torch.Tensor."""
        x = to.tensor([[1.0, 2.0]])
        result = distances(x)

        assert isinstance(result, to.Tensor)

    # Basic functionality tests
    def test_square_matrix(self):
        """Test with square input matrix."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        result = distances(x)

        assert result.shape == (2, 2)
        expected = to.tensor([[0.0, 2.0], [2.0, 0.0]])
        to.testing.assert_close(result, expected)

    def test_rectangular_matrix(self):
        """Test with rectangular input matrix."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = distances(x)

        assert result.shape == (3, 3)
        expected = to.tensor([
            [0.0, 2.0, 8.0],
            [2.0, 0.0, 2.0],
            [8.0, 2.0, 0.0]
        ])
        to.testing.assert_close(result, expected)

    def test_two_different_inputs(self):
        """Test with two different input tensors."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        y = to.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        result = distances(x, y)

        assert result.shape == (2, 3)
        expected = to.tensor([
            [8.0, 18.0, 32.0],
            [2.0, 8.0, 18.0]
        ])
        to.testing.assert_close(result, expected)

    def test_y_is_none_uses_x(self):
        """Test that y=None uses x as second input."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        result_with_y = distances(x, x)
        result_without_y = distances(x)

        to.testing.assert_close(
            result_with_y, result_without_y, rtol=0, atol=1e-11
        )

    # Value correctness tests
    def test_zero_distance(self):
        """Test that distance to self is zero."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = distances(x)

        to.testing.assert_close(
            to.diag(result), to.zeros(2), rtol=0, atol=1e-11
        )

    def test_known_values(self):
        """Test with known distance values."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]])
        result = distances(x)

        # Distance between (0,0) and (3,4) is 5, squared is 25
        assert result[0, 1] == 25.0
        assert result[1, 0] == 25.0

    def test_single_row(self):
        """Test with single row tensor."""
        x = to.tensor([[1.0, 2.0, 3.0]])
        result = distances(x)

        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = distances(x)

        to.testing.assert_close(result, result.T, rtol=0, atol=1e-11)

    # Error handling tests
    def test_1d_input_raises(self):
        """Test that 1D input raises ValueError."""
        x = to.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2-dimensional"):
            distances(x)

    def test_3d_input_raises(self):
        """Test that 3D input raises ValueError."""
        x = to.rand(2, 3, 4)
        with pytest.raises(ValueError, match="2-dimensional"):
            distances(x)

    def test_mismatched_dimensions_raises(self):
        """Test that mismatched second dimensions raise ValueError."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = to.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="same second dimension"):
            distances(x, y)

    # Edge cases
    def test_empty_tensor(self):
        """Test with empty tensor."""
        x = to.empty(0, 3)
        result = distances(x)

        assert result.shape == (0, 0)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float32)
        result = distances(x)

        assert result.dtype == to.float32
        expected = to.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=to.float32)
        to.testing.assert_close(result, expected)

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float64)
        result = distances(x)

        assert result.dtype == to.float64
        expected = to.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=to.float64)
        to.testing.assert_close(result, expected)


class Test_ordered_eigsy:
    """Test suite for ordered_eigsy function."""

    @pytest.fixture
    def dummy_data(self):
        """Generate dummy data for testing."""
        data, _, _ = dummy_data(
            n_row=20, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )
        return data.iloc[:, :10].values.copy()

    @pytest.fixture
    def sym_matrix(self, dummy_data):
        """Create a symmetric matrix from dummy data."""
        data_tensor = to.from_numpy(dummy_data)
        return to.matmul(data_tensor, data_tensor.T)

    @pytest.fixture
    def expected_decomposition(self, sym_matrix):
        """Compute expected eigen decomposition using numpy."""
        exp_sp, exp_ev = np.linalg.eigh(sym_matrix.numpy())
        # Revert to decreasing eigen value order
        exp_sp = np.flip(exp_sp)
        exp_ev = np.flip(exp_ev, axis=1)
        return exp_sp, exp_ev

    def test_output_types(self, sym_matrix):
        """Test that output types are correct."""
        sp, ev = ordered_eigsy(sym_matrix, eps=None, clip=False)

        assert isinstance(sp, to.Tensor)
        assert isinstance(ev, to.Tensor)

    def test_output_shapes_no_clipping(self, sym_matrix):
        """Test output shapes without clipping."""
        sp, ev = ordered_eigsy(sym_matrix, eps=None, clip=False)

        assert list(sp.shape) == [sym_matrix.shape[0]]
        assert list(ev.shape) == [sym_matrix.shape[0], sym_matrix.shape[0]]

    def test_output_shapes_with_clipping(self, sym_matrix):
        """Test output shapes with clipping."""
        with pytest.warns(UserWarning):
            sp, ev = ordered_eigsy(sym_matrix, eps=1e-12, clip=True)

        assert sp.shape[0] < sym_matrix.shape[0]
        assert list(ev.shape) == [sym_matrix.shape[0], sp.shape[0]]

    def test_eigendecomp_no_clipping(self, sym_matrix, expected_decomposition):
        """Test eigenvectors match numpy result without clipping."""
        exp_sp, exp_ev = expected_decomposition
        sp, ev = ordered_eigsy(sym_matrix, eps=None, clip=False)

        # Check eigenvalues
        np.testing.assert_allclose(
            sp.numpy(), exp_sp, rtol=0, atol=1e-11
        )

        # Check eigenvectors for non-null eigenvalues
        mask = sp >= 1e-12
        assert_same_eigenvectors(
            ev.numpy()[:, mask],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

    def test_eigendecomp_with_clipping_default_eps(
        self, sym_matrix, expected_decomposition
    ):
        """Test eigenvectors with clipping and default eps."""
        exp_sp, exp_ev = expected_decomposition
        with pytest.warns(UserWarning, match="Clipping"):
            sp, ev = ordered_eigsy(sym_matrix, eps=None, clip=True)

        # Check eigenvalues match first k eigenvalues
        np.testing.assert_allclose(
            sp.numpy(), exp_sp[:sp.shape[0]], rtol=0, atol=1e-11
        )

        # Check eigenvectors for non-null eigenvalues
        mask = sp >= 1e-12
        assert_same_eigenvectors(
            ev.numpy()[:, mask],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

    def test_eigendecomp_with_clipping_custom_eps(
        self, sym_matrix, expected_decomposition
    ):
        """Test eigenvectors with clipping and custom eps."""
        exp_sp, exp_ev = expected_decomposition
        with pytest.warns(UserWarning, match="Clipping"):
            sp, ev = ordered_eigsy(sym_matrix, eps=1e-12, clip=True)

        # Check eigenvalues match first k eigenvalues
        np.testing.assert_allclose(
            sp.numpy(), exp_sp[:sp.shape[0]], rtol=0, atol=1e-11
        )

        # Check eigenvectors
        assert_same_eigenvectors(
            ev.numpy(),
            exp_ev[:, :sp.shape[0]],
            rtol=0, atol=1e-12
        )

    def test_no_warning_without_clipping(self, sym_matrix):
        """Test that no warning is emitted when clip=False."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Convert warnings to errors
            ordered_eigsy(sym_matrix, eps=None, clip=False)

    def test_no_warning_when_no_clipping_needed(self):
        """Test that no warning is emitted when all eigenvalues are above
        threshold."""
        # Create a full-rank matrix
        matrix = to.eye(5) * 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sp, ev = ordered_eigsy(matrix, eps=1e-12, clip=True)

        assert sp.shape[0] == 5

    def test_decreasing_eigenvalue_order(self, sym_matrix):
        """Test that eigenvalues are in decreasing order."""
        sp, _ = ordered_eigsy(sym_matrix, eps=None, clip=False)

        # Check eigenvalues are decreasing
        assert all(sp[i] >= sp[i+1] for i in range(len(sp) - 1))

    def test_identity_matrix(self):
        """Test with identity matrix."""
        matrix = to.eye(5)
        sp, ev = ordered_eigsy(matrix, eps=None, clip=False)

        # Particular case: flipping required again
        # (because getting rotated identity)
        ev = ev.flip(dims=(1,))

        # All eigenvalues should be 1
        np.testing.assert_allclose(sp.numpy(), np.ones(5), rtol=0, atol=1e-12)

        # Eigenvectors should form identity (up to sign)
        assert_same_eigenvectors(ev.numpy(), np.eye(5), rtol=0, atol=1e-12)

    def test_diagonal_matrix(self):
        """Test with diagonal matrix."""
        matrix = to.diag(to.tensor([4.0, 3.0, 2.0, 1.0]))
        sp, ev = ordered_eigsy(matrix, eps=None, clip=False)

        # Eigenvalues should be sorted in decreasing order
        np.testing.assert_allclose(
            sp.numpy(), [4.0, 3.0, 2.0, 1.0], rtol=0, atol=1e-12
        )

        # Eigenvectors should be identity (up to sign)
        assert_same_eigenvectors(ev.numpy(), np.eye(4), rtol=0, atol=1e-12)

    def test_zero_matrix(self):
        """Test with zero matrix."""
        matrix = to.zeros((5, 5))
        sp, ev = ordered_eigsy(matrix, eps=None, clip=False)

        # All eigenvalues should be zero
        np.testing.assert_allclose(sp.numpy(), np.zeros(5), rtol=0, atol=1e-12)

    def test_clipping_all_eigenvalues(self):
        """Test when all eigenvalues are clipped."""
        matrix = to.zeros((5, 5))
        with pytest.warns(UserWarning):
            sp, ev = ordered_eigsy(matrix, eps=1e-12, clip=True)

        assert sp.shape[0] == 0
        assert ev.shape == (5, 0)

    def test_increasing_clipping_threshold(self, sym_matrix):
        """Test with custom eps threshold."""
        # Use a high threshold to clip more eigenvalues
        with pytest.warns(UserWarning):
            sp_high_eps, _ = ordered_eigsy(sym_matrix, eps=1.0, clip=True)

        with pytest.warns(UserWarning):
            sp_low_eps, _ = ordered_eigsy(sym_matrix, eps=1e-12, clip=True)

        # Higher eps should clip more eigenvalues
        assert sp_high_eps.shape[0] <= sp_low_eps.shape[0]

    def test_reconstruction_property(self, sym_matrix):
        """Test that the eigendecomposition reconstructs the original matrix."""
        sp, ev = ordered_eigsy(sym_matrix, eps=None, clip=False)

        # Reconstruct matrix: A = V @ diag(s) @ V.T
        reconstructed = ev @ to.diag(sp) @ ev.T

        np.testing.assert_allclose(
            reconstructed.numpy(), sym_matrix.numpy(), rtol=0, atol=1e-10
        )

    def test_orthogonality_of_eigenvectors(self, sym_matrix):
        """Test that eigenvectors are orthogonal."""
        _, ev = ordered_eigsy(sym_matrix, eps=None, clip=False)

        # Check orthogonality: V.T @ V should be identity
        gram = ev.T @ ev
        np.testing.assert_allclose(
            gram.numpy(), np.eye(ev.shape[1]), rtol=0, atol=1e-10
        )

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        for dtype in [to.float32, to.float64]:
            matrix = to.eye(3, dtype=dtype) * 2.0
            sp, ev = ordered_eigsy(matrix, eps=None, clip=False)

            assert sp.dtype == dtype
            assert ev.dtype == dtype

            # Particular case: flipping required again
            # (because getting rotated identity)
            ev = ev.flip(dims=(1,))

            # All eigenvalues should be 1
            np.testing.assert_allclose(
                sp.numpy(), np.ones(3) * 2.0, rtol=0, atol=1e-12
            )

            # Eigenvectors should form identity (up to sign)
            assert_same_eigenvectors(ev.numpy(), np.eye(3), rtol=0, atol=1e-12)

    def test_non_symmetric_matrix(self):
        """
        Test with non-symmetric matrix (should still work but may have
        complex eigenvalues).
        """
        matrix = to.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Should either work or raise an appropriate error
        try:
            sp, ev = ordered_eigsy(matrix, eps=None, clip=False)
            # If it works, eigenvalues should be real
            assert not to.is_complex(sp)
        except RuntimeError:
            pass  # Acceptable if it raises an error for non-symmetric matrices

    def test_single_element_matrix(self):
        """Test with 1x1 matrix."""
        matrix = to.tensor([[3.0]])
        sp, ev = ordered_eigsy(matrix, eps=None, clip=False)

        assert sp.shape == (1,)
        assert ev.shape == (1, 1)
        np.testing.assert_allclose(sp.numpy(), [3.0], rtol=0, atol=1e-12)

    def test_reproducibility(self, sym_matrix):
        """Test that results are reproducible."""
        sp1, ev1 = ordered_eigsy(sym_matrix, eps=None, clip=False)
        sp2, ev2 = ordered_eigsy(sym_matrix, eps=None, clip=False)

        np.testing.assert_allclose(sp1.numpy(), sp2.numpy(), rtol=0, atol=1e-12)
        assert_same_eigenvectors(ev1.numpy(), ev2.numpy(), rtol=0, atol=1e-12)


class Test_calculate_XXinv_and_ProjImX:
    """Test suite for _calculate_XXinv_and_ProjImX function."""

    # Basic functionality tests
    def test_output_shapes(self):
        """Test output shapes for a rectangular matrix."""
        X = to.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        assert XXinv.shape == (2, 2)  # n_features x n_features
        assert ProjImX.shape == (3, 3)  # n_samples x n_samples

    def test_full_rank_square_matrix(self):
        """Test with full-rank square matrix."""
        X = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # For full-rank square matrix, XXinv should be inverse of X^T X
        XX = to.matmul(X.T, X)
        expected_XXinv = to.inverse(XX)
        to.testing.assert_close(XXinv, expected_XXinv, atol=1e-5, rtol=1e-5)

    def test_projection_is_idempotent(self):
        """Test that projection matrix is idempotent."""
        X = to.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # P @ P = P
        result = to.matmul(ProjImX, ProjImX)
        to.testing.assert_close(result, ProjImX, atol=1e-5, rtol=1e-5)

    def test_projection_is_symmetric(self):
        """Test that projection matrix is symmetric."""
        X = to.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _, ProjImX = _calculate_XXinv_and_ProjImX(X)

        to.testing.assert_close(ProjImX, ProjImX.T, atol=1e-5, rtol=1e-5)

    def test_projection_property(self):
        """Test that P @ X = X."""
        X = to.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _, ProjImX = _calculate_XXinv_and_ProjImX(X)

        result = to.matmul(ProjImX, X)
        to.testing.assert_close(result, X, atol=1e-5, rtol=1e-5)

    # Comparison with numpy
    def test_compare_with_numpy_full_rank(self):
        """Test against numpy for full-rank matrix."""
        X_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X = to.from_numpy(X_np)

        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # Compute with numpy
        XX_np = X_np.T @ X_np
        expected_XXinv_np = np.linalg.pinv(XX_np)
        expected_ProjImX_np = X_np @ expected_XXinv_np @ X_np.T

        np.testing.assert_allclose(
            XXinv.numpy(), expected_XXinv_np, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            ProjImX.numpy(), expected_ProjImX_np, rtol=1e-5, atol=1e-5
        )

    def test_compare_with_numpy_rank_deficient(self):
        """Test against numpy for rank-deficient matrix."""
        # Rank-deficient: second column is multiple of first
        X_np = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        X = to.from_numpy(X_np)

        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # Compute with numpy
        XX_np = X_np.T @ X_np
        expected_XXinv_np = np.linalg.pinv(XX_np)
        expected_ProjImX_np = X_np @ expected_XXinv_np @ X_np.T

        np.testing.assert_allclose(
            XXinv.numpy(), expected_XXinv_np, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            ProjImX.numpy(), expected_ProjImX_np, rtol=1e-5, atol=1e-5
        )

    # Known value tests
    def test_identity_matrix(self):
        """Test with identity matrix."""
        X = to.eye(3)
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # X^T X = I, so XXinv = I
        to.testing.assert_close(XXinv, to.eye(3), atol=1e-5, rtol=1e-5)
        # P = X @ I @ X.T = I
        to.testing.assert_close(ProjImX, to.eye(3), atol=1e-5, rtol=1e-5)

    def test_single_column(self):
        """Test with single column matrix."""
        X = to.tensor([[1.0], [2.0], [3.0]])
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # XX = X^T X = 1+4+9 = 14
        # XXinv = 1/14
        assert XXinv.shape == (1, 1)
        assert XXinv[0, 0] == pytest.approx(1.0 / 14.0, abs=1e-5)

        # ProjImX = X @ XXinv @ X.T
        expected_Proj = (1.0 / 14.0) * to.tensor([[1.0, 2.0, 3.0],
                                                  [2.0, 4.0, 6.0],
                                                  [3.0, 6.0, 9.0]])
        to.testing.assert_close(ProjImX, expected_Proj, atol=1e-5, rtol=1e-5)

    def test_orthogonal_columns(self):
        """Test with orthogonal columns."""
        X = to.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # X^T X = [[1, 0], [0, 1]], so XXinv = I
        to.testing.assert_close(XXinv, to.eye(2), atol=1e-5, rtol=1e-5)

        # ProjImX should project onto the xy-plane
        expected_Proj = to.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0]])
        to.testing.assert_close(ProjImX, expected_Proj, atol=1e-5, rtol=1e-5)

    # Edge cases
    def test_single_row(self):
        """Test with single row matrix."""
        X = to.tensor([[1.0, 2.0, 3.0]])
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        assert XXinv.shape == (3, 3)
        assert ProjImX.shape == (1, 1)
        # ProjImX should be 1 (projection onto 1D space)
        assert ProjImX[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_rank_deficient_square(self):
        """Test with rank-deficient square matrix."""
        X = to.tensor([[1.0, 2.0], [2.0, 4.0]])
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # Rank is 1, so XXinv should be rank-1
        assert to.linalg.matrix_rank(XXinv) == 1
        assert ProjImX.shape == (2, 2)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        X = to.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=to.float32)
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        assert XXinv.dtype == to.float32
        assert ProjImX.dtype == to.float32

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        X = to.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=to.float64)
        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        assert XXinv.dtype == to.float64
        assert ProjImX.dtype == to.float64

    # Random matrix tests
    def test_random_matrix_against_numpy(self):
        """Test random matrix against numpy computation."""
        to.manual_seed(42)
        X = to.randn(10, 4)
        X_np = X.numpy()

        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # Compute with numpy
        XX_np = X_np.T @ X_np
        expected_XXinv_np = np.linalg.pinv(XX_np)
        expected_ProjImX_np = X_np @ expected_XXinv_np @ X_np.T

        np.testing.assert_allclose(
            XXinv.numpy(), expected_XXinv_np, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            ProjImX.numpy(), expected_ProjImX_np, rtol=1e-5, atol=1e-5
        )

    def test_random_rank_deficient_against_numpy(self):
        """Test random rank-deficient matrix against numpy."""
        to.manual_seed(123)
        # Create rank-deficient matrix (rank 2 out of 4 columns)
        X_base = to.randn(10, 2)
        X = to.cat([X_base, 2 * X_base, -X_base], dim=1)  # Rank 2
        X_np = X.numpy()

        XXinv, ProjImX = _calculate_XXinv_and_ProjImX(X)

        # Compute with numpy
        XX_np = X_np.T @ X_np
        expected_XXinv_np = np.linalg.pinv(XX_np)
        expected_ProjImX_np = X_np @ expected_XXinv_np @ X_np.T

        np.testing.assert_allclose(
            XXinv.numpy(), expected_XXinv_np, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            ProjImX.numpy(), expected_ProjImX_np, rtol=1e-5, atol=1e-5
        )
