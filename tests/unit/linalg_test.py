import warnings

import pytest
import numpy as np
import torch as to

from kaov.linalg import ordered_eigsy
from kaov.testing import dummy_data, assert_same_eigenvectors


class Test_ordered_eigsy(object):
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
        """Test with non-symmetric matrix (should still work but may have complex eigenvalues)."""
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
