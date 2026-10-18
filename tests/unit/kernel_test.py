import pytest
import torch as to

from kaov.kernel import linear_kernel, gauss_kernel, _median_dist


class Test_linear_kernel:
    """Test suite for linear_kernel function."""

    # Output
    def test_returns_tensor(self):
        """Test that return type is torch.Tensor."""
        x = to.tensor([[1.0, 2.0]])
        result = linear_kernel(x)

        assert isinstance(result, to.Tensor)

    # Basic functionality tests
    def test_square_matrix(self):
        """Test with square input matrix."""
        x = to.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = linear_kernel(x)

        assert result.shape == (2, 2)
        expected = to.tensor([[1.0, 0.0], [0.0, 1.0]])
        to.testing.assert_close(result, expected)

    def test_rectangular_matrix(self):
        """Test with rectangular input matrix."""
        x = to.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = linear_kernel(x)

        assert result.shape == (3, 3)
        expected = to.tensor([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        to.testing.assert_close(result, expected)

    def test_two_different_inputs(self):
        """Test with two different input tensors."""
        x = to.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = to.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        result = linear_kernel(x, y)

        assert result.shape == (2, 3)
        expected = to.tensor([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ])
        to.testing.assert_close(result, expected)

    def test_y_is_none_uses_x(self):
        """Test that y=None uses x as second input."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        result_with_y = linear_kernel(x, x)
        result_without_y = linear_kernel(x)

        to.testing.assert_close(result_with_y, result_without_y)

    # Value correctness tests
    def test_known_values(self):
        """Test with known kernel values."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = to.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = linear_kernel(x, y)

        expected = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        to.testing.assert_close(result, expected)

    def test_identity_kernel(self):
        """Test that kernel of identity is identity."""
        x = to.eye(3)
        result = linear_kernel(x)

        to.testing.assert_close(result, to.eye(3))

    def test_ones_kernel(self):
        """Test with ones matrix."""
        x = to.ones(2, 3)
        result = linear_kernel(x)

        expected = to.full((2, 2), 3.0)
        to.testing.assert_close(result, expected)

    def test_zero_matrix(self):
        """Test with zero matrix."""
        x = to.zeros(3, 2)
        result = linear_kernel(x)

        to.testing.assert_close(result, to.zeros(3, 3))

    def test_symmetry(self):
        """Test that kernel matrix is symmetric."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = linear_kernel(x)

        to.testing.assert_close(result, result.T)

    def test_positive_semidefinite(self):
        """Test that kernel matrix is positive semi-definite."""
        x = to.randn(5, 3)
        result = linear_kernel(x)

        # Check eigenvalues are non-negative
        eigenvalues = to.linalg.eigvalsh(result)
        assert (eigenvalues >= -1e-6).all()

    # Edge cases
    def test_single_row(self):
        """Test with single row tensor."""
        x = to.tensor([[1.0, 2.0, 3.0]])
        result = linear_kernel(x)

        assert result.shape == (1, 1)
        assert result[0, 0] == 14.0  # 1² + 2² + 3²

    def test_single_element(self):
        """Test with single element tensor."""
        x = to.tensor([[5.0]])
        result = linear_kernel(x)

        assert result.shape == (1, 1)
        assert result[0, 0] == 25.0

    def test_empty_tensor(self):
        """Test with empty tensor."""
        x = to.empty(0, 3)
        result = linear_kernel(x)

        assert result.shape == (0, 0)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        x = to.tensor([[1.0, 2.0]], dtype=to.float32)
        result = linear_kernel(x)

        assert result.dtype == to.float32
        assert result.shape == (1, 1)
        assert result[0, 0] == 5.0

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        x = to.tensor([[1.0, 2.0]], dtype=to.float64)
        result = linear_kernel(x)

        assert result.dtype == to.float64
        assert result.shape == (1, 1)
        assert result[0, 0] == 5.0


class Test_gauss_kernel(object):
    """Test suite for gauss_kernel function."""

    # Output
    def test_returns_tensor(self):
        """Test that return type is torch.Tensor."""
        x = to.tensor([[0.0, 0.0]])
        result = gauss_kernel(x)

        assert isinstance(result, to.Tensor)

    # Basic functionality tests
    def test_square_matrix(self):
        """Test with square input matrix."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        result = gauss_kernel(x)

        assert result.shape == (2, 2)
        # k(0,0) = exp(0) = 1, k(0,1) = exp(-2/2) = exp(-1)
        expected = to.tensor([[1.0, to.exp(to.tensor(-1.0))],
                              [to.exp(to.tensor(-1.0)), 1.0]])
        to.testing.assert_close(result, expected)

    def test_rectangular_matrix(self):
        """Test with rectangular input matrix."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = gauss_kernel(x)

        assert result.shape == (3, 3)
        expected = to.tensor([
            [1.0, to.exp(to.tensor(-1.0)), to.exp(to.tensor(-4.0))],
            [to.exp(to.tensor(-1.0)), 1.0, to.exp(to.tensor(-1.0))],
            [to.exp(to.tensor(-4.0)), to.exp(to.tensor(-1.0)), 1.0]
        ])
        to.testing.assert_close(result, expected)

    def test_two_different_inputs(self):
        """Test with two different input tensors."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        y = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = gauss_kernel(x, y)

        assert result.shape == (2, 3)
        expected = to.tensor([
            [1.0, to.exp(to.tensor(-1.0)), to.exp(to.tensor(-4.0))],
            [to.exp(to.tensor(-1.0)), 1.0, to.exp(to.tensor(-1.0))]
        ])
        to.testing.assert_close(result, expected)

    def test_y_is_none_uses_x(self):
        """Test that y=None uses x as second input."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        result_with_y = gauss_kernel(x, x)
        result_without_y = gauss_kernel(x)

        to.testing.assert_close(result_with_y, result_without_y)

    # Value correctness tests
    def test_diagonal_is_one(self):
        """Test that diagonal elements are 1 (distance to self is 0)."""
        x = to.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = gauss_kernel(x)

        to.testing.assert_close(to.diag(result), to.ones(3))

    def test_known_values_sigma_1(self):
        """Test with known values for sigma=1."""
        x = to.tensor([[0.0, 0.0], [1.0, 0.0]])
        result = gauss_kernel(x)

        # Distance between (0,0) and (1,0) is 1, so k = exp(-1/2)
        expected_off_diag = to.exp(to.tensor(-0.5))
        expected = to.tensor([[1.0, expected_off_diag],
                              [expected_off_diag, 1.0]])
        to.testing.assert_close(result, expected)

    def test_known_values_sigma_2(self):
        """Test with known values for sigma=2."""
        x = to.tensor([[0.0, 0.0], [2.0, 0.0]])
        result = gauss_kernel(x, sigma=2)

        # Distance between (0,0) and (2,0) is 2, so k = exp(-4/8) = exp(-0.5)
        expected_off_diag = to.exp(to.tensor(-0.5))
        expected = to.tensor([[1.0, expected_off_diag],
                              [expected_off_diag, 1.0]])
        to.testing.assert_close(result, expected)

    def test_identical_points(self):
        """Test with identical points."""
        x = to.tensor([[1.0, 2.0], [1.0, 2.0]])
        result = gauss_kernel(x)

        expected = to.ones(2, 2)
        to.testing.assert_close(result, expected)

    def test_zero_matrix(self):
        """Test with zero matrix."""
        x = to.zeros(3, 2)
        result = gauss_kernel(x)

        expected = to.ones(3, 3)
        to.testing.assert_close(result, expected)

    def test_symmetry(self):
        """Test that kernel matrix is symmetric."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = gauss_kernel(x)

        to.testing.assert_close(result, result.T)

    def test_values_between_0_and_1(self):
        """Test that all values are between 0 and 1."""
        x = to.randn(5, 3)
        result = gauss_kernel(x)

        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_positive_definite(self):
        """Test that kernel matrix is positive definite."""
        x = to.randn(5, 3)
        result = gauss_kernel(x)

        # Check eigenvalues are positive
        eigenvalues = to.linalg.eigvalsh(result)
        assert (eigenvalues > 0).all()

    # Sigma parameter tests
    def test_sigma_increases_similarity(self):
        """Test that larger sigma gives larger kernel values."""
        x = to.tensor([[0.0, 0.0], [2.0, 0.0]])

        result_sigma_1 = gauss_kernel(x, sigma=1)
        result_sigma_2 = gauss_kernel(x, sigma=2)
        result_sigma_10 = gauss_kernel(x, sigma=10)

        # Off-diagonal values should increase with sigma
        assert result_sigma_1[0, 1] < result_sigma_2[0, 1]
        assert result_sigma_2[0, 1] < result_sigma_10[0, 1]

    def test_sigma_zero_nan(self):
        """Test that sigma=0 raises an error."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        result = gauss_kernel(x, sigma=0)
        assert to.any(to.isnan(result))

    def test_sigma_negative(self):
        """Test with negative sigma (should work since sigma is squared)."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        result_neg = gauss_kernel(x, sigma=-1)
        result_pos = gauss_kernel(x, sigma=1)

        to.testing.assert_close(result_neg, result_pos)

    # Edge cases
    def test_single_row(self):
        """Test with single row tensor."""
        x = to.tensor([[0.0, 0.0]])
        result = gauss_kernel(x)

        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_single_element(self):
        """Test with single element tensor."""
        x = to.tensor([[5.0]])
        result = gauss_kernel(x)

        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_empty_tensor(self):
        """Test with empty tensor."""
        x = to.empty(0, 3)
        result = gauss_kernel(x)

        assert result.shape == (0, 0)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float32)
        result = gauss_kernel(x)

        assert result.dtype == to.float32

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float64)
        result = gauss_kernel(x)

        assert result.dtype == to.float64


class Test_median_dist(object):
    """Test suite for _median_dist function."""

    # Basic functionality tests
    def test_basic_call(self):
        """Test median of pairwise squared distances."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # Distances: 0, 2, 8, 2, 0, 2
        # Sorted: 0, 0, 2, 2, 2, 8
        # Median = (2 + 2) / 2 = 2
        to.testing.assert_close(result, to.tensor(2.0))

    def test_median_with_two_points(self):
        """Test median with two points."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]])

        with pytest.warns(UserWarning, match="median is null"):
            # because bad median definition in PyTorch
            result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # Distance: 0, 25, 25, 0
        # Sorted: 0, 0, 25, 25
        # Median = (0 + 25) / 2 = 12.5 (PyTorch median = 0)
        to.testing.assert_close(result, to.tensor(12.5))

    def test_median_with_y_provided(self):
        """Test median when y is provided."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        y = to.tensor([[2.0, 2.0], [3.0, 3.0]])

        result = _median_dist(x, y)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # All points: (0,0), (1,1), (2,2), (3,3)
        # Distances: 0, 2, 8, 18, 2, 0, 2, 8, 8, 2, 0, 2, 18, 8, 2, 0
        # Sorted: 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 18, 18
        # Median = (2 + 2) / 2 = 2
        to.testing.assert_close(result, to.tensor(2.0))

    def test_y_is_none_uses_x_only(self):
        """Test that y=None computes distances within x only."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.warns(UserWarning, match="median is null"):
            # because bad median definition in PyTorch
            result_without_y = _median_dist(x)

        with pytest.warns(UserWarning, match="median is null"):
            # because bad median definition in PyTorch
            result_with_y = _median_dist(x, x)

        # Check that output is a scalar tensor
        assert isinstance(result_without_y, to.Tensor)
        assert result_without_y.ndim == 0
        assert isinstance(result_with_y, to.Tensor)
        assert result_with_y.ndim == 0

        # Without y:
        # - Distances: 0, 2, 2, 0
        # - Sorted: 0, 0, 2, 2
        # - Median = (0 + 2) / 2 = 1  (PyTorch median = 0)
        # With y:
        # - Distances: 0, 2, 0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0, 2, 0
        # - Sorted: 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2
        # - Median = (0 + 2) / 2 = 1  (PyTorch median = 0)
        to.testing.assert_close(result_without_y, to.tensor(1.0))
        to.testing.assert_close(result_with_y, to.tensor(1.0))

    # Zero median handling tests
    def test_zero_median_replaced_by_mean(self):
        """Test that zero median is replaced by mean with warning."""
        x = to.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 2.0]])

        with pytest.warns(UserWarning, match="median is null"):
            result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # Distances: 0, 0, 1, 0, 0, 1, 1, 1, 0
        # Sorted: 0, 0, 0, 0, 0, 1, 1, 1, 1
        # Median = 0
        # Mean = (0+0+0+0+0+1+1+1+1)/9 = 4/9 ≈ 0.44
        to.testing.assert_close(result, to.tensor(4/9))

    def test_all_zero_dataset(self):
        """Test with all-zero dataset."""
        x = to.zeros(3, 2)

        with pytest.warns(UserWarning, match="median is null"):
            with pytest.warns(UserWarning, match="whole dataset is null"):
                result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # Only distance is 0
        assert result.item() == 0.0

    # Edge cases
    def test_single_point_with_warning(self):
        """Test single point triggers warning for zero median."""
        x = to.tensor([[1.0, 2.0]])

        with pytest.warns(UserWarning, match="median is null"):
            with pytest.warns(UserWarning, match="whole dataset is null"):
                result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # Only distance is 0
        assert result.item() == 0.0

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float32)

        with pytest.warns(UserWarning, match="median is null"):
            # because bad median definition in PyTorch
            result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0
        assert result.dtype == to.float32

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float64)

        with pytest.warns(UserWarning, match="median is null"):
            # because bad median definition in PyTorch
            result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0
        assert result.dtype == to.float64

    # Known value tests
    def test_known_values_regular_grid(self):
        """Test with regular grid points."""
        x = to.tensor([[0.0], [1.0], [2.0], [3.0]])
        result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # Distances: 0,1,4,9, 1,0,1,4, 4,1,0,1, 9,4,1,0
        # Sorted: 0,0,0,0, 1,1,1,1, 1,1, 4,4,4, 9,9
        # Median = (1+1)/2 = 1
        assert result.item() == 1.0

    def test_known_values_irregular(self):
        """Test with irregular spacing."""
        x = to.tensor([[0.0], [2.0], [5.0]])
        result = _median_dist(x)

        # Check that output is a scalar tensor
        assert isinstance(result, to.Tensor)
        assert result.ndim == 0

        # Distances: 0, 4, 25, 4, 0, 9, 25, 9, 0
        # Sorted: 0, 0, 0, 4, 4, 9, 9, 25, 25
        # Median = 4
        assert result.item() == 4.0
