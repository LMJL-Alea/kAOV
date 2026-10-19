import pytest
import torch as to

from kaov.kernel import (
    linear_kernel, gauss_kernel, _median_dist, gauss_kernel_median,
    _quantile_dist, gauss_kernel_quantile
)


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


class Test_gauss_kernel_median(object):
    """Test suite for gauss_kernel_median function."""

    # Output
    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = gauss_kernel_median(x)

        assert isinstance(result, tuple)
        assert len(result) == 2

        result = gauss_kernel_median(x, return_bandwidth=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

    # Basic functionality tests
    def test_returns_kernel_and_none(self):
        """Test that function returns kernel and None by default."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_median(x)

        assert callable(kernel)
        assert bandwidth is None

    def test_returns_kernel_and_bandwidth(self):
        """Test that function returns kernel and bandwidth when requested."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_median(x, return_bandwidth=True)

        assert callable(kernel)
        assert bandwidth is not None
        assert isinstance(bandwidth, to.Tensor)

    # Bandwidth calculation tests
    def test_median_bandwidth_value(self):
        """Test that median bandwidth is correctly computed."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, bandwidth = gauss_kernel_median(x, return_bandwidth=True)

        # Median distance = 2, sqrt(2) ≈ 1.414
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(2.0)))

    def test_median_bandwidth_with_coef(self):
        """Test median bandwidth with coefficient."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, bandwidth = gauss_kernel_median(
            x, median_coef=4, return_bandwidth=True
        )

        # Median distance = 2, sqrt(2 * 4) = sqrt(8) ≈ 2.828
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(8.0)))

    def test_float_bandwidth(self):
        """Test with float bandwidth value."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_median(
            x, bandwidth=2.0, return_bandwidth=True
        )

        assert bandwidth == 2.0
        assert callable(kernel)

    # Kernel behavior tests
    def test_kernel_with_median_bandwidth(self):
        """Test kernel output with median bandwidth."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        # Median distance = 2, so bandwidth = 1 with median_coef = 0.5
        kernel, _ = gauss_kernel_median(x, median_coef=0.5)
        result = kernel(x)

        assert result.shape == (3, 3)

        expected = to.tensor([
            [1.0, to.exp(to.tensor(-1.0)), to.exp(to.tensor(-4.0))],
            [to.exp(to.tensor(-1.0)), 1.0, to.exp(to.tensor(-1.0))],
            [to.exp(to.tensor(-4.0)), to.exp(to.tensor(-1.0)), 1.0]
        ])
        to.testing.assert_close(result, expected)

    def test_kernel_with_float_bandwidth(self):
        """Test kernel output with float bandwidth."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, _ = gauss_kernel_median(x, bandwidth=1.0)
        result = kernel(x)

        assert result.shape == (3, 3)

        expected = to.tensor([
            [1.0, to.exp(to.tensor(-1.0)), to.exp(to.tensor(-4.0))],
            [to.exp(to.tensor(-1.0)), 1.0, to.exp(to.tensor(-1.0))],
            [to.exp(to.tensor(-4.0)), to.exp(to.tensor(-1.0)), 1.0]
        ])
        to.testing.assert_close(result, expected)

    def test_kernel_with_y_argument(self):
        """Test kernel with y argument."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        y = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        # Median distance = 2, so bandwidth = 1 with median_coef = 0.5
        kernel, _ = gauss_kernel_median(x, y, median_coef=0.5)
        result = kernel(x, y)

        assert result.shape == (2, 3)
        expected = to.tensor([
            [1.0, to.exp(to.tensor(-1.0)), to.exp(to.tensor(-4.0))],
            [to.exp(to.tensor(-1.0)), 1.0, to.exp(to.tensor(-1.0))]
        ])
        to.testing.assert_close(result, expected)

    def test_kernel_equivalence_with_gauss_kernel(self):
        """Test that kernel matches gauss_kernel with same sigma."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_median(
            x, return_bandwidth=True
        )

        result = kernel(x)
        expected = gauss_kernel(x, sigma=bandwidth)
        to.testing.assert_close(result, expected)

    # Edge cases
    def test_single_point(self):
        """Test with single point."""
        x = to.tensor([[1.0, 2.0]])

        with pytest.warns(UserWarning):
            kernel, bandwidth = gauss_kernel_median(
                x, return_bandwidth=True
            )

        assert callable(kernel)
        assert bandwidth == 0.0  # sqrt(0) = 0

    def test_identical_points(self):
        """Test with identical points."""
        x = to.tensor([[1.0, 2.0], [1.0, 2.0]])

        # null mediane distance
        with pytest.warns(UserWarning):
            kernel, bandwidth = gauss_kernel_median(
                x, return_bandwidth=True
            )

        # All distances are 0, median = 0, sqrt(0) = 0
        assert bandwidth == 0.0

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=to.float32)

        # null mediane distance
        with pytest.warns(UserWarning):
            kernel, bandwidth = gauss_kernel_median(
                x, return_bandwidth=True
            )

        assert bandwidth.dtype == to.float32

        # c.f. Test_mediane_dist.test_median_with_two_points
        to.testing.assert_close(
            bandwidth, to.sqrt(to.tensor(12.5, dtype=to.float32))
        )

        # Verify kernel values
        result = kernel(x)

        assert result.dtype == to.float32

        # With sigma=sqrt(12.5), k(0,1) = exp(-25/(2*12.5)) = exp(-1)
        expected_off_diag = to.exp(to.tensor(-1))
        expected = to.tensor(
            [[1.0, expected_off_diag],
             [expected_off_diag, 1.0]],
            dtype=to.float32
        )
        to.testing.assert_close(result, expected)

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=to.float64)

        # null mediane distance
        with pytest.warns(UserWarning):
            kernel, bandwidth = gauss_kernel_median(
                x, return_bandwidth=True
            )

        assert bandwidth.dtype == to.float64

        # c.f. Test_mediane_dist.test_median_with_two_points
        to.testing.assert_close(
            bandwidth, to.sqrt(to.tensor(12.5, dtype=to.float64))
        )

        # Verify kernel values
        result = kernel(x)

        assert result.dtype == to.float64

        # With sigma=sqrt(12.5), k(0,1) = exp(-25/(2*12.5)) = exp(-1)
        expected_off_diag = to.exp(to.tensor(-1))
        expected = to.tensor(
            [[1.0, expected_off_diag],
             [expected_off_diag, 1.0]],
            dtype=to.float64
        )
        to.testing.assert_close(result, expected)


class Test_quantile_dist(object):
    """Test suite for _quantile_dist function."""

    # Output
    def test_returns_tensor(self):
        """Test that return type is torch.Tensor."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        result = _quantile_dist(x)

        assert isinstance(result, to.Tensor)

    # Basic functionality tests
    def test_median_default(self):
        """Test default quantile (q=0.5, median)."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = _quantile_dist(x)

        # Non-zero distances: 2, 8, 2, 2 -> sorted: 2, 2, 2, 8
        # Median = (2 + 2) / 2 = 2
        to.testing.assert_close(result, to.tensor(2.0))

    def test_median_with_two_points(self):
        """Test median with two points."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]])
        result = _quantile_dist(x)

        # Non-zero distance: 25 (only one)
        to.testing.assert_close(result, to.tensor(25.0))

    def test_median_with_y_provided(self):
        """Test median when y is provided."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        y = to.tensor([[2.0, 2.0], [3.0, 3.0]])
        result = _quantile_dist(x, y)

        # All points: (0,0), (1,1), (2,2), (3,3)
        # Non-zero distances: 2, 8, 18, 2, 2, 8, 8, 2, 2, 18, 8, 2
        # Sorted: 2,2,2,2,2,2, 8,8,8,8, 18,18
        # Median = (2 + 8) / 2 = 5
        to.testing.assert_close(result, to.tensor(5.0))

    # Quantile parameter tests
    def test_quantile_25(self):
        """Test with q=0.25 (first quartile)."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = _quantile_dist(x, q=0.25)

        # Non-zero distances: 2, 8, 2, 2 -> sorted: 2, 2, 2, 8
        # 25th percentile = 2
        to.testing.assert_close(result, to.tensor(2.0))

    def test_quantile_75(self):
        """Test with q=0.75 (third quartile)."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = _quantile_dist(x, q=0.75)

        # Non-zero distances: 2, 8, 2, 2 -> sorted: 2, 2, 2, 8
        # 75th percentile = 6.5 (interpolated between 2 and 8)
        to.testing.assert_close(result, to.tensor(6.5))

    def test_quantile_1(self):
        """Test with q=1.0 (maximum)."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = _quantile_dist(x, q=1.0)

        # Non-zero distances: 2, 8, 2, 2 -> max = 8
        to.testing.assert_close(result, to.tensor(8.0))

    def test_quantile_0(self):
        """Test with q=0.0 (minimum)."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = _quantile_dist(x, q=0.0)

        # Non-zero distances: 2, 8, 2, 2 -> min = 2
        to.testing.assert_close(result, to.tensor(2.0))

    # exclude_zero parameter tests
    def test_exclude_zero_true(self):
        """Test with exclude_zero=True (default)."""
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        result = _quantile_dist(x)

        # Non-zero distances: 4, 4, 4 -> median = 4
        to.testing.assert_close(result, to.tensor(4.0))

    def test_exclude_zero_false(self):
        """Test with exclude_zero=False."""

        # Median
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        result = _quantile_dist(x, exclude_zero=False)

        # All distances: 0, 0, 4, 0, 4, 0, 4, 4, 0
        # Sorted: 0, 0, 0, 0, 0, 4, 4, 4, 4
        # Median = 0
        to.testing.assert_close(result, to.tensor(0.0))

        # 25th percentile
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        result = _quantile_dist(x, q=0.25, exclude_zero=False)

        # All distances: 0, 0, 4, 0, 4, 0, 4, 4, 0
        # Sorted: 0, 0, 0, 0, 0, 4, 4, 4, 4
        # 25th percentile = 0
        to.testing.assert_close(result, to.tensor(0.0))

        # 75th percentile
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        result = _quantile_dist(x, q=0.75, exclude_zero=False)

        # All distances: 0, 0, 4, 0, 4, 0, 4, 4, 0
        # Sorted: 0, 0, 0, 0, 0, 4, 4, 4, 4
        # 75th percentile = 4
        to.testing.assert_close(result, to.tensor(4.0))

    # All-zero distance tests
    def test_all_zero_distances_warns(self):
        """Test that all-zero distances trigger warning."""
        x = to.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

        with pytest.warns(
            UserWarning, match="All distances between observations are zero"
        ):
            result = _quantile_dist(x)

        to.testing.assert_close(result, to.tensor(0.0))

    def test_all_zero_distances_exclude_zero_false(self):
        """Test all-zero distances with exclude_zero=False."""
        x = to.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

        result = _quantile_dist(x, exclude_zero=False)

        # All distances are 0, no warning needed since zeros are included
        to.testing.assert_close(result, to.tensor(0.0))

    def test_single_point_warns(self):
        """Test single point triggers warning."""
        x = to.tensor([[1.0, 2.0]])

        with pytest.warns(
            UserWarning, match="All distances between observations are zero"
        ):
            result = _quantile_dist(x)

        to.testing.assert_close(result, to.tensor(0.0))

    # Edge cases
    def test_single_point_exclude_zero_false(self):
        """Test single point with exclude_zero=False."""
        x = to.tensor([[1.0, 2.0]])
        result = _quantile_dist(x, exclude_zero=False)

        # Only distance is 0
        to.testing.assert_close(result, to.tensor(0.0))

    def test_identical_points_warns(self):
        """Test identical points trigger warning."""
        x = to.tensor([[1.0, 2.0], [1.0, 2.0]])

        with pytest.warns(
            UserWarning, match="All distances between observations are zero"
        ):
            result = _quantile_dist(x)

        to.testing.assert_close(result, to.tensor(0.0))

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float32)
        result = _quantile_dist(x)

        assert result.dtype == to.float32

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=to.float64)
        result = _quantile_dist(x)

        assert result.dtype == to.float64

    # Known value tests
    def test_known_values_regular_grid(self):
        """Test with regular grid points."""
        x = to.tensor([[0.0], [1.0], [2.0], [3.0]])
        result = _quantile_dist(x)

        # Non-zero distances: 1,4,9, 1,1,4, 4,1,1, 9,4,1
        # Sorted: 1,1,1,1,1,1, 4,4,4,4, 9,9
        # Median = (1 + 4) / 2 = 2.5
        to.testing.assert_close(result, to.tensor(2.5))

    def test_known_values_irregular(self):
        """Test with irregular spacing."""
        x = to.tensor([[0.0], [2.0], [5.0]])
        result = _quantile_dist(x)

        # Non-zero distances: 4, 25, 9, 9, 25, 4
        # Sorted: 4, 4, 9, 9, 25, 25
        # Median = (9 + 9) / 2 = 9
        to.testing.assert_close(result, to.tensor(9.0))


class Test_gauss_kernel_quantile(object):
    """Test suite for gauss_kernel_quantile function."""

    # Output structure tests
    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = gauss_kernel_quantile(x)

        assert isinstance(result, tuple)
        assert len(result) == 2

        result = gauss_kernel_quantile(x, return_bandwidth=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_kernel_and_none(self):
        """Test that function returns kernel and None by default."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_quantile(x)

        assert callable(kernel)
        assert bandwidth is None

    def test_returns_kernel_and_bandwidth(self):
        """Test that function returns kernel and bandwidth when requested."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_quantile(x, return_bandwidth=True)

        assert callable(kernel)
        assert bandwidth is not None
        assert isinstance(bandwidth, to.Tensor)

    # Bandwidth calculation tests
    def test_quantile_bandwidth_default(self):
        """Test default quantile bandwidth (q=0.5, median)."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, bandwidth = gauss_kernel_quantile(x, return_bandwidth=True)

        # Non-zero distances: 2, 8, 2, 2 -> median = 2, sqrt(2) ≈ 1.414
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(2.0)))

    def test_quantile_bandwidth_with_scaling(self):
        """Test quantile bandwidth with scaling coefficient."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, bandwidth = gauss_kernel_quantile(
            x, scaling_coef=4, return_bandwidth=True
        )

        # Median distance = 2, sqrt(2 * 4) = sqrt(8) ≈ 2.828
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(8.0)))

    def test_quantile_bandwidth_q25(self):
        """Test bandwidth with q=0.25."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, bandwidth = gauss_kernel_quantile(
            x, q=0.25, return_bandwidth=True
        )

        # Non-zero distances: 2, 8, 2, 2 -> 25th percentile = 2
        # sqrt(2) ≈ 1.414
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(2.0)))

    def test_quantile_bandwidth_q75(self):
        """Test bandwidth with q=0.75."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, bandwidth = gauss_kernel_quantile(
            x, q=0.75, return_bandwidth=True
        )

        # Non-zero distances: 2, 8, 2, 2 -> 75th percentile = 6.5
        # sqrt(6.5) ≈ 2.55
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(6.5)))

    def test_float_bandwidth(self):
        """Test with float bandwidth value."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_quantile(
            x, bandwidth=2.0, return_bandwidth=True
        )

        assert bandwidth == 2.0
        assert callable(kernel)

    # Exclude zero parameter tests
    def test_exclude_zero_true(self):
        """Test with exclude_zero=True (default)."""
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        _, bandwidth = gauss_kernel_quantile(x, return_bandwidth=True)

        # Non-zero distances: 4, 4, 4 -> median = 4, sqrt(4) = 2
        to.testing.assert_close(bandwidth, to.tensor(2.0))

    def test_exclude_zero_false(self):
        """Test with exclude_zero=False."""
        # Median
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        _, bandwidth = gauss_kernel_quantile(
            x, exclude_zero=False, return_bandwidth=True
        )

        # All distances: 0, 0, 4, 0, 4, 0, 4, 4, 0
        # Sorted: 0, 0, 0, 0, 0, 4, 4, 4, 4
        # Median = 0
        to.testing.assert_close(bandwidth, to.tensor(0.0))

        # 25th percentile
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        _, bandwidth = gauss_kernel_quantile(
            x, q=0.25, exclude_zero=False, return_bandwidth=True
        )

        # All distances: 0, 0, 4, 0, 4, 0, 4, 4, 0
        # Sorted: 0, 0, 0, 0, 0, 4, 4, 4, 4
        # 25th percentile = 0
        to.testing.assert_close(bandwidth, to.tensor(0.0))

        # 75th percentile
        x = to.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        _, bandwidth = gauss_kernel_quantile(
            x, q=0.75, exclude_zero=False, return_bandwidth=True
        )

        # All distances: 0, 0, 4, 0, 4, 0, 4, 4, 0
        # Sorted: 0, 0, 0, 0, 0, 4, 4, 4, 4
        # 75th percentile = 4
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(4.0)))

    # Kernel behavior tests
    def test_kernel_with_quantile_bandwidth(self):
        """Test kernel output with quantile bandwidth."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        # Median distance = 2, bandwidth = sqrt(2) ≈ 1.414
        kernel, _ = gauss_kernel_quantile(x)
        result = kernel(x)

        assert result.shape == (3, 3)

        # With sigma=sqrt(2), k(0,1) = exp(-2/(2*2)) = exp(-0.5)
        expected_off_diag = to.exp(to.tensor(-0.5))
        expected = to.tensor([
            [1.0, expected_off_diag, to.exp(to.tensor(-2.0))],
            [expected_off_diag, 1.0, expected_off_diag],
            [to.exp(to.tensor(-2.0)), expected_off_diag, 1.0]
        ])
        to.testing.assert_close(result, expected)

    def test_kernel_with_float_bandwidth(self):
        """Test kernel output with float bandwidth."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, _ = gauss_kernel_quantile(x, bandwidth=1.0)
        result = kernel(x)

        assert result.shape == (3, 3)

        expected = to.tensor([
            [1.0, to.exp(to.tensor(-1.0)), to.exp(to.tensor(-4.0))],
            [to.exp(to.tensor(-1.0)), 1.0, to.exp(to.tensor(-1.0))],
            [to.exp(to.tensor(-4.0)), to.exp(to.tensor(-1.0)), 1.0]
        ])
        to.testing.assert_close(result, expected)

    def test_kernel_with_y_argument(self):
        """Test kernel with y argument."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0]])
        y = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        kernel, _ = gauss_kernel_quantile(x, y)
        result = kernel(x, y)

        assert result.shape == (2, 3)

    def test_kernel_equivalence_with_gauss_kernel(self):
        """Test that kernel matches gauss_kernel with same sigma."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_quantile(
            x, return_bandwidth=True
        )

        result = kernel(x)
        expected = gauss_kernel(x, sigma=bandwidth)
        to.testing.assert_close(result, expected)

    # Edge cases
    def test_single_point(self):
        """Test with single point."""
        x = to.tensor([[1.0, 2.0]])

        with pytest.warns(
            UserWarning, match="All distances between observations are zero"
        ):
            kernel, bandwidth = gauss_kernel_quantile(
                x, return_bandwidth=True
            )

        assert callable(kernel)
        assert bandwidth == 0.0  # sqrt(0) = 0

    def test_identical_points(self):
        """Test with identical points."""
        x = to.tensor([[1.0, 2.0], [1.0, 2.0]])

        with pytest.warns(
            UserWarning, match="All distances between observations are zero"
        ):
            kernel, bandwidth = gauss_kernel_quantile(
                x, return_bandwidth=True
            )

        # All distances are 0, quantile = 0, sqrt(0) = 0
        assert bandwidth == 0.0

    def test_identical_points_exclude_zero_false(self):
        """Test identical points with exclude_zero=False."""
        x = to.tensor([[1.0, 2.0], [1.0, 2.0]])
        kernel, bandwidth = gauss_kernel_quantile(
            x, exclude_zero=False, return_bandwidth=True
        )

        # All distances are 0, quantile = 0, sqrt(0) = 0
        assert bandwidth == 0.0

    # Dtype tests
    def test_float32_dtype(self):
        """Test with float32 dtype."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=to.float32)
        kernel, bandwidth = gauss_kernel_quantile(
            x, return_bandwidth=True
        )

        assert bandwidth.dtype == to.float32

        # Distance = 25, quantile = 25, sqrt(25) = 5
        to.testing.assert_close(bandwidth, to.tensor(5.0, dtype=to.float32))

        # Verify kernel values
        result = kernel(x)
        assert result.dtype == to.float32

        # With sigma=5, k(0,1) = exp(-25/(2*25)) = exp(-0.5)
        expected_off_diag = to.exp(to.tensor(-0.5))
        expected = to.tensor(
            [[1.0, expected_off_diag],
             [expected_off_diag, 1.0]],
            dtype=to.float32
        )
        to.testing.assert_close(result, expected)

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=to.float64)
        kernel, bandwidth = gauss_kernel_quantile(
            x, return_bandwidth=True
        )

        assert bandwidth.dtype == to.float64

        # Distance = 25, quantile = 25, sqrt(25) = 5
        to.testing.assert_close(bandwidth, to.tensor(5.0, dtype=to.float64))

        # Verify kernel values
        result = kernel(x)
        assert result.dtype == to.float64

        # With sigma=5, k(0,1) = exp(-25/(2*25)) = exp(-0.5)
        expected_off_diag = to.exp(to.tensor(-0.5))
        expected = to.tensor(
            [[1.0, expected_off_diag],
             [expected_off_diag, 1.0]],
            dtype=to.float64
        )
        to.testing.assert_close(result, expected)

    # Known value tests
    def test_known_values(self):
        """Test with known values."""
        x = to.tensor([[0.0, 0.0], [3.0, 4.0]])
        kernel, bandwidth = gauss_kernel_quantile(
            x, return_bandwidth=True
        )

        # Distance = 25, quantile = 25, sqrt(25) = 5
        assert bandwidth == 5.0

        # With sigma=5, k(0,1) = exp(-25/(2*25)) = exp(-0.5)
        result = kernel(x)
        expected_off_diag = to.exp(to.tensor(-0.5))
        expected = to.tensor([[1.0, expected_off_diag],
                              [expected_off_diag, 1.0]])
        to.testing.assert_close(result, expected)

    def test_known_values_q75(self):
        """Test with known values for q=0.75."""
        x = to.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kernel, bandwidth = gauss_kernel_quantile(
            x, q=0.75, return_bandwidth=True
        )

        # Non-zero distances: 2, 8, 2, 2 -> 75th percentile = 6.5
        # sqrt(6.5) ≈ 2.55
        to.testing.assert_close(bandwidth, to.sqrt(to.tensor(6.5)))
