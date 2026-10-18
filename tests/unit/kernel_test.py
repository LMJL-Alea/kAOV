import torch as to

from kaov.kernel import linear_kernel


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

