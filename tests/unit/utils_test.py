import numpy as np
import pandas as pd
import pytest
import torch as to

from kaov.utils import convert_to_torch


class Test_convert_to_torch(object):
    """Test suite for convert_to_torch function."""

    # Test with NumPy arrays
    def test_numpy_1d_array(self):
        """Test conversion of 1D numpy array."""
        A = np.array([1.0, 2.0, 3.0])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (3,)
        np.testing.assert_allclose(B.numpy(), A)

    def test_numpy_2d_array(self):
        """Test conversion of 2D numpy array."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (2, 2)
        np.testing.assert_allclose(B.numpy(), A)

    def test_numpy_integer_array(self):
        """Test conversion of integer numpy array."""
        A = np.array([1, 2, 3])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64  # Should be converted to double
        np.testing.assert_allclose(B.numpy(), A.astype(np.float64))

    def test_numpy_float32_array(self):
        """Test conversion of float32 numpy array."""
        A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64  # Should be converted to double
        np.testing.assert_allclose(B.numpy(), A.astype(np.float64))

    def test_numpy_readonly_array(self):
        """Test conversion of read-only numpy array."""
        A = np.array([1.0, 2.0, 3.0])
        A.setflags(write=False)  # Make read-only
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (3,)
        np.testing.assert_allclose(B.numpy(), A)

    # Test with Pandas Series
    def test_pandas_series(self):
        """Test conversion of pandas Series."""
        A = pd.Series([1.0, 2.0, 3.0])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (3, 1)  # Should be reshaped to column vector
        np.testing.assert_allclose(B.numpy().flatten(), A.values)

    def test_pandas_series_with_index(self):
        """Test conversion of pandas Series with custom index."""
        A = pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c'])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (3, 1)
        np.testing.assert_allclose(B.numpy().flatten(), A.values)

    def test_pandas_series_single_element(self):
        """Test conversion of single-element pandas Series."""
        A = pd.Series([5.0])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (1, 1)
        assert B[0, 0] == 5.0

    def test_pandas_series_integer(self):
        """Test conversion of integer pandas Series."""
        A = pd.Series([1, 2, 3])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (3, 1)
        np.testing.assert_allclose(B.numpy().flatten(), A.values.astype(np.float64))

    # Test with Pandas DataFrame
    def test_pandas_dataframe(self):
        """Test conversion of pandas DataFrame."""
        A = pd.DataFrame({'col1': [1.0, 2.0], 'col2': [3.0, 4.0]})
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (2, 2)
        np.testing.assert_allclose(B.numpy(), A.values)

    def test_pandas_dataframe_with_index(self):
        """Test conversion of pandas DataFrame with custom index."""
        A = pd.DataFrame(
            {'col1': [1.0, 2.0], 'col2': [3.0, 4.0]},
            index=['row1', 'row2']
        )
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (2, 2)
        np.testing.assert_allclose(B.numpy(), A.values)

    def test_pandas_dataframe_single_column(self):
        """Test conversion of single-column pandas DataFrame."""
        A = pd.DataFrame({'col1': [1.0, 2.0, 3.0]})
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (3, 1)
        np.testing.assert_allclose(B.numpy(), A.values)

    def test_pandas_dataframe_integer(self):
        """Test conversion of integer pandas DataFrame."""
        A = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (2, 2)
        np.testing.assert_allclose(B.numpy(), A.values.astype(np.float64))

    def test_pandas_dataframe_mixed_types(self):
        """Test conversion of pandas DataFrame with mixed types."""
        A = pd.DataFrame({'num': [1.0, 2.0], 'str': ['a', 'b']})
        with pytest.raises(
            ValueError,
            match="could not convert string to float"
        ):
            convert_to_torch(A)

    # Test with torch.Tensor
    def test_torch_tensor(self):
        """Test conversion of torch.Tensor."""
        A = to.tensor([1.0, 2.0, 3.0])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert to.equal(B, A.double())

    def test_torch_tensor_float32(self):
        """Test conversion of float32 torch.Tensor."""
        A = to.tensor([1.0, 2.0, 3.0], dtype=to.float32)
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64  # Should be converted to double
        assert to.equal(B, A.double())

    def test_torch_tensor_2d(self):
        """Test conversion of 2D torch.Tensor."""
        A = to.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (2, 2)
        assert to.equal(B, A.double())

    def test_torch_tensor_integer(self):
        """Test conversion of integer torch.Tensor."""
        A = to.tensor([1, 2, 3])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64  # Should be converted to double
        assert to.equal(B, A.double())

    # Test with unsupported types
    def test_unsupported_type_string(self):
        """Test with unsupported type (string)."""
        A = "not an array"
        with pytest.raises(
            TypeError,
            match="Unsupported input type"
        ):
            convert_to_torch(A)

    def test_unsupported_type_dict(self):
        """Test with unsupported type (dict)."""
        A = {'key': 'value'}
        with pytest.raises(
            TypeError,
            match="Unsupported input type"
        ):
            convert_to_torch(A)

    def test_unsupported_type_none(self):
        """Test with None."""
        A = None
        with pytest.raises(
            TypeError,
            match="Unsupported input type"
        ):
            convert_to_torch(A)

    def test_unsupported_type_list(self):
        """Test with Python list (not supported by this implementation)."""
        A = [1.0, 2.0, 3.0]
        with pytest.raises(
            TypeError,
            match="Unsupported input type"
        ):
            convert_to_torch(A)

    def test_unsupported_type_tuple(self):
        """Test with Python tuple (not supported by this implementation)."""
        A = (1.0, 2.0, 3.0)
        with pytest.raises(
            TypeError,
            match="Unsupported input type"
        ):
            convert_to_torch(A)

    def test_unsupported_type_scalar(self):
        """Test with scalar value (not supported by this implementation)."""
        A = 5.0
        with pytest.raises(
            TypeError,
            match="Unsupported input type"
        ):
            convert_to_torch(A)

    # Test edge cases
    def test_empty_numpy_array(self):
        """Test conversion of empty numpy array."""
        A = np.array([])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (0,)

    def test_empty_pandas_series(self):
        """Test conversion of empty pandas Series."""
        A = pd.Series([], dtype=float)
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (0, 1)

    def test_empty_pandas_dataframe(self):
        """Test conversion of empty pandas DataFrame."""
        A = pd.DataFrame()
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (0, 0)

    def test_single_value_numpy(self):
        """Test conversion of single-value numpy array."""
        A = np.array([5.0])
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.shape == (1,)
        assert B[0] == 5.0

    def test_numpy_3d_array(self):
        """Test conversion of 3D numpy array."""
        A = np.random.rand(2, 3, 4)
        B = convert_to_torch(A)

        assert isinstance(B, to.Tensor)
        assert B.dtype == to.float64
        assert B.shape == (2, 3, 4)
        np.testing.assert_allclose(B.numpy(), A)

    # Test dtype consistency
    def test_dtype_is_always_float64(self):
        """Test that output is always float64."""
        test_inputs = [
            np.array([1, 2, 3]),  # int array
            np.array([1.0, 2.0], dtype=np.float32),  # float32
            pd.Series([1, 2, 3]),  # int series
            pd.DataFrame({'a': [1, 2]}),  # int dataframe
            to.tensor([1, 2, 3]),  # int tensor
            to.tensor([1.0, 2.0], dtype=to.float32),  # float32 tensor
        ]

        for A in test_inputs:
            B = convert_to_torch(A)
            assert B.dtype == to.float64, f"Failed for input type {A}"

    # Test shape preservation
    def test_shape_preservation(self):
        """Test that shapes are preserved (except Series which becomes column)."""
        test_cases = [
            (np.array([[1.0, 2.0], [3.0, 4.0]]), (2, 2)),
            (pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}), (2, 2)),
            (to.tensor([[1.0, 2.0], [3.0, 4.0]]), (2, 2)),
        ]

        for A, expected_shape in test_cases:
            B = convert_to_torch(A)
            assert B.shape == expected_shape, f"Failed for input {A}"

    # Test value preservation
    def test_value_preservation(self):
        """Test that values are preserved after conversion."""
        test_inputs = [
            np.array([1.0, 2.0, 3.0]),
            pd.Series([1.0, 2.0, 3.0]),
            pd.DataFrame({'a': [1.0, 2.0, 3.0]}),
            to.tensor([1.0, 2.0, 3.0]),
        ]

        expected_values = np.array([1.0, 2.0, 3.0])

        for A in test_inputs:
            B = convert_to_torch(A)
            np.testing.assert_allclose(
                B.numpy().flatten(), expected_values,
                err_msg=f"Failed for input {A}"
            )
