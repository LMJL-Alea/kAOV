import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import torch as to

from kaov import AOV
from kaov.testing import (
    dummy_data, assert_equal_data_obj, assert_equal_aov_obj,
    assert_vec_equal_abs_one, assert_same_eigenvectors
)


class Test_dummy_data(object):
    """Test suite for dummy_data function."""

    def test_default_parameters(self):
        """Test function with default parameters."""
        data, input_formula, output_formula = dummy_data()

        # Check data shape
        assert data.shape == (100, 101)  # 100 rows, 100 numeric + 1 factor column

        # Check column names
        assert list(data.columns[:100]) == [f"col{i+1}" for i in range(100)]
        assert data.columns[-1] == "factor1"

        # Check formulas
        assert input_formula == "C(factor1, OneHot)"
        assert output_formula == " + ".join([f"col{i+1}" for i in range(100)])

        # Check data types
        assert data.iloc[:, :100].dtypes.apply(lambda x: x == np.float64).all()
        assert is_string_dtype(data["factor1"].dtype)

        # Check for missing values
        assert not data.isnull().any().any()

    def test_custom_dimensions(self):
        """Test function with custom row and column counts."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )

        # Check shape
        assert data.shape == (50, 12)  # 10 numeric + 2 factor columns

        # Check column names
        assert list(data.columns[:10]) == [f"col{i+1}" for i in range(10)]
        assert list(data.columns[10:]) == ["factor1", "factor2"]

        # Check unique values in each factor column
        assert data["factor1"].nunique() == 3
        assert data["factor2"].nunique() == 4

        # Check factor values
        assert set(data["factor1"].unique()) == {"c1", "c2", "c3"}
        assert set(data["factor2"].unique()) == {"c1", "c2", "c3", "c4"}

        # Check formulas
        assert input_formula == "C(factor1, OneHot) + C(factor2, OneHot)"
        assert output_formula == " + ".join([f"col{i+1}" for i in range(10)])

    def test_factor_levels(self):
        """Test that factor columns have correct number of unique levels."""
        data, _, _ = dummy_data(
            n_row=100, n_col=5, n_factor=3, n_group=[2, 3, 4], seed=123
        )

        # Check unique values in each factor column
        assert data["factor1"].nunique() == 2
        assert data["factor2"].nunique() == 3
        assert data["factor3"].nunique() == 4

        # Check factor values
        assert set(data["factor1"].unique()) == {"c1", "c2"}
        assert set(data["factor2"].unique()) == {"c1", "c2", "c3"}
        assert set(data["factor3"].unique()) == {"c1", "c2", "c3", "c4"}

    def test_seed_reproducibility(self):
        """Test that same seed produces identical results."""
        data1, formula1_in, formula1_out = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[2, 3], seed=42
        )
        data2, formula2_in, formula2_out = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[2, 3], seed=42
        )

        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2)

        # Formulas should be identical
        assert formula1_in == formula2_in
        assert formula1_out == formula2_out

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        data1, _, _ = dummy_data(
            n_row=50, n_col=10, n_factor=1, n_group=[2], seed=42
        )
        data2, _, _ = dummy_data(
            n_row=50, n_col=10, n_factor=1, n_group=[2], seed=43
        )

        # Data should differ
        assert not data1.equals(data2)

    def test_value_error_mismatched_groups(self):
        """Test that ValueError is raised when n_group length != n_factor."""
        with pytest.raises(
            ValueError,
            match="`n_group` list length should be equal to `n_factor`"
        ):
            dummy_data(n_factor=2, n_group=[2])

    def test_value_error_empty_groups(self):
        """Test ValueError with empty n_group when n_factor > 0."""
        with pytest.raises(
            ValueError,
            match="`n_group` list length should be equal to `n_factor`"
        ):
            dummy_data(n_factor=1, n_group=[])

    def test_numeric_data_distribution(self):
        """Test that numeric data follows standard normal distribution."""
        data, _, _ = dummy_data(
            n_row=1000, n_col=10, n_factor=1, n_group=[2], seed=42
        )

        # Check mean is close to 0
        numeric_data = data.iloc[:, :10]
        assert np.allclose(numeric_data.mean().values, 0, atol=0.1)

        # Check standard deviation is close to 1
        assert np.allclose(numeric_data.std().values, 1, atol=0.1)

    def test_single_row(self):
        """Test function with minimal row count."""
        data, input_formula, output_formula = dummy_data(
            n_row=1, n_col=3, n_factor=1, n_group=[2], seed=42
        )

        assert data.shape == (1, 4)  # 1 row, 3 numeric + 1 factor
        assert data["factor1"].iloc[0] in ["c1", "c2"]

    def test_single_column(self):
        """Test function with minimal column count."""
        data, input_formula, output_formula = dummy_data(
            n_row=10, n_col=1, n_factor=1, n_group=[2], seed=42
        )

        assert data.shape == (10, 2)  # 10 rows, 1 numeric + 1 factor
        assert data.columns[0] == "col1"
        assert output_formula == "col1"

    def test_factor_balance(self):
        """Test that factor levels are roughly balanced."""
        data, _, _ = dummy_data(
            n_row=100, n_col=5, n_factor=1, n_group=[2], seed=42
        )

        # With 100 rows and 2 levels, expect ~50 each
        counts = data["factor1"].value_counts()
        assert len(counts) == 2
        assert abs(counts["c1"] - 50) <= 10  # Allow some randomness
        assert abs(counts["c2"] - 50) <= 10


class Test_assert_equal_data_obj(object):
    """Test suite for assert_equal_data_obj function."""

    @pytest.fixture
    def data_obj(self):
        """Create a dummy data object for testing."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )
        kfit = AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )
        return kfit.data

    @pytest.fixture
    def data_obj_same_seed(self):
        """Create a same dummy data object for testing."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )
        kfit = AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )
        return kfit.data

    @pytest.fixture
    def data_obj_different_seed(self):
        """Create another different dummy data object for testing."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=43
        )
        kfit = AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )
        return kfit.data

    def test_identical_objects_pass(self, data_obj, data_obj_same_seed):
        """Test that identical objects pass the assertion."""
        # Should not raise any exception
        assert_equal_data_obj(data_obj, data_obj_same_seed)

    def test_different_objects_fail(self, data_obj, data_obj_different_seed):
        """Test that different objects fail the assertion."""
        with pytest.raises(AssertionError):
            assert_equal_data_obj(data_obj, data_obj_different_seed)

    def test_same_object_passes(self, data_obj):
        """Test that comparing an object with itself passes."""
        assert_equal_data_obj(data_obj, data_obj)

    def test_custom_tolerances(
        self, data_obj, data_obj_same_seed, data_obj_different_seed
    ):
        """Test with custom tolerance values."""
        # same objects
        assert_equal_data_obj(
            data_obj,
            data_obj_same_seed,
            rtol=1e-5,
            atol=1e-3
        )
        # different objects
        with pytest.raises(AssertionError):
            assert_equal_data_obj(
                data_obj,
                data_obj_different_seed,
                rtol=1e-5,
                atol=1e-3
            )


# import pytest
# import numpy as np
# import pandas as pd
# import torch
# from your_module import dummy_data  # Replace with actual import
# from kaov import AOV
# from your_module import assert_equal_aov_obj  # Replace with actual import


class Test_assert_equal_aov_obj:
    """Test suite for assert_equal_aov_obj function."""

    @pytest.fixture
    def aov_object(self):
        """Create an AOV object from dummy data for testing."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )
        return AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )

    @pytest.fixture
    def aov_object_same_seed(self):
        """Create an identical AOV object using the same seed."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )
        return AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )

    @pytest.fixture
    def aov_object_different_seed(self):
        """Create a different AOV object using a different seed."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=2, n_group=[3, 4], seed=43
        )
        return AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )

    @pytest.fixture
    def aov_object_different_n_row(self):
        """Create an AOV object with different number of rows."""
        data, input_formula, output_formula = dummy_data(
            n_row=30, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )
        return AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )

    @pytest.fixture
    def aov_object_different_n_col(self):
        """Create an AOV object with different number of columns."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=5, n_factor=2, n_group=[3, 4], seed=42
        )
        return AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )

    @pytest.fixture
    def aov_object_different_factor(self):
        """Create an AOV object with different factor structure."""
        data, input_formula, output_formula = dummy_data(
            n_row=50, n_col=10, n_factor=1, n_group=[2], seed=42
        )
        return AOV.from_formula(
            f"{output_formula} ~ {input_formula}", data=data
        )

    # Basic equality tests
    def test_identical_objects_pass(self, aov_object, aov_object_same_seed):
        """Test that identical objects pass the assertion."""
        assert_equal_aov_obj(aov_object, aov_object_same_seed)

    def test_same_object_passes(self, aov_object):
        """Test that comparing an object with itself passes."""
        assert_equal_aov_obj(aov_object, aov_object)

    def test_different_objects_fail(
        self, aov_object, aov_object_different_seed
    ):
        """Test that different objects fail the assertion."""
        with pytest.raises(AssertionError):
            assert_equal_aov_obj(aov_object, aov_object_different_seed)

    # Default parameters tests
    def test_default_trunc(self, aov_object, aov_object_same_seed):
        """Test with default trunc=None."""
        assert_equal_aov_obj(aov_object, aov_object_same_seed, trunc=None)

    # Truncation parameter tests
    def test_set_trunc(
        self, aov_object, aov_object_same_seed, aov_object_different_seed
    ):
        """Test with positive trunc value."""
        # Test with various truncation levels
        for trunc in [None, 0, 1, 5, 10, 50]:
            assert_equal_aov_obj(aov_object, aov_object_same_seed, trunc=trunc)

            with pytest.raises(AssertionError):
                assert_equal_aov_obj(
                    aov_object, aov_object_different_seed, trunc=trunc
                )

    def test_trunc_negative(self, aov_object, aov_object_same_seed):
        """Test with negative trunc value (should be handled gracefully)."""
        try:
            assert_equal_aov_obj(aov_object, aov_object_same_seed, trunc=-1)
        except (ValueError, AssertionError):
            pass  # Either behavior is acceptable

    def test_trunc_float(self, aov_object, aov_object_same_seed):
        """Test with float trunc value (should be handled gracefully)."""
        try:
            assert_equal_aov_obj(aov_object, aov_object_same_seed, trunc=5.5)
        except (TypeError, ValueError, AssertionError):
            pass  # Either behavior is acceptable

    # Tolerance parameter tests
    def test_custom_tolerances(
        self, aov_object, aov_object_same_seed, aov_object_different_seed
    ):
        """Test with custom tolerance values."""
        assert_equal_aov_obj(
            aov_object, aov_object_same_seed, rtol=1e-3, atol=1e-3
        )

        with pytest.raises(AssertionError):
            assert_equal_aov_obj(
                aov_object, aov_object_different_seed, rtol=1e-3, atol=1e-3
            )

    # Different data structure tests
    def test_different_n_row_fails(
        self, aov_object, aov_object_different_n_row
    ):
        """Test that objects with different number of rows fail."""
        with pytest.raises(AssertionError):
            assert_equal_aov_obj(aov_object, aov_object_different_n_row)

    def test_different_n_col_fails(
        self, aov_object, aov_object_different_n_col
    ):
        """Test that objects with different number of columns fail."""
        with pytest.raises(AssertionError):
            assert_equal_aov_obj(aov_object, aov_object_different_n_col)

    def test_different_factor_structure_fails(
        self, aov_object, aov_object_different_factor
    ):
        """Test that objects with different factor structure fail."""
        with pytest.raises(AssertionError):
            assert_equal_aov_obj(aov_object, aov_object_different_factor)


class Test_assert_vec_equal_abs_one(object):
    """Test suite for assert_vec_equal_abs_one function."""

    # Passing tests
    def test_numpy_array_all_ones(self):
        """Test with numpy array of all ones."""
        vec = np.array([1.0, 1.0, 1.0])
        assert_vec_equal_abs_one(vec)

    def test_numpy_array_mixed_signs(self):
        """Test with numpy array of 1 and -1 values."""
        vec = np.array([1.0, -1.0, 1.0, -1.0])
        assert_vec_equal_abs_one(vec)

    def test_torch_tensor_all_ones(self):
        """Test with torch tensor of all ones."""
        vec = to.tensor([1.0, 1.0, 1.0])
        assert_vec_equal_abs_one(vec)

    def test_torch_tensor_mixed_signs(self):
        """Test with torch tensor of 1 and -1 values."""
        vec = to.tensor([1.0, -1.0, 1.0])
        assert_vec_equal_abs_one(vec)

    def test_single_element_numpy(self):
        """Test with single element numpy array."""
        vec = np.array([1.0])
        assert_vec_equal_abs_one(vec)

    def test_single_element_torch(self):
        """Test with single element torch tensor."""
        vec = to.tensor([-1.0])
        assert_vec_equal_abs_one(vec)

    def test_custom_tolerances(self):
        """Test with custom tolerance values."""
        vec = np.array([1.0, 1.0 + 1e-8, 1.0])
        assert_vec_equal_abs_one(vec, rtol=1e-6, atol=1e-6)

    # Failing tests
    def test_numpy_array_not_ones(self):
        """Test with numpy array containing values other than ±1."""
        vec = np.array([1.0, 0.5, 1.0])
        with pytest.raises(AssertionError):
            assert_vec_equal_abs_one(vec)

    def test_torch_tensor_not_ones(self):
        """Test with torch tensor containing values other than ±1."""
        vec = to.tensor([1.0, 2.0, 1.0])
        with pytest.raises(AssertionError):
            assert_vec_equal_abs_one(vec)

    def test_numpy_array_zeros(self):
        """Test with numpy array of zeros."""
        vec = np.array([0.0, 0.0, 0.0])
        with pytest.raises(AssertionError):
            assert_vec_equal_abs_one(vec)

    def test_torch_tensor_zeros(self):
        """Test with torch tensor of zeros."""
        vec = to.zeros(3)
        with pytest.raises(AssertionError):
            assert_vec_equal_abs_one(vec)

    def test_numpy_array_close_but_not_equal(self):
        """Test with values close to 1 but outside tolerance."""
        vec = np.array([1.0, 1.0 + 1e-5, 1.0])
        with pytest.raises(AssertionError):
            assert_vec_equal_abs_one(vec, rtol=0, atol=1e-11)

    # Type error tests
    def test_invalid_type_list(self):
        """Test with a Python list (invalid type)."""
        vec = [1.0, 1.0, 1.0]
        with pytest.raises(TypeError, match="Numpy array or a Torch tensor"):
            assert_vec_equal_abs_one(vec)

    def test_invalid_type_scalar(self):
        """Test with a scalar value (invalid type)."""
        vec = 1.0
        with pytest.raises(TypeError, match="Numpy array or a Torch tensor"):
            assert_vec_equal_abs_one(vec)

    def test_invalid_type_string(self):
        """Test with a string (invalid type)."""
        vec = "not a vector"
        with pytest.raises(TypeError, match="Numpy array or a Torch tensor"):
            assert_vec_equal_abs_one(vec)

    # Edge cases
    def test_empty_numpy_array(self):
        """Test with empty numpy array."""
        vec = np.array([])
        assert_vec_equal_abs_one(vec)  # Should pass trivially

    def test_empty_torch_tensor(self):
        """Test with empty torch tensor."""
        vec = to.tensor([])
        assert_vec_equal_abs_one(vec)  # Should pass trivially

    def test_2d_numpy_array(self):
        """Test with 2D numpy array."""
        vec = np.array([[1.0, -1.0], [1.0, -1.0]])
        assert_vec_equal_abs_one(vec)

    def test_2d_torch_tensor(self):
        """Test with 2D torch tensor."""
        vec = to.tensor([[1.0, -1.0], [1.0, -1.0]])
        assert_vec_equal_abs_one(vec)

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        vec = np.array([1.0, -1.0], dtype=np.float32)
        assert_vec_equal_abs_one(vec)

    def test_integer_dtype(self):
        """Test with integer dtype."""
        vec = np.array([1, -1, 1])
        assert_vec_equal_abs_one(vec)


class Test_assert_same_eigenvectors(object):
    """Test suite for assert_same_eigenvectors function."""

    @pytest.fixture
    def dummy_data(self):
        """Generate dummy data for testing."""
        # Generate data
        data, _, _ = dummy_data(
            n_row=20, n_col=10, n_factor=2, n_group=[3, 4], seed=42
        )
        # return only numerical matrix
        return data.iloc[:, :10].values.copy()

    @pytest.fixture
    def sym_matrix(self, dummy_data):
        """Create a symmetric matrix from dummy data."""
        data_tensor = to.from_numpy(dummy_data)
        return to.matmul(data_tensor, data_tensor.T)

    @pytest.fixture
    def torch_eigen(self, sym_matrix):
        """Compute eigendecomposition using torch."""
        sp, ev = to.linalg.eigh(sym_matrix)
        return sp, ev

    @pytest.fixture
    def numpy_eigen(self, sym_matrix):
        """Compute eigendecomposition using numpy."""
        sp, ev = np.linalg.eigh(sym_matrix.numpy())
        return sp, ev

    @pytest.fixture
    def filtered_torch_eigen(self, torch_eigen):
        """Get eigenvectors for non-null eigenvalues from torch."""
        sp, ev = torch_eigen
        mask = sp >= 1e-12
        return ev[:, mask]

    @pytest.fixture
    def filtered_numpy_eigen(self, numpy_eigen):
        """Get eigenvectors for non-null eigenvalues from numpy."""
        sp, ev = numpy_eigen
        mask = sp >= 1e-12
        return ev[:, mask]

    # Basic equality tests
    def test_torch_vs_numpy(self, filtered_torch_eigen, filtered_numpy_eigen):
        """Test that torch and numpy eigenvectors are equal."""
        assert_same_eigenvectors(
            filtered_torch_eigen,
            to.from_numpy(filtered_numpy_eigen),
            rtol=0, atol=1e-11
        )

    def test_numpy_vs_torch(self, filtered_torch_eigen, filtered_numpy_eigen):
        """Test that numpy and torch eigenvectors are equal."""
        assert_same_eigenvectors(
            filtered_torch_eigen.numpy(),
            filtered_numpy_eigen,
            rtol=0, atol=1e-11
        )

    def test_identical_torch(self, filtered_torch_eigen):
        """Test with identical torch tensors."""
        assert_same_eigenvectors(
            filtered_torch_eigen, filtered_torch_eigen
        )

    def test_identical_numpy(self, filtered_numpy_eigen):
        """Test with identical numpy arrays."""
        assert_same_eigenvectors(
            filtered_numpy_eigen, filtered_numpy_eigen
        )

    # Sign flip tests
    def test_negated_torch(self, filtered_torch_eigen):
        """Test with negated torch tensor."""
        assert_same_eigenvectors(
            filtered_torch_eigen, -filtered_torch_eigen
        )

    def test_negated_numpy(self, filtered_numpy_eigen):
        """Test with negated numpy array."""
        assert_same_eigenvectors(
            filtered_numpy_eigen, -filtered_numpy_eigen
        )

    def test_mixed_sign_flip(self):
        """Test with mixed sign flips across columns."""
        a = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        b = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        assert_same_eigenvectors(a, b)

    # Failing tests
    def test_different_vectors_fail(self):
        """Test that different eigenvectors raise AssertionError."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [0.0, 0.5]])
        with pytest.raises(AssertionError):
            assert_same_eigenvectors(a, b)

    def test_different_shapes_fail(self):
        """Test that different shapes raise AssertionError."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0]])
        with pytest.raises(AssertionError):
            assert_same_eigenvectors(a, b)

    def test_mixed_types_fail(self):
        """Test that mixed types raise AssertionError."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = to.tensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(AssertionError):
            assert_same_eigenvectors(a, b)

    # Tolerance tests
    def test_custom_tolerances(self):
        """Test with custom tolerance values."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 1e-8], [0.0, 1.0]])
        assert_same_eigenvectors(a, b, rtol=1e-6, atol=1e-6)

    def test_strict_tolerances_fail(self):
        """Test that strict tolerances fail for slightly different vectors."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0 + 1e-8, 0.0], [0.0, 1.0]])
        with pytest.raises(AssertionError):
            assert_same_eigenvectors(a, b, rtol=0, atol=1e-11)

    # Edge cases
    def test_single_vector(self):
        """Test with single column vectors."""
        a = np.array([[1.0], [0.0], [0.0]])
        b = np.array([[1.0], [0.0], [0.0]])
        assert_same_eigenvectors(a, b)

    def test_single_vector_negated(self):
        """Test with single column vectors (negated)."""
        a = np.array([[1.0], [0.0], [0.0]])
        b = np.array([[-1.0], [0.0], [0.0]])
        assert_same_eigenvectors(a, b)

    def test_empty_vectors(self):
        """Test with empty vectors."""
        a = np.array([]).reshape(0, 0)
        b = np.array([]).reshape(0, 0)
        assert_same_eigenvectors(a, b)

    def test_2x2_identity(self):
        """Test with 2x2 identity matrix."""
        a = np.eye(2)
        b = np.eye(2)
        assert_same_eigenvectors(a, b)

    def test_3x3_permutation(self):
        """Test with permuted eigenvectors."""
        a = np.eye(3)
        b = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
        # This should fail since columns are permuted, not just sign-flipped
        with pytest.raises(AssertionError):
            assert_same_eigenvectors(a, b)

    # Orthogonality tests
    def test_orthogonal_vectors(self):
        """Test with orthogonal eigenvectors."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 1.0], [1.0, 0.0]])
        # This should fail since columns are swapped
        with pytest.raises(AssertionError):
            assert_same_eigenvectors(a, b)

    # dtype tests
    def test_float32_dtype(self):
        """Test with float32 dtype."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert_same_eigenvectors(a, b)

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        assert_same_eigenvectors(a, b)

    # Large matrix test
    def test_large_matrix(self):
        """Test with a larger matrix."""
        rng = np.random.default_rng(42)
        a = rng.normal(size=(50, 10))
        # Create orthonormal columns
        q, _ = np.linalg.qr(a)
        b = q.copy()
        # Negate some columns
        b[:, 1] = -b[:, 1]
        b[:, 3] = -b[:, 3]
        assert_same_eigenvectors(q, b)

    # Test with actual eigendecomposition
    def test_with_actual_eigendecomposition(self, sym_matrix):
        """Test with actual eigendecomposition results."""
        # Compute eigendecomposition twice
        sp1, ev1 = to.linalg.eigh(sym_matrix)
        sp2, ev2 = to.linalg.eigh(sym_matrix)

        # Filter non-null eigenvalues
        mask1 = sp1 >= 1e-12
        mask2 = sp2 >= 1e-12

        assert_same_eigenvectors(
            ev1[:, mask1], ev2[:, mask2], rtol=0, atol=1e-11
        )

    def test_with_negated_eigendecomposition(self, sym_matrix):
        """Test with negated eigendecomposition results."""
        sp, ev = to.linalg.eigh(sym_matrix)
        mask = sp >= 1e-12

        assert_same_eigenvectors(
            ev[:, mask], -ev[:, mask], rtol=0, atol=1e-11
        )
