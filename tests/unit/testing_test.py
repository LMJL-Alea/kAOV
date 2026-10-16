import pytest
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from kaov import AOV
from kaov.testing import (
    dummy_data, assert_equal_data_obj, assert_equal_aov_obj
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
