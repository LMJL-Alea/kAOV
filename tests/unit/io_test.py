import gzip
import os

import dill
import pytest

from kaov import AOV
from kaov.io import save, load
from kaov.testing import dummy_data, assert_equal_aov_obj


class Test_save_load:
    """Test suite for save and load functions."""

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
    def saved_compressed_path(self, aov_object, tmp_path):
        """Save an object with compression and return the file path."""
        file_path = str(tmp_path / "model")
        save(aov_object, file_path, compress=True)
        return file_path + ".gz"

    @pytest.fixture
    def saved_uncompressed_path(self, aov_object, tmp_path):
        """Save an object without compression and return the file path."""
        file_path = str(tmp_path / "model.pkl")
        save(aov_object, file_path, compress=False)
        return file_path

    def test_save_with_compression(self, aov_object, saved_compressed_path):
        """Test saving with compression enabled (default)."""
        # Check file exists with .gz extension
        assert os.path.exists(saved_compressed_path)

        # Verify it's actually gzip compressed
        with open(saved_compressed_path, 'rb') as f:
            magic_bytes = f.read(2)
        assert magic_bytes == b'\x1f\x8b'  # gzip magic number

        # Try to read it as gzip
        with gzip.open(saved_compressed_path, 'rb') as f:
            loaded_obj = dill.load(f)
        # Verify the loaded object
        assert_equal_aov_obj(aov_object, loaded_obj)

    def test_save_without_compression(
        self, aov_object, saved_uncompressed_path
    ):
        """Test saving without compression."""
        # Check file exists without .gz extension
        assert os.path.exists(saved_uncompressed_path)
        assert not saved_uncompressed_path.endswith('.gz')

        # Try to read it as pickle
        with open(saved_uncompressed_path, 'rb') as f:
            loaded_obj = dill.load(f)
        # Verify the loaded object
        assert_equal_aov_obj(aov_object, loaded_obj)

    def test_load_compressed(self, aov_object, saved_compressed_path):
        """Test roundtrip save and load with compression."""
        loaded_obj = load(saved_compressed_path, compressed=True)
        # Verify the loaded object
        assert_equal_aov_obj(aov_object, loaded_obj)

    def test_load_uncompressed(self, aov_object, saved_uncompressed_path):
        """Test roundtrip save and load without compression."""
        loaded_obj = load(saved_uncompressed_path, compressed=False)
        # Verify the loaded object
        assert_equal_aov_obj(aov_object, loaded_obj)

    def test_load_compressed_with_default_param(
        self, aov_object, saved_compressed_path
    ):
        """Test loading compressed file with default compressed=True."""
        loaded_obj = load(saved_compressed_path)
        # Verify the loaded object
        assert_equal_aov_obj(aov_object, loaded_obj)

    def test_save_with_explicit_gz_extension(self, aov_object, tmp_path):
        """Test saving with explicit .gz extension doesn't duplicate it."""
        file_path = str(tmp_path / "model.gz")
        save(aov_object, file_path)
        # Should not add another .gz
        assert os.path.exists(file_path)
        assert not os.path.exists(file_path + ".gz")

    def test_save_to_nonexistent_directory(self, aov_object, tmp_path):
        """Test saving to a nonexistent directory raises error."""
        file_path = str(tmp_path / "nonexistent" / "model")

        with pytest.raises(FileNotFoundError):
            save(aov_object, file_path)

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a nonexistent file raises error."""
        file_path = str(tmp_path / "nonexistent_model.gz")

        with pytest.raises(FileNotFoundError):
            load(file_path, compressed=True)

    def test_load_wrong_compression_flag(self, aov_object, tmp_path):
        """Test loading with wrong compression flag raises error."""
        # Save with compression
        compressed_path = str(tmp_path / "model_compressed")
        save(aov_object, compressed_path, compress=True)

        # Try to load compressed file with compressed=False
        with pytest.raises(Exception):  # gzip.BadGzipFile or similar
            load(compressed_path + ".gz", compressed=False)

        # Save without compression
        uncompressed_path = str(tmp_path / "model_uncompressed.pkl")
        save(aov_object, uncompressed_path, compress=False)

        # Try to load uncompressed file with compressed=True
        with pytest.raises(Exception):  # EOFError or similar
            load(uncompressed_path, compressed=True)
