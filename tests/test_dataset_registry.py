"""Tests for TBI.data — unified dataset loading."""

import pytest

from TBI.data import load_dataset, list_datasets, DATASETS


# ===========================================================================
# list_datasets
# ===========================================================================

class TestListDatasets:
    """Tests for list_datasets()."""

    def test_returns_dict(self):
        ds = list_datasets()
        assert isinstance(ds, dict)

    def test_contains_primary_datasets(self):
        ds = list_datasets()
        for name in ["cmipb", "suez2018"]:
            assert name in ds

    def test_two_datasets(self):
        ds = list_datasets()
        assert len(ds) == 2

    def test_keys_match_DATASETS(self):
        ds = list_datasets()
        assert set(ds.keys()) == set(DATASETS.keys())


# ===========================================================================
# load_dataset — unknown name
# ===========================================================================

class TestLoadDatasetInvalid:
    """Requesting a nonexistent dataset should raise."""

    def test_raises_on_nonexistent(self):
        with pytest.raises((KeyError, ValueError)):
            load_dataset("nonexistent_dataset_xyz")

    def test_error_message_lists_available(self):
        with pytest.raises((KeyError, ValueError), match="Available"):
            load_dataset("bogus_name")
