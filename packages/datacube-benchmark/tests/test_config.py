"""Tests for the ``Config`` dataclass."""

import dataclasses

from datacube_benchmark.config import Config


def test_defaults():
    config = Config()
    assert config.create_data is True
    assert config.target_array_size == "25 MB"
    assert config.data_var == "data"
    assert config.num_samples == 1
    assert config.warmup_samples == 0
    assert config.credential_provider is None
    assert config.zarr_concurrency == 128


def test_fields_can_be_overridden():
    config = Config(num_samples=5, create_data=False, target_array_size="10 MB")
    assert config.num_samples == 5
    assert config.create_data is False
    assert config.target_array_size == "10 MB"


def test_is_a_real_dataclass_with_fields():
    # Regression: previously the attributes had no annotations, so @dataclass
    # generated zero fields and keyword overrides raised TypeError.
    field_names = {f.name for f in dataclasses.fields(Config)}
    assert "num_samples" in field_names
    assert "compressor" in field_names


def test_compressor_default_is_per_instance():
    # default_factory must hand each Config its own compressor instance.
    assert Config().compressor is not Config().compressor
