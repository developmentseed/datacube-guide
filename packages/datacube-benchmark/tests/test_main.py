"""Smoke test for the ``datacube_benchmark.main`` CLI entry point."""

import datacube_benchmark


def test_main_prints_array_and_chunk_sizes(capsys):
    # main() builds a lazy dask-backed DataArray, so nothing is allocated.
    datacube_benchmark.main()
    out = capsys.readouterr().out
    assert "Array size:" in out
    assert "Chunk size:" in out
