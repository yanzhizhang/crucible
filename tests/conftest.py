"""Shared fixtures: one synthetic store built once per test session."""

from __future__ import annotations

import pytest

from quarry.db import open_db
from quarry.synth import SynthTruth, make_store


@pytest.fixture(scope="session")
def store(tmp_path_factory: pytest.TempPathFactory) -> SynthTruth:
    """A generated A-share store plus its injected ground truth.

    40 symbols so cross-sections clear the default ``min_names=20`` guard in
    :mod:`assay` -- a narrower panel would null every IC and make the step 2
    gate pass vacuously. Intraday bars are thinned to 60 per session because
    the tests exercise bar *alignment*, not bar count.
    """
    root = tmp_path_factory.mktemp("crucible_store")
    return make_store(root, n_symbols=40, bars_per_day=60, with_ticks=True, tick_symbols=2)


@pytest.fixture(scope="session")
def conn(store: SynthTruth):  # noqa: ANN201
    """DuckDB connection with views registered over the store."""
    c = open_db(store.root)
    yield c
    c.close()
