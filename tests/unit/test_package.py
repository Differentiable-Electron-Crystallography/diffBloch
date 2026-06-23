"""Smoke tests: the package imports and exposes a version."""

import diffBloch


def test_version_is_a_string() -> None:
    assert isinstance(diffBloch.__version__, str)
    assert diffBloch.__version__
