"""Public package metadata contract."""

from opendubstream import APPLICATION_NAME, __version__


def test_package_exposes_the_canonical_application_name() -> None:
    """Consumers can identify the bootstrap package without importing product code."""
    assert APPLICATION_NAME == "OpenDubStream"


def test_package_exposes_the_bootstrap_version() -> None:
    """Consumers receive a concrete version value from the package boundary."""
    assert __version__ == "0.1.0"
