"""Gate @pytest.mark.slow tests behind --runslow (bending runs use dt≈0.002)."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow (dt=0.002 bending) tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end test (opt-in via --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
