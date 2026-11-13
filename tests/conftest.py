import pytest

from tsbench.config import ROOT, load_protocol
from tsbench.data.prepare import load_processed


@pytest.fixture(scope="session")
def protocol():
    return load_protocol()


@pytest.fixture(scope="session")
def df(protocol):
    # Raw data is not committed; CI (and fresh clones) skip data-backed tests until `bench prepare` has run.
    if not (ROOT / protocol.dataset.processed).exists():
        pytest.skip(f"{protocol.dataset.processed} missing: run scripts/download_data.py and `bench prepare`")
    return load_processed(protocol)
