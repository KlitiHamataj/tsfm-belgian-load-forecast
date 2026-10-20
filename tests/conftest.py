import pytest

from tsbench.config import load_protocol
from tsbench.data.prepare import load_processed


@pytest.fixture(scope="session")
def protocol():
    return load_protocol()


@pytest.fixture(scope="session")
def df(protocol):
    return load_processed(protocol)
