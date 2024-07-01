import pytest
from unittest.mock import MagicMock
import pandas as pd

@pytest.fixture
def mock_pandas_read_csv(monkeypatch):
    # Create a MagicMock object to replace pd.read_csv
    mock = MagicMock(name='read_csv')

    # Use monkeypatch to replace pd.read_csv with our mock
    def mock_read_csv(filepath, *args, **kwargs):
        return mock(filepath, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    yield mock

    mock.reset_mock()  # Reset mock after each test
