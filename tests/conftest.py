#!/usr/bin/env python3

import pytest
from dotenv import load_dotenv


@pytest.fixture
def load_env() -> None:
    """Load environment variables."""
    load_dotenv()


@pytest.fixture
def topic() -> str:
    """Provide a research topic for unit testing."""
    return "syzygy"
