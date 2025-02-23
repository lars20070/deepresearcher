#!/usr/bin/env python3

import pytest


@pytest.fixture
def topic() -> str:
    """Provide a research topic for unit testing."""
    return "syzygy"
