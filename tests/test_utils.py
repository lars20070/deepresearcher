#!/usr/bin/env python3

import os

import pytest

from deepresearcher.logger import logger
from deepresearcher.utils import tavily_search


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="No API key in CI. Run only locally.",
)
def test_tavily_search(topic: str, load_env: None) -> None:
    # Number of results
    n = 3

    # Check whether TAVILY_API_KEY is set
    assert os.getenv("TAVILY_API_KEY") is not None

    logger.info("Searching with Tavily.")
    result = tavily_search(topic, include_raw_content=True, max_results=n)
    logger.debug(f"Entire search result: {result}")

    # Check whether the result contains a 'results' key
    assert "results" in result
    logger.debug(f"Search result list: {result['results']}")

    # Check if the number of results is correct
    assert len(result["results"]) == n
