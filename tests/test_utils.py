#!/usr/bin/env python3

import os

from deepresearcher.logger import logger
from deepresearcher.utils import format_sources, perplexity_search, tavily_search


def test_tavily_search(topic: str, load_env: None) -> None:
    # Number of results
    n = 3

    # Check whether TAVILY_API_KEY is set
    assert os.getenv("TAVILY_API_KEY") is not None

    logger.info("Testing searching with Tavily.")
    result = tavily_search(topic, include_raw_content=True, max_results=n)
    logger.debug(f"Entire search result: {result}")

    # Check whether the result contains a 'results' key
    assert "results" in result
    logger.debug(f"Number of search results: {len(result['results'])}")
    logger.debug(f"Search result list: {result['results']}")

    # Check if the number of results is correct
    assert len(result["results"]) == n


def test_perplexity_search(topic: str, load_env: None) -> None:
    # Number of search loops i.e. Perplexity API calls
    n = 1

    logger.info("Testing searching with Perplexity.")
    result = perplexity_search(topic, perplexity_search_loop_count=n)
    logger.debug(f"Entire search result: {result}")

    # Check whether the result contains a 'results' key
    assert "results" in result
    logger.debug(f"Number of search results: {len(result['results'])}")
    logger.debug(f"Search result list: {result['results']}")

    # Check if the number of results is correct
    assert len(result["results"]) > 0  # TODO: Why is the number of results not equal the number of Perplexity search loops?


def test_format_sources() -> None:
    logger.info("Testing formatting search results.")
    mock_results = {
        "results": [{"title": "First Article", "url": "https://example1.com"}, {"title": "Second Article", "url": "https://example2.com"}]
    }
    formatted = format_sources(mock_results)

    assert formatted == "* First Article : https://example1.com\n* Second Article : https://example2.com"
