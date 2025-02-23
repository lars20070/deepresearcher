#!/usr/bin/env python3

import os

import pytest

from deepresearcher.logger import logger
from deepresearcher.utils import deduplicate_and_format_sources, format_sources, perplexity_search, tavily_search


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


@pytest.mark.skip(reason="Each Perplexity API call costs money.")
def test_perplexity_search(topic: str, load_env: None) -> None:
    # Number of search loops
    n = 0

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


def test_deduplicate_and_format_sources() -> None:
    logger.info("Testing deduplication and formatting of search results.")

    # Test data with duplicate URLs
    mock_results = {
        "results": [
            {"title": "First Article", "url": "https://example.com", "content": "Summary 1", "raw_content": "Full content 1"},
            {
                "title": "Second Article",
                "url": "https://example.com",  # Duplicate URL
                "content": "Summary 2",
                "raw_content": "Full content 2",
            },
            {"title": "Third Article", "url": "https://example2.com", "content": "Summary 3", "raw_content": "Full content 3"},
        ]
    }

    formatted = deduplicate_and_format_sources(mock_results, max_tokens_per_source=10, include_raw_content=True)
    logger.debug(f"Formatted search results: {formatted}")

    # Verify deduplication
    assert formatted.count("https://example.com") == 1  # URL should appear only once
    assert formatted.count("https://example2.com") == 1

    # Verify formatting
    assert "Sources:" in formatted
    assert "Most relevant content from source:" in formatted
    assert "First Article" in formatted
    assert "Second Article" not in formatted  # Duplicate should be removed
    assert "Third Article" in formatted
