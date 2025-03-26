#!/usr/bin/env python3

import json
import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from deepresearcher.logger import logger
from deepresearcher.state import Feedback
from deepresearcher.utils import deduplicate_and_format_sources, duckduckgo_search, format_sources, invoke_llm, perplexity_search, tavily_search


def test_duckduckgo_search(topic: str, load_env: None) -> None:
    # Number of results
    n = 3

    logger.info("Testing searching with DuckDuckGo.")
    result = duckduckgo_search(topic, max_results=n, fetch_full_page=False)
    logger.debug(f"Entire search result: {result}")

    # Check whether the result contains a 'results' key
    assert "results" in result
    logger.debug(f"Number of search results: {len(result['results'])}")
    if len(result["results"]) > 0:
        for i, item in enumerate(result["results"]):
            logger.debug(f"Result {i + 1}:\n{json.dumps(item, indent=2)}")

    # Check if the number of results is correct
    assert len(result["results"]) == n


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


@pytest.mark.paid
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


@pytest.mark.ollama
@pytest.mark.example
def test_ollama_structured_output() -> None:
    """Minimal example of generating structured output using a local Ollama model"""
    logger.info("Minimal example of generating structured output using a local Ollama model.")

    class Person(BaseModel):
        age: int
        name: str

    model = ChatOllama(model="llama3.3").with_structured_output(Person, method="json_schema")
    result = model.invoke("My name is Bill and I am 27 years old.")
    logger.debug(f"Structured output:\n{result}")

    assert result.age is not None
    assert result.age == 27
    assert result.name is not None
    assert "Bill" in result.name


@pytest.mark.paid
def test_invoke_llm(topic: str, load_env: None) -> None:
    """Test invoke_llm with both structured and unstructured output."""

    logger.info("Testing invoke_llm() method with unstructured output.")
    provider = "openai"
    model = "gpt-4o"

    prompt = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"Write a short paragraph about {topic}."),
    ]

    unstructured_response = invoke_llm(
        provider=provider,
        model=model,
        prompt=prompt,
    )
    logger.debug(f"Unstructured response:\n{unstructured_response}")

    assert unstructured_response.content is not None
    assert isinstance(unstructured_response.content, str)
    assert len(unstructured_response.content) > 0
    assert topic in unstructured_response.content.lower()

    logger.info("Testing invoke_llm() method with structured output.")

    provider = "openai"
    model = "gpt-4o"
    prompt = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"Please provide feedback for research on {topic}. I do not really know what that is."),
    ]

    structured_response = invoke_llm(
        provider=provider,
        model=model,
        prompt=prompt,
        schema_class=Feedback,
    )
    logger.debug(f"Structured response:\n{structured_response}")

    assert structured_response.grade is not None
    assert structured_response.grade in ["pass", "fail"]
    assert structured_response.follow_up_queries is not None
    assert len(structured_response.follow_up_queries) > 0


@pytest.mark.ollama
def test_invoke_llm_ollama(topic: str) -> None:
    """Test invoke_llm with both structured and unstructured output."""

    logger.info("Testing invoke_llm() method with unstructured output.")
    provider = "ollama"
    model = "llama3.3"

    prompt = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"Write a short paragraph about {topic}."),
    ]

    unstructured_response = invoke_llm(
        provider=provider,
        model=model,
        prompt=prompt,
    )
    logger.debug(f"Unstructured response:\n{unstructured_response}")

    assert unstructured_response.content is not None
    assert isinstance(unstructured_response.content, str)
    assert len(unstructured_response.content) > 0
    assert topic in unstructured_response.content.lower()

    logger.info("Testing invoke_llm() method with structured output.")

    provider = "ollama"
    model = "llama3.3"
    prompt = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"Please provide feedback for research on {topic}. I do not really know what that is."),
    ]

    structured_response = invoke_llm(
        provider=provider,
        model=model,
        prompt=prompt,
        schema_class=Feedback,
    )
    logger.debug(f"Structured response:\n{structured_response}")

    assert structured_response.grade is not None
    assert structured_response.grade in ["pass", "fail"]
    assert structured_response.follow_up_queries is not None
    assert len(structured_response.follow_up_queries) > 0

    pass
