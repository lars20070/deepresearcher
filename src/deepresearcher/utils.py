#!/usr/bin/env python3

import asyncio
import os
from enum import Enum
from typing import Any, TypeVar

import requests
from duckduckgo_search import DDGS
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel
from tavily import AsyncTavilyClient, TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from deepresearcher.logger import logger
from deepresearcher.state import SearchQuery, Section

# Return type for schema classes
T = TypeVar("T", bound=BaseModel)


def retry_with_backoff(func: callable) -> callable:
    """
    Retry decorator with exponential backoff.

    For example, the first retry will wait 20 seconds, the second 40 seconds, the third 80 seconds, and so on. Stopping after 5 attempts.
    """
    retry_min = 20
    retry_max = 1000
    retry_attempts = 5

    return retry(wait=wait_exponential(min=retry_min, max=retry_max), stop=stop_after_attempt(retry_attempts))(func)


@retry_with_backoff
@traceable
def fetch_page_content(url: str) -> str:
    """Fetch the content of a webpage given its URL."""
    import urllib.error
    import urllib.request

    from bs4 import BeautifulSoup

    try:
        response = urllib.request.urlopen(url, timeout=10)
        html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
    except urllib.error.HTTPError as e:
        if e.code in (403, 401):
            logger.error(f"Authentication error for {url}: {e.code}")
            return f"[Error: Access denied to {url} (code {e.code})]"
        else:
            logger.error(f"HTTP error for {url}: {e.code}")
            raise  # Will be retried by decorator
    except urllib.error.URLError as e:
        logger.error(f"Network error for {url}: {str(e)}")
        raise  # Will be retried by decorator


@retry_with_backoff
@traceable
def duckduckgo_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> dict[str, list[dict[str, str]]]:
    """Search the web using DuckDuckGo.

    Args:
        query (str): The search query to execute
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Same as content since DDG doesn't provide full page content
    """
    logger.info(f"Searching the web using DuckDuckGo for: {query}")
    results = []

    try:
        with DDGS() as ddgs:
            try:
                search_results = list(ddgs.text(query, max_results=max_results))
                if not search_results:
                    logger.warning(f"DuckDuckGo returned no results for: {query}")
                    return {"results": []}

            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Network error during search: {str(e)}")
                raise  # Will be retried by decorator

            for r in search_results:
                url = r.get("href")
                title = r.get("title")
                content = r.get("body")

                if not all([url, title, content]):
                    logger.warning(f"Warning: Incomplete result from DuckDuckGo: {r}")
                    continue

                raw_content = content
                if fetch_full_page:
                    try:
                        raw_content = fetch_page_content(url)

                    except Exception as e:
                        logger.error(f"Error: Failed to fetch full page content for {url}: {str(e)}")

                # Add result to list
                result = {"title": title, "url": url, "content": content, "raw_content": raw_content}
                results.append(result)

            return {"results": results}

    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}")
        raise  # Will be retried by decorator


@traceable
def tavily_search(query: str, include_raw_content: bool = True, max_results: int = 3) -> dict:
    """
    Search the web using the Tavily API.

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
    """
    logger.info(f"Searching the web using Tavily for: {query}")
    tavily_client = TavilyClient()

    return tavily_client.search(query, max_results=max_results, include_raw_content=include_raw_content)


@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> dict[str, Any]:
    """Search the web using the Perplexity API.

    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int): The loop step for perplexity search (starts at 0)

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
    """
    logger.info(f"Searching the web using Perplexity for: {query}")

    headers = {"accept": "application/json", "content-type": "application/json", "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Search the web and provide factual information with sources."},
            {"role": "user", "content": query},
        ],
    }

    response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
    response.raise_for_status()  # Raise exception for bad status codes

    # Parse the response
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Perplexity returns a list of citations for a single search result
    citations = data.get("citations", ["https://perplexity.ai"])

    # Return first citation with full content, others just as references
    results = [
        {"title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1", "url": citations[0], "content": content, "raw_content": content}
    ]

    # Add additional citations without duplicating content
    for i, citation in enumerate(citations[1:], start=2):
        results.append(
            {
                "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
                "url": citation,
                "content": "See above for full content",
                "raw_content": None,
            }
        )

    return {"results": results}


def format_sources(search_results: dict) -> str:
    """
    Format search results into a bullet-point list of sources.

    Args:
        search_results (dict): Tavily search response containing results

    Returns:
        str: Formatted string with sources and their URLs
    """
    logger.info("Formatting search results")
    return "\n".join(f"* {source['title']} : {source['url']}" for source in search_results["results"])


def deduplicate_and_format_sources(search_response: dict, max_tokens_per_source: int, include_raw_content: bool = False) -> str:
    """
    Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for _i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def get_config_value(value: str | Enum) -> str:
    """
    Helper function to handle both string and enum cases of configuration values
    """
    return value if isinstance(value, str) else value.value


def format_sections(sections: list[Section]) -> str:
    """Format a list of sections into a string"""
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
            {"=" * 60}
            Section {idx}: {section.name}
            {"=" * 60}
            Description:
            {section.description}
            Requires Research: 
            {section.research}

            Content:
            {section.content if section.content else "[Not yet written]"}

            """
    return formatted_str


@traceable
async def tavily_search_async(search_queries: list[str]) -> list[dict]:
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
            List[dict]: List of search responses from Tavily API, one per query. Each response has format:
                {
                    'query': str, # The original search query
                    'follow_up_questions': None,
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """

    search_tasks = []
    tavily_async_client = AsyncTavilyClient()
    for query in search_queries:
        search_tasks.append(tavily_async_client.search(query, max_results=5, include_raw_content=True, topic="general"))

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs


@traceable
def perplexity_search_2(search_queries: list[SearchQuery]) -> list[dict]:
    """Search the web using the Perplexity API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """

    headers = {"accept": "application/json", "content-type": "application/json", "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}

    search_docs = []
    for query in search_queries:
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "Search the web and provide factual information with sources."},
                {"role": "user", "content": query},
            ],
        }

        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse the response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])

        # Create results list for this query
        results = []

        # First citation gets the full content
        results.append(
            {
                "title": "Perplexity Search, Source 1",
                "url": citations[0],
                "content": content,
                "raw_content": content,
                "score": 1.0,  # Adding score to match Tavily format
            }
        )

        # Add additional citations without duplicating content
        for i, citation in enumerate(citations[1:], start=2):
            results.append(
                {
                    "title": f"Perplexity Search, Source {i}",
                    "url": citation,
                    "content": "See primary source for full content",
                    "raw_content": None,
                    "score": 0.5,  # Lower score for secondary sources
                }
            )

        # Format response to match Tavily structure
        search_docs.append({"query": query, "follow_up_questions": None, "answer": None, "images": [], "results": results})

    return search_docs


def invoke_llm(provider: str, model: str, prompt: list[SystemMessage | HumanMessage], schema_class: type[T] = None) -> T:
    """
    Invoke an LLM to generate either unstructured content
    or structured content according to a schema class.

    Args:
        provider: Model provider e.g. 'openai', 'anthropic'
        model: Model name to use e.g. 'o1', 'laude-3-5-sonnet-latest'
        prompt: prompt passing to the LLM
        schema_class: Pydantic model class for structured output

    Returns:
        The structured response object
    """
    # Initialize model
    llm = init_chat_model(
        model_provider=provider,
        model=model,
    )
    if schema_class is not None:
        # LLM generates structured output
        llm = llm.with_structured_output(schema_class)

    # Generate response
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        logger.error(f"Error from structured LLM: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "json"):
            logger.error(f"Error details: {e.response.json()}")
        raise
