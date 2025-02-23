#!/usr/bin/env python3

from langsmith import traceable
from tavily import TavilyClient

from deepresearcher.logger import logger


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
