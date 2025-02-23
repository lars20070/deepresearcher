#!/usr/bin/env python3
import os

import pytest

from deepresearcher.graph import generate_query, graph, web_research
from deepresearcher.logger import logger
from deepresearcher.state import SummaryState


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Ollama not running in the CI pipeline. Run only locally.",
)
def test_generate_query_explicit(topic: str) -> None:
    logger.info("Testing generate_query() function.")
    state = SummaryState(research_topic=topic)
    result = generate_query(state, config={})
    logger.info(f"Final state: {result}")

    assert "search_query" in result
    assert topic in result["search_query"]


def test_web_research(topic: str, load_env: None) -> None:
    logger.info("Testing web research.")
    logger.debug(f"TOPIC: {topic}")
    state = SummaryState(research_topic=topic, search_query=f"Tell me about {topic}")
    result = web_research(state, config={})
    logger.debug(f"Web search result: {result}")

    assert "sources_gathered" in result
    assert "research_loop_count" in result
    assert "web_research_results" in result
    assert len(result["web_research_results"]) > 0


def test_graph_compiles() -> None:
    logger.info("Testing graph compiles correctly.")
    logger.info(f"Graph nodes: {graph.nodes}")

    assert "__start__" in graph.nodes
    assert "generate_query" in graph.nodes
    # assert "__end__" in graph.nodes # TODO: Why is __end__ not in the nodes?
