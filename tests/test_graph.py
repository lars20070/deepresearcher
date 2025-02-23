#!/usr/bin/env python3
import os

import pytest

from deepresearcher.graph import finalize_summary, generate_query, graph, reflect_on_summary, route_research, summarize_sources, web_research
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
    state = SummaryState(research_topic=topic, search_query=f"Tell me about {topic}")
    result = web_research(state, config={})
    logger.debug(f"Web search result: {result}")

    assert "sources_gathered" in result
    assert "research_loop_count" in result
    assert "web_research_results" in result
    assert len(result["web_research_results"]) > 0


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Ollama not running in the CI pipeline. Run only locally.",
)
def test_summarize_sources(topic: str, summary_state: dict) -> None:
    logger.info("Testing summarize_sources() function.")
    result = summarize_sources(summary_state, config={})
    logger.debug(f"Summarized sources: {result}")

    assert "running_summary" in result


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Ollama not running in the CI pipeline. Run only locally.",
)
def test_reflect_on_summary(topic: str, summary_state: dict) -> None:
    logger.info("Testing reflect_on_summary() function.")
    result = reflect_on_summary(summary_state, config={})
    logger.debug(f"New search query: {result}")

    assert "search_query" in result


def test_finalize_summary() -> None:
    logger.info("Testing finalize_summary() function.")

    # Create test state with summary and sources
    state = SummaryState(
        running_summary="Test summary content", sources_gathered=["* Source 1 : http://example1.com", "* Source 2 : http://example2.com"]
    )

    # Run finalize_summary
    result = finalize_summary(state)
    logger.debug(f"Finalized summary: {result}")

    # Verify structure and content
    assert "running_summary" in result
    assert result["running_summary"].startswith("## Summary")
    assert "### Sources:" in result["running_summary"]
    assert "Source 1" in result["running_summary"]
    assert "Source 2" in result["running_summary"]


def test_route_research() -> None:
    logger.info("Testing route_research() function.")

    # Test continuing research (loop count within limit)
    state = SummaryState(research_loop_count=2)
    route = route_research(state, config={"configurable": {"max_web_research_loops": 3}})
    assert route == "web_research"

    # Test finishing research (loop count at limit)
    state = SummaryState(research_loop_count=3)
    route = route_research(state, config={"configurable": {"max_web_research_loops": 3}})
    assert route == "finalize_summary"

    # Test finishing research (loop count exceeds limit)
    state = SummaryState(research_loop_count=4)
    route = route_research(state, config={"configurable": {"max_web_research_loops": 3}})
    assert route == "finalize_summary"


def test_graph_compiles() -> None:
    logger.info("Testing graph compiles correctly.")
    logger.info(f"Graph nodes: {graph.nodes}")

    assert "__start__" in graph.nodes
    assert "generate_query" in graph.nodes
    # assert "__end__" in graph.nodes # TODO: Why is __end__ not in the nodes?
