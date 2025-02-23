#!/usr/bin/env python3
import os

import pytest

from deepresearcher.graph import generate_query, graph
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


def test_graph_compiles() -> None:
    logger.info("Testing graph compiles correctly.")
    logger.info(f"Graph nodes: {graph.nodes}")

    assert "__start__" in graph.nodes
    assert "generate_query" in graph.nodes
    # assert "__end__" in graph.nodes # TODO: Why is __end__ not in the nodes?
