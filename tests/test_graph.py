#!/usr/bin/env python3

from deepresearcher.graph import generate_query, graph
from deepresearcher.logger import logger
from deepresearcher.state import SummaryState


def test_generate_query_explicit() -> None:
    logger.info("Testing generate_query() function.")
    state = SummaryState(research_topic="AI")
    result = generate_query(state, config={})
    logger.info(f"Final state: {result}")

    assert "search_query" in result
    assert result["search_query"] == "dummy"


def test_graph_compiles() -> None:
    logger.info("Testing graph compiles correctly.")
    logger.info(f"Graph nodes: {graph.nodes}")

    assert "__start__" in graph.nodes
    assert "generate_query" in graph.nodes
    # assert "__end__" in graph.nodes # TODO: Why is __end__ not in the nodes?
