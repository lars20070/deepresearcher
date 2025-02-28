#!/usr/bin/env python3
import os

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from deepresearcher.graph import (
    finalize_summary,
    generate_query,
    generate_report_plan,
    graph,
    graph_report,
    reflect_on_summary,
    route_research,
    summarize_sources,
    web_research,
)
from deepresearcher.logger import logger
from deepresearcher.state import ReportState, SummaryState


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
def test_summarize_sources(summary_state: dict) -> None:
    logger.info("Testing summarize_sources() function.")
    result = summarize_sources(summary_state, config={})
    logger.debug(f"Summarized sources: {result}")

    assert "running_summary" in result


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Ollama not running in the CI pipeline. Run only locally.",
)
def test_reflect_on_summary(summary_state: dict) -> None:
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


@pytest.mark.skip(reason="Each API call costs money.")
def test_EXAMPLE_chat_model_anthropic(topic: str, load_env: None) -> None:
    """
    Minimal example of a chat model in LangChain
    https://python.langchain.com/api_reference/langchain/chat_models.html#
    """
    logger.info("Starting minimal example of a chat model in LangChain (Anthropic).")

    # Ensure API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Initialize the chat model
    model = init_chat_model(
        model="claude-3-5-sonnet-latest",
        model_provider="anthropic",
        temperature=0.7,
    )

    # Create messages
    system_message = SystemMessage(content="You are an unhelpful reluctant AI assistant.")
    user_message = HumanMessage(content=f"What is {topic}?")

    # Send messages to the model
    response = model.invoke([system_message, user_message])
    logger.debug(f"Response:\n{response.content}")
    logger.debug(f"Response metadata:\n{response.response_metadata}")

    assert response.content is not None
    assert "claude-3-5-sonnet" in response.response_metadata["model"]


@pytest.mark.skip(reason="Each API call costs money.")
def test_EXAMPLE_chat_model_openai(topic: str, load_env: None) -> None:
    """
    Minimal example of a chat model in LangChain
    https://python.langchain.com/api_reference/langchain/chat_models.html#
    """
    logger.info("Starting minimal example of a chat model in LangChain (OpenAI).")
    # model_name = "o1-mini"  # Does not support structured output
    model_name = "gpt-4o"

    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    model = init_chat_model(model=model_name, model_provider="openai")
    response = model.invoke(f"What is {topic}?")
    logger.debug(f"Response:\n{response.content}")
    logger.debug(f"Response metadata:\n{response.response_metadata}")

    assert response.content is not None
    assert model_name in response.response_metadata["model_name"]  # TODO: Inconsistent with Anthropic response metadata (model vs model_name)

    # Example of structured response
    # Simply pass a Pydantic model to the LangChain interface

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: int | None = Field(default=None, description="How funny the joke is, from 1 to 10")

    model_structured = model.with_structured_output(Joke)
    joke_response = model_structured.invoke(f"Tell me a joke about {topic}")
    logger.debug(f"Joke response: {joke_response}")

    assert joke_response.setup is not None
    assert joke_response.punchline is not None


@pytest.mark.skip(reason="Each API call costs money.")
@pytest.mark.asyncio
async def test_generate_report_plan(topic: str, load_env: None) -> None:
    logger.info("Testing generation of the report plan.")

    state = ReportState(topic=topic)
    result = await generate_report_plan(state, config={})
    logger.debug(f"Report plan: {result}")


def test_graph_report_compiles() -> None:
    logger.info("Testing report graph compiles correctly.")
    logger.info(f"Graph nodes: {graph_report.nodes}")

    assert "__start__" in graph_report.nodes
    assert "generate_report_plan" in graph_report.nodes
    # assert "__end__" in graph_report.nodes # TODO: Why is __end__ not in the nodes?
