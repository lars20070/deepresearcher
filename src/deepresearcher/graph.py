#!/usr/bin/env python3

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from deepresearcher.configuration import Configuration
from deepresearcher.logger import logger
from deepresearcher.prompts import (
    query_writer_instructions,
    reflection_instructions,
    summarizer_instructions,
)
from deepresearcher.state import SummaryState, SummaryStateInput, SummaryStateOutput
from deepresearcher.utils import (
    deduplicate_and_format_sources,
    format_sources,
    perplexity_search,
    tavily_search,
)


# Define nodes
def generate_query(state: SummaryState, config: RunnableConfig) -> dict:
    """Generate a query for web search"""
    logger.info(f"Generating a query for the research topic: {state.research_topic}")

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)
    logger.info(f"Formatted prompt: {query_writer_instructions_formatted}")

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    logger.info(f"Creating LLM model: {configurable.local_llm}")
    llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [
            SystemMessage(content=query_writer_instructions_formatted),
            HumanMessage(content="Generate a query for web search:"),
        ]
    )
    logger.info(f"LLM response: {result.content}")

    # Parse JSON response
    try:
        query = json.loads(result.content)
        if not isinstance(query, dict) or "query" not in query:
            raise ValueError("LLM response missing required 'query' field")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in LLM response: {e}")
        return {"search_query": f"search about {state.research_topic}"}
    except ValueError as e:
        logger.error(f"Invalid response structure: {e}")
        return {"search_query": f"search about {state.research_topic}"}

    return {"search_query": query["query"]}
    # TODO: How can I see in LangGraph Studio not only the `query` but also the `aspect`?
    # return {"search_query": query["query"], "search_aspect": query["aspect"]}


def web_research(state: SummaryState, config: RunnableConfig) -> dict:
    """Gather information from the web"""
    logger.info(f"Web research with search query: {state.search_query}")

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


def summarize_sources(state: SummaryState, config: RunnableConfig) -> dict:
    """Summarize the gathered sources"""
    logger.info("Summarizing the gathered sources")

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=configurable.local_llm, temperature=0)
    result = llm.invoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig) -> dict:
    """Reflect on the summary and generate a follow-up query"""
    logger.info("Reflecting on the summary and generating a follow-up query")

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [
            SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
            HumanMessage(
                content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}"
            ),
        ]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get("follow_up_query")

    # JSON mode can fail in some cases
    if not query:
        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query["follow_up_query"]}


def finalize_summary(state: SummaryState) -> dict:
    """Finalize the summary"""
    logger.info("Finalizing the summary")

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}


def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """Route the research based on the follow-up query"""
    logger.info("Routing the research based on the follow-up query")

    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"Current research loop count: {state.research_loop_count}")
    logger.debug(f"Max research loops: {configurable.max_web_research_loops}")
    if state.research_loop_count < configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"


# Initialize the graph
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)

# Add nodes
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

# Compile the graph
graph = builder.compile()
