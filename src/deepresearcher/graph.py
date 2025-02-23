#!/usr/bin/env python3

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from deepresearcher.configuration import Configuration
from deepresearcher.logger import logger
from deepresearcher.prompts import (
    query_writer_instructions,
)
from deepresearcher.state import SummaryState, SummaryStateInput, SummaryStateOutput


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


# def web_research(state: SummaryState, config: RunnableConfig):
#     """Gather information from the web"""

#     logger.info(f"Web research with search query: {state.search_query}")

#     # Configure
#     configurable = Configuration.from_runnable_config(config)

#     # Handle both cases for search_api:
#     # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
#     # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
#     if isinstance(configurable.search_api, str):
#         search_api = configurable.search_api
#     else:
#         search_api = configurable.search_api.value

#     # Search the web
#     if search_api == "tavily":
#         search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
#         search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
#     elif search_api == "perplexity":
#         search_results = perplexity_search(state.search_query, state.research_loop_count)
#         search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
#     else:
#         raise ValueError(f"Unsupported search API: {configurable.search_api}")

#     return {
#         "sources_gathered": [format_sources(search_results)],
#         "research_loop_count": state.research_loop_count + 1,
#         "web_research_results": [search_str],
#     }


# Initialize the graph
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)

# Add nodes
builder.add_node("generate_query", generate_query)

# Add edges
builder.add_edge(START, "generate_query")
# builder.add_edge("generate_query", "web_research")
builder.add_edge("generate_query", END)

# Compile the graph
graph = builder.compile()
