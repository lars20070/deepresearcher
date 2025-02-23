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
builder.add_edge("generate_query", END)

# Compile the graph
graph = builder.compile()
