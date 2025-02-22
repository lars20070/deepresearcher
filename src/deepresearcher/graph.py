#!/usr/bin/env python3

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deepresearcher.configuration import Configuration
from deepresearcher.state import SummaryState, SummaryStateInput, SummaryStateOutput


# Define nodes
def generate_query(state: SummaryState, config: RunnableConfig) -> dict:
    """Generate a query for web search"""

    # logger.info(f"Generating query for the research topic: {state.research_topic}")

    # # Format the prompt
    # query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # logger.info(f"Formatted prompt: {query_writer_instructions_formatted}")

    # # Generate a query
    # configurable = Configuration.from_runnable_config(config)
    # llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")
    # logger.info("LLM model created.")
    # result = llm_json_mode.invoke(
    #     [
    #         SystemMessage(content=query_writer_instructions_formatted),
    #         HumanMessage(content=f"Generate a query for web search:"),
    #     ]
    # )
    # logger.info(f"LLM response: {result.content}")
    # query = json.loads(result.content)

    # return {"search_query": query["query"]}
    return {"search_query": "dummy"}


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

graph = builder.compile()
