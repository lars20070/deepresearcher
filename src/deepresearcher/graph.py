#!/usr/bin/env python3

# from langgraph.graph import END, START, StateGraph

# from deepresearcher.state import SummaryState, SummaryStateInput, SummaryStateOutput

# # Initialize the graph
# builder = StateGraph(
#     SummaryState,
#     input=SummaryStateInput,
#     output=SummaryStateOutput,
#     config_schema=Configuration,
# )

# # Add nodes
# builder.add_node("generate_query", generate_query)

# # Add edges
# builder.add_edge(START, "generate_query")
# builder.add_edge("generate_query", END)

# graph = builder.compile()
