#!/usr/bin/env python3

import json
import os
import re
from typing import Literal

import pypandoc
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from deepresearcher.configuration import Configuration, ConfigurationReport
from deepresearcher.logger import logger
from deepresearcher.prompts import (
    final_section_writer_instructions,
    query_writer_instructions,
    query_writer_instructions_2,
    reflection_instructions,
    report_planner_instructions,
    report_planner_query_writer_instructions,
    section_grader_instructions,
    section_writer_instructions,
    summarizer_instructions,
)
from deepresearcher.state import (
    Feedback,
    Queries,
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    SectionOutputState,
    Sections,
    SectionState,
    SummaryState,
    SummaryStateInput,
    SummaryStateOutput,
)
from deepresearcher.utils import (
    deduplicate_and_format_sources,
    duckduckgo_search,
    format_sections,
    format_sources,
    get_config_value,
    perplexity_search,
    perplexity_search_2,
    retry_with_backoff,
    tavily_search,
    tavily_search_async,
)

#########################################################################
#
# Define the graph for the simple deep researcher assistant (no HITL)
#
#########################################################################


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
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=False)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
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

#########################################################################
#
# Define the graph for the report deep researcher assistant (with HITL)
#
#########################################################################


def _normalize_state(state: ReportState | SectionState | dict, state_class: type[BaseModel]) -> BaseModel:
    """
    Cast `state` object to pydantic class.

    LangGraph is passing `dict` objects between nodes. But all state objects are pydantic.
    That makes type coercion at runtime at the beginning of each node method necessary.
    """
    if isinstance(state, dict):
        return state_class(**state)
    return state


def _generate_queries(provider: str, model: str, instructions: str) -> list:
    """
    Generate queries
    """

    writer_model = init_chat_model(
        model_provider=provider,
        model=model,
        temperature=0,
    )
    structured_llm = writer_model.with_structured_output(Queries)

    # Generate queries
    try:
        results = structured_llm.invoke(
            [SystemMessage(content=instructions)]
            + [HumanMessage(content="Generate search queries that will help with planning the sections of the report.")]
        )
    except Exception as e:
        logger.error(f"Error from structured LLM: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "json"):
            logger.error(f"Error details: {e.response.json()}")
        raise
    logger.debug(f"Queries generated:\n{results.queries}")

    # Web search
    query_list = [query.search_query for query in results.queries]

    return query_list


# Define nodes
@retry_with_backoff
async def generate_report_plan(state: ReportState | dict, config: RunnableConfig) -> dict:
    state = _normalize_state(state, ReportState)

    logger.info(f"Generating the report plan for the topic: {state.topic}")

    # Inputs
    topic = state.topic
    feedback = state.feedback_on_report_plan

    # Get configuration
    configurable = ConfigurationReport.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    logger.debug(f"Complete configurable object: {configurable}")

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic, report_organization=report_structure, number_of_queries=number_of_queries
    )
    logger.debug(f"System instructions:\n{system_instructions_query}")

    # Set writer model (model used for query writing and section writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    logger.debug(f"Writer provider: {writer_provider}")
    logger.debug(f"Writer model: {writer_model_name}")

    query_list = _generate_queries(
        provider=writer_provider,
        model=writer_model_name,
        instructions=system_instructions_query,
    )

    # Get the search API
    search_api = get_config_value(configurable.search_api)
    logger.debug(f"Search API: {search_api}")

    # Search the web
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "perplexity":
        search_results = perplexity_search_2(query_list)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "duckduckgo":
        logger.info("Searching with DuckDuckGo")
        search_results = []
        for query in query_list:
            result = duckduckgo_search(query, max_results=3, fetch_full_page=True)
            search_results.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": result["results"],
                }
            )
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")
    logger.debug(f"Search results:\n{source_str}")

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(
        topic=topic, report_organization=report_structure, context=source_str, feedback=feedback
    )

    # Set the planner provider
    if isinstance(configurable.planner_provider, str):
        planner_provider = configurable.planner_provider
    else:
        planner_provider = configurable.planner_provider.value

    # Set the planner model
    if isinstance(configurable.planner_model, str):
        planner_model = configurable.planner_model
    else:
        planner_model = configurable.planner_model.value
    logger.debug(f"Planner provider: {planner_provider}")
    logger.debug(f"Planner model: {planner_model}")

    # Set the planner model
    planner_llm = init_chat_model(model=planner_model, model_provider=planner_provider)

    # Generate sections
    structured_llm = planner_llm.with_structured_output(Sections)
    try:
        report_sections = structured_llm.invoke(
            [SystemMessage(content=system_instructions_sections)]
            + [
                HumanMessage(
                    content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. \
                        Each section must have: name, description, plan, research, and content fields."
                )
            ]
        )
    except Exception as e:
        logger.error(f"Error from structured LLM: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "json"):
            logger.error(f"Error details: {e.response.json()}")
        raise

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}


def human_feedback(state: ReportState | dict, config: RunnableConfig) -> Command[Literal["generate_report_plan", "build_section_with_web_research"]]:
    state = _normalize_state(state, ReportState)

    logger.info("Getting human feedback on the report plan")

    # Get sections
    sections = state.sections
    sections_str = "\n\n".join(
        f"Section: {section.name}\nDescription: {section.description}\nResearch needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    feedback = interrupt(
        f"""Please provide feedback on the following report plan.
        
        {sections_str}
        
        Does the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan:"""
    )

    # If the user approves the report plan, kick off section writing
    # if isinstance(feedback, bool) and feedback is True:
    if isinstance(feedback, bool):
        # Treat this as approve and kick off section writing
        return Command(goto=[Send("build_section_with_web_research", {"section": s, "search_iterations": 0}) for s in sections if s.research])

    # If the user provides feedback, regenerate the report plan
    elif isinstance(feedback, str):
        # treat this as feedback
        return Command(goto="generate_report_plan", update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


@retry_with_backoff
def generate_queries(state: SectionState | dict, config: RunnableConfig) -> dict:
    state = _normalize_state(state, SectionState)

    logger.info(f"Generating search queries for the section: {state.section.name}")

    # Get state
    section = state.section

    # Get configuration
    configurable = ConfigurationReport.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, temperature=0)
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions_2.format(section_topic=section.description, number_of_queries=number_of_queries)

    # Generate queries
    queries = structured_llm.invoke(
        [SystemMessage(content=system_instructions)] + [HumanMessage(content="Generate search queries on the provided topic.")]
    )

    return {"search_queries": queries.queries}


@retry_with_backoff
async def search_web(state: SectionState | dict, config: RunnableConfig) -> dict:
    """Search the web for each query, then return a list of raw sources and a formatted string of sources."""
    state = _normalize_state(state, SectionState)

    logger.info("Searching the web for each query")

    # Get state
    search_queries = state.search_queries

    # Get configuration
    configurable = ConfigurationReport.from_runnable_config(config)

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # Search the web
    # TODO: max_tokens_per_source reduced from 5000 to 1000 due to Anthropic rate limiting. Find workaround.
    if search_api == "tavily":
        logger.info("Searching with Tavily")
        search_results = await tavily_search_async(query_list)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    elif search_api == "perplexity":
        logger.info("Searching with Perplexity")
        search_results = perplexity_search_2(query_list)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "duckduckgo":
        logger.info("Searching with DuckDuckGo")
        search_results = []
        for query in query_list:
            # Get results for each query
            result = duckduckgo_search(query, max_results=3, fetch_full_page=True)
            # Add the query information to match expected format
            search_results.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": result["results"],
                }
            )
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {"source_str": source_str, "search_iterations": state.search_iterations + 1}


@retry_with_backoff
def write_section(state: SectionState | dict, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report"""
    state = _normalize_state(state, SectionState)

    logger.info(f"Writing the section: {state.section.name}")

    # Get state
    section = state.section
    source_str = state.source_str

    # Get configuration
    configurable = ConfigurationReport.from_runnable_config(config)

    # Format system instructions
    system_instructions = section_writer_instructions.format(
        section_title=section.name, section_topic=section.description, context=source_str, section_content=section.content
    )

    # Generate section
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, temperature=0)
    section_content = writer_model.invoke(
        [SystemMessage(content=system_instructions)] + [HumanMessage(content="Generate a report section based on the provided sources.")]
    )

    # Write content to the section object
    section.content = section_content.content

    # Grade prompt
    section_grader_instructions_formatted = section_grader_instructions.format(section_topic=section.description, section=section.content)

    # Feedback
    structured_llm = writer_model.with_structured_output(Feedback)
    feedback = structured_llm.invoke(
        [SystemMessage(content=section_grader_instructions_formatted)]
        + [HumanMessage(content="Grade the report and consider follow-up questions for missing information:")]
    )

    if feedback.grade == "pass" or state.search_iterations >= configurable.max_search_depth:
        # Publish the section to completed sections
        return Command(update={"completed_sections": [section]}, goto=END)
    else:
        # Update the existing section with new content and update search queries
        return Command(update={"search_queries": feedback.follow_up_queries, "section": section}, goto="search_web")


def gather_completed_sections(state: ReportState | dict) -> dict:
    """Gather completed sections from research and format them as context for writing the final sections"""
    state = _normalize_state(state, ReportState)

    logger.info("Gathering completed sections from research")

    # List of completed sections
    completed_sections = state.completed_sections

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}


@retry_with_backoff
def write_final_sections(state: SectionState | dict, config: RunnableConfig) -> dict:
    """Write final sections of the report, which do not require web search and use the completed sections as context"""
    state = _normalize_state(state, SectionState)

    logger.info("Writing final sections of the report")

    # Get configuration
    configurable = ConfigurationReport.from_runnable_config(config)

    # Get sections
    section = state.section
    completed_report_sections = state.report_sections_from_research

    # Format system instructions
    system_instructions = final_section_writer_instructions.format(
        section_title=section.name, section_topic=section.description, context=completed_report_sections
    )

    # Generate section
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, temperature=0)
    section_content = writer_model.invoke(
        [SystemMessage(content=system_instructions)] + [HumanMessage(content="Generate a report section based on the provided sources.")]
    )

    # Write content to section
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


def initiate_final_section_writing(state: ReportState | dict) -> Command[Literal[END, "write_final_sections"]]:
    """Write any final sections using the Send API to parallelize the process"""
    state = _normalize_state(state, ReportState)

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"section": s, "report_sections_from_research": state.report_sections_from_research})
        for s in state.sections
        if not s.research
    ]


def compile_final_report(state: ReportState | dict, config: RunnableConfig) -> dict:
    """Compile the final report"""
    state = _normalize_state(state, ReportState)

    # Get configuration
    configurable = ConfigurationReport.from_runnable_config(config)

    logger.info("Compiling the final report")

    # Get sections
    sections = state.sections
    completed_sections = {s.name: s.content for s in state.completed_sections}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    # Export the final report to output directory
    if os.path.exists(configurable.output_dir):
        logger.info(f"Writing the final report as markdown to '{configurable.output_dir}'")
        file_name = re.sub(r"[^a-zA-Z0-9]", "_", state.topic).lower()
        path_md = os.path.join(configurable.output_dir, f"{file_name}.md")
        with open(path_md, "w", encoding="utf-8") as f:
            f.write(all_sections)

        # Convert markdown to PDF using Pandoc
        try:
            logger.info(f"Writing the final report as PDF to '{configurable.output_dir}'")
            logger.debug(f"Pandoc version {pypandoc.get_pandoc_version()} is installed at path: '{pypandoc.get_pandoc_path()}'")
            path_pdf = os.path.join(configurable.output_dir, f"{file_name}.pdf")
            pypandoc.convert_file(
                path_md,
                "pdf",
                outputfile=path_pdf,
                extra_args=[
                    "--pdf-engine=xelatex",
                    "-V",
                    "colorlinks=true",
                    "-V",
                    "linkcolor=blue",  # Internal links
                    "-V",
                    "urlcolor=blue",  # External links
                    "-V",
                    "citecolor=blue",
                    "--from",
                    "markdown+autolink_bare_uris",  # Ensures bare URLs are also hyperlinked
                ],
            )
        except Exception:
            logger.error("Pandoc is not installed. Skipping conversion to PDF.")
    else:
        logger.error(f"Output directory {configurable.output_dir} does not exist. Skipping writing the final report.")

    return {"final_report": all_sections}


# Initialize the graph
builder_report = StateGraph(
    ReportState,
    input=ReportStateInput,
    output=ReportStateOutput,
    config_schema=ConfigurationReport,
)

# Report section sub-graph --

# Add nodes
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Add nodes
builder_report.add_node("generate_report_plan", generate_report_plan)
builder_report.add_node("human_feedback", human_feedback)
builder_report.add_node("build_section_with_web_research", section_builder.compile())
builder_report.add_node("gather_completed_sections", gather_completed_sections)
builder_report.add_node("write_final_sections", write_final_sections)
builder_report.add_node("compile_final_report", compile_final_report)

# Outer graph --

# Add edges
builder_report.add_edge(START, "generate_report_plan")
builder_report.add_edge("generate_report_plan", "human_feedback")
builder_report.add_edge("build_section_with_web_research", "gather_completed_sections")
builder_report.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder_report.add_edge("write_final_sections", "compile_final_report")
builder_report.add_edge("compile_final_report", END)

graph_report = builder_report.compile()
