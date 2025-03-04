#!/usr/bin/env python3

from deepresearcher import logger
from deepresearcher.state import (
    Feedback,
    Queries,
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    SearchQuery,
    Section,
    SectionOutputState,
    Sections,
    SectionState,
    SummaryState,
    SummaryStateInput,
    SummaryStateOutput,
)


def test_summary_state_defaults() -> None:
    logger.info("Testing SummaryState defaults.")
    state = SummaryState()
    assert state.research_topic is None
    assert state.web_research_results == []


def test_summary_state_custom_values(topic: str) -> None:
    logger.info("Testing SummaryState with custom values.")
    state = SummaryState(research_topic=topic, research_loop_count=2)
    assert state.research_topic == topic
    assert state.research_loop_count == 2


def test_summary_state_input_defaults() -> None:
    logger.info("Testing SummaryStateInput defaults.")
    input_state = SummaryStateInput()
    assert input_state.research_topic is None


def test_summary_state_output_defaults() -> None:
    logger.info("Testing SummaryStateOutput defaults.")
    output_state = SummaryStateOutput()
    assert output_state.running_summary is None


def test_section() -> None:
    logger.info("Testing Section.")
    section = Section(name="Introduction", description="Overview", research=True, content="Content")
    assert section.name == "Introduction"
    assert section.description == "Overview"
    assert section.research is True
    assert section.content == "Content"


def test_sections() -> None:
    logger.info("Testing Sections.")
    section = Section(name="Body", description="Details", research=False, content="More content")
    sections = Sections(sections=[section])
    assert len(sections.sections) == 1
    assert sections.sections[0].name == "Body"


def test_search_query() -> None:
    logger.info("Testing SearchQuery.")
    sq = SearchQuery(search_query="test query")
    assert sq.search_query == "test query"


def test_queries() -> None:
    logger.info("Testing Queries.")
    sq1 = SearchQuery(search_query="query1")
    sq2 = SearchQuery(search_query="query2")
    queries = Queries(queries=[sq1, sq2])
    assert len(queries.queries) == 2
    assert queries.queries[0].search_query == "query1"


def test_feedback() -> None:
    logger.info("Testing Feedback.")
    fb = Feedback(grade="pass", follow_up_queries=[SearchQuery(search_query="follow up")])
    assert fb.grade == "pass"
    assert fb.follow_up_queries[0].search_query == "follow up"


def test_report_state_input() -> None:
    logger.info("Testing ReportStateInput.")
    report_input = ReportStateInput(topic="Test topic")
    assert report_input.topic == "Test topic"


def test_report_state_output() -> None:
    logger.info("Testing ReportStateOutput.")
    report_output = ReportStateOutput(final_report="Final report content")
    assert report_output.final_report == "Final report content"


def test_report_state() -> None:
    logger.info("Testing ReportState.")
    state: ReportState = {
        "topic": "Test topic",
        "feedback_on_report_plan": "Looks good",
        "sections": [Section(name="Sec1", description="Desc", research=True, content="Content")],
        "completed_sections": [],
        "report_sections_from_research": "Research content",
        "final_report": "Finalized report",
    }
    assert state["topic"] == "Test topic"
    assert state["feedback_on_report_plan"] == "Looks good"
    assert len(state["sections"]) == 1
    assert state["final_report"] == "Finalized report"


def test_section_state() -> None:
    logger.info("Testing SectionState.")
    section = Section(
        name="Conclusion",
        description="Summary",
        research=False,
        content="End content",
    )
    state = SectionState(
        section=section,
        search_iterations=2,
        search_queries=[SearchQuery(search_query="q1")],
        source_str="Formatted sources",
        report_sections_from_research="Research sections",
        completed_sections=[section],
    )
    assert state.section.name == "Conclusion"
    assert state.search_iterations == 2
    assert state.source_str == "Formatted sources"
    assert len(state.completed_sections) == 1


def test_section_output_state() -> None:
    logger.info("Testing SectionOutputState.")
    state = SectionOutputState(completed_sections=[Section(name="Intro", description="Desc", research=True, content="Content")])
    assert len(state.completed_sections) == 1
    assert state.completed_sections[0].name == "Intro"
