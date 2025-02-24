#!/usr/bin/env python3

from deepresearcher import logger
from deepresearcher.state import SummaryState, SummaryStateInput, SummaryStateOutput


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
