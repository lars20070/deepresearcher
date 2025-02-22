#!/usr/bin/env python3

from deepresearcher.state import SummaryState, SummaryStateInput, SummaryStateOutput


def test_summary_state_defaults() -> None:
    state = SummaryState()
    assert state.research_topic is None
    assert state.web_research_results == []


def test_summary_state_custom_values() -> None:
    state = SummaryState(research_topic="AI", research_loop_count=2)
    assert state.research_topic == "AI"
    assert state.research_loop_count == 2


def test_summary_state_input_defaults() -> None:
    input_state = SummaryStateInput()
    assert input_state.research_topic is None


def test_summary_state_output_defaults() -> None:
    output_state = SummaryStateOutput()
    assert output_state.running_summary is None
