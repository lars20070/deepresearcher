#!/usr/bin/env python3

import os

import pytest
from dotenv import load_dotenv

from deepresearcher.state import SearchQuery, Section, SectionState, SummaryState


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "paid: mark tests which require a paid API key")
    config.addinivalue_line("markers", "ollama: mark tests which require a local Ollama instance")


@pytest.fixture(autouse=True)
def skip_ollama_tests(request: pytest.FixtureRequest) -> None:
    """
    Skip tests marked with 'ollama' when running in CI environment.
    Run these tests only locally.
    """
    if request.node.get_closest_marker("ollama") and os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Tests requiring Ollama skipped in CI environment")


@pytest.fixture
def load_env() -> None:
    """Load environment variables."""
    load_dotenv()


@pytest.fixture
def topic() -> str:
    """Provide a research topic for unit testing."""
    return "syzygy"


@pytest.fixture
def summary_state(topic: str) -> dict:
    """Provide a summary state for unit testing."""
    state = SummaryState(
        research_topic=topic,
        search_query=f"Tell me about {topic}",
        web_research_results=[
            """Sources:\n\nSource Syzygy - (Earth Science) - Vocab, Definition, Explanations - Most relevant content \
            from source: Syzygy is an astronomical term that describes the alignment of three celestial bodies in a \
            straight line. This alignment is most commonly associated with the Earth, the Moon, and the Sun during \
            events such as eclipses and the occurrence of tides. Understanding syzygy is crucial in grasping how \
            gravitational forces affect tidal patterns on Earth and the visual phenomena observed during eclipses."""
        ],
        sources_gathered=[
            "* Syzygy - (Earth Science) - Vocab, Definition, Explanations - Fiveable : https://library.fiveable.me/key-terms/hs-earth-science/syzygy"
        ],
        research_loop_count=42,
        running_summary="Test summary content",
    )
    return state


@pytest.fixture
def section_state(topic: str) -> dict:
    """Provide a section state for unit testing."""
    section = Section(
        name="Core Concepts",
        description=f"Understanding the core concepts of {topic}",
        research=True,
        content="",
    )
    query_1 = SearchQuery(search_query="syzygy astronomy definition celestial alignment planets")
    query_2 = SearchQuery(search_query="syzygy astronomical phenomena effects tides orbital mechanics")
    source_str = """
    Sources:
    
    Source Syzygy - Definition & Detailed Explanation - Sentinel Mission:
    ===
    URL: https://sentinelmission.org/astronomical-objects-glossary/syzygy/
    ===
    Most relevant content from source: Syzygy - Definition & Detailed Explanation - Astronomical Objects Glossary -
    Sentinel Mission Syzygy – Definition & Detailed Explanation – Astronomical Objects Glossary Syzygy is a term used
    in astronomy to describe the alignment of three or more celestial bodies in a straight line. This phenomenon occurs
    when the Earth, Sun, and Moon are in a straight line, creating either a solar or lunar eclipse. Syzygy occurs when
    the gravitational forces between celestial bodies cause them to align in a straight line. Syzygy can have various
    effects on Earth, depending on the type of alignment that occurs. In addition to observing syzygy from Earth,
    astronomers also study the phenomenon from space using satellites and spacecraft.
    ===
    Full source content limited to 1000 tokens: Published Time: 2024-04-28T09:26:25+00:00
    Syzygy - Definition & Detailed Explanation - Astronomical Objects Glossary - Sentinel Mission
    Skip to content
    """
    section_1 = Section(
        name="Core Concepts",
        description=f"Understanding the core concepts of {topic}",
        research=True,
        content="""
        ## Understanding Core Concepts of Syzygy

        **Syzygy represents a fundamental astronomical alignment phenomenon where three or more celestial bodies form a
        precise straight-line configuration in space.** This alignment occurs most notably in the Earth-Sun-Moon system,
        creating conditions necessary for solar and lunar eclipses.
        
        Key characteristics of syzygy include:
        - Perfect or near-perfect linear alignment of celestial objects
        - Gravitational force amplification along the alignment axis
        - Temporary duration as objects continue their orbital paths
        - Observable effects like eclipses and tidal variations
        
        ### Sources
        # - Syzygy - Definition & Detailed Explanation : https://sentinelmission.org/astronomical-objects-glossary/syzygy/
        """,
    )
    section_2 = Section(
        name="Background Information",
        description=f"Exploring the historical context of {topic}",
        research=True,
        content="""
        ## Understanding the Historical Context of Syzygy

        **The historical significance of syzygy spans multiple cultures and historical periods, reflecting its importance
        in astronomical observations and interpretations.**
        """,
    )
    state = SectionState(
        section=section,
        search_iterations=1,
        search_queries=[query_1, query_2],
        source_str=source_str,
        report_sections_from_research="",
        completed_sections=[section_1, section_2],
    )
    return state
