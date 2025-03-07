#!/usr/bin/env python3

import pytest
from dotenv import load_dotenv

from deepresearcher.state import SummaryState


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
