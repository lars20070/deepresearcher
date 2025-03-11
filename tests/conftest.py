#!/usr/bin/env python3

import os

import pytest
from dotenv import load_dotenv

from deepresearcher.state import ReportState, SearchQuery, Section, SectionState, SummaryState


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
    report_sections_from_research = """
    ============================================================
    Section 1: Core Concepts
    ============================================================
    Description:
    Understanding the core concepts of syzygy
    Requires Research: True
    
    Content:
    
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
    - Syzygy - Definition & Detailed Explanation : https://sentinelmission.org/astronomical-objects-glossary/syzygy/

    ============================================================
    Section 2: Background Information
    ============================================================
    Description:
    Exploring the historical context of syzygy
    Requires Research: True

    Content:
    


    **The historical significance of syzygy spans multiple cultures and historical periods, reflecting its importance
    in astronomical observations and interpretations.**
    """
    state = SectionState(
        section=section,
        search_iterations=1,
        search_queries=[query_1, query_2],
        source_str=source_str,
        report_sections_from_research=report_sections_from_research,
        completed_sections=[section_1, section_2],
    )
    return state


@pytest.fixture
def report_state(topic: str) -> dict:
    """Provide a report state for unit testing."""
    section_1 = Section(
        name="Introduction",
        description="Overview of the report",
        research=False,
        content="",
    )
    section_2 = Section(
        name="Methodology",
        description="Research methods used",
        research=False,
        content="",
    )
    section_3 = Section(
        name="Conclusion",
        description="Summary of the report",
        research=False,
        content="",
    )
    state = ReportState(
        topic=topic,
        feedback_on_report_plan="Looks good",
        sections=[section_1, section_2, section_3],
        completed_sections=[section_1, section_2, section_3],
        report_sections_from_research="Research content",
        final_report="Finalized report",
    )
    return state


@pytest.fixture
def search_result() -> str:
    """
    Web search result
    """
    return """
        Sources:

        Source Syzygy in Astronomy - Definition, Pronunciation, Examples:
        ===
        URL: https://sciencenotes.org/syzygy-in-astronomy-definition-pronunciation-examples/
        ===
        Most relevant content from source: Home » Science Notes Posts » Astronomy » Syzygy in Astronomy – Definition, Pronunciation, Examples This alignment often involves the Earth, the Moon, and the Sun but can apply to any celestial bodies. Syzygy underlies many celestial phenomena, particularly those involving the Sun, Moon, and Earth. Transit: The famous transit of Venus, which occurs approximately every 105 or 121 years, is an example of syzygy involving the Sun, Venus, and Earth. Occultation: When the Moon occults a star, it’s a temporary but fascinating display of syzygy, where celestial bodies align to block one from view. Eclipses are direct consequences of syzygy involving the Earth, Sun, and Moon. This alignment only occurs during a full moon when the three bodies are again in syzygy.
        ===
        Source Syzygy (astronomy) - Definition & Detailed Explanation - Sentinel Mission:
        ===
        URL: https://sentinelmission.org/astronomical-phenomena-glossary/syzygy-astronomy/
        ===
        Most relevant content from source: Syzygy (astronomy) - Definition & Detailed Explanation - Astronomical Phenomena Glossary - Sentinel Mission Syzygy (astronomy) – Definition & Detailed Explanation – Astronomical Phenomena Glossary This alignment is known as a solar or lunar syzygy, depending on whether the Moon is between the Earth and the Sun (new moon) or the Earth is between the Moon and the Sun (full moon). One notable example of syzygy is the alignment of the Sun, Earth, and Moon during a solar or lunar eclipse. Planetary syzygies can occur when two or more planets align with the Sun from the perspective of Earth, creating unique opportunities for astronomers to study the orbital dynamics and gravitational interactions of the planets.
        ===
        Source Syzygy - Definition & Detailed Explanation - Sentinel Mission:
        ===
        URL: https://sentinelmission.org/astronomical-objects-glossary/syzygy/
        ===
        Most relevant content from source: Syzygy - Definition & Detailed Explanation - Astronomical Objects Glossary - Sentinel Mission Syzygy – Definition & Detailed Explanation – Astronomical Objects Glossary Syzygy is a term used in astronomy to describe the alignment of three or more celestial bodies in a straight line. This phenomenon occurs when the Earth, Sun, and Moon are in a straight line, creating either a solar or lunar eclipse. Syzygy occurs when the gravitational forces between celestial bodies cause them to align in a straight line. Syzygy can have various effects on Earth, depending on the type of alignment that occurs. In addition to observing syzygy from Earth, astronomers also study the phenomenon from space using satellites and spacecraft.
        ===
        Source Syzygy - Definition & Detailed Explanation - Sentinel Mission:
        ===
        URL: https://sentinelmission.org/astronomical-phenomena-glossary/syzygy-2/
        ===
        Most relevant content from source: Syzygy - Definition & Detailed Explanation - Astronomical Phenomena Glossary - Sentinel Mission Syzygy – Definition & Detailed Explanation – Astronomical Phenomena Glossary Syzygy occurs when three celestial bodies align in a straight line, with the Earth being in the middle. For example, a syzygy can occur when the Sun, Earth, and Moon align during a solar or lunar eclipse. 1. Solar Syzygy: This occurs when the Sun, Earth, and another celestial body align in a straight line. Lunar Syzygy: This occurs when the Sun, Earth, and Moon align in a straight line. 3. Planetary Syzygy: This occurs when two or more planets align with the Earth in between them. 1. Solar Eclipses: Solar eclipses occur when the Moon passes between the Earth and the Sun, creating a solar syzygy.
        ===
        Source Syzygy (astronomy) - Wikipedia:
        ===
        URL: https://en.wikipedia.org/wiki/Syzygy_(astronomy)
        ===
        Most relevant content from source: Solar and lunar eclipses occur at times of syzygy, as do transits and occultations. An eclipse occurs when a body totally or partially disappears from view, either by an occultation, as with a solar eclipse, or by passing into the shadow of another body, as with a lunar eclipse (thus both are listed on NASA's eclipse page). At the new and full moon, the Sun and Moon are in syzygy. On June 3, 2014, the Curiosity rover on Mars observed the planet Mercury transiting the Sun, marking the first time a planetary transit has been observed from a celestial body besides Earth.[7] Moon Full moon Lunar eclipse Solar eclipses on the Moon
        ===
        Source Syzygy and the Language of Science | The American Biology Teacher ...:
        ===
        URL: https://online.ucpress.edu/abt/article/86/1/47/199004/Syzygy-and-the-Language-of-Science
        ===
        Most relevant content from source: This occurs, for example, during solar and lunar eclipses, when the sun, Earth, and moon are all aligned. "Syzygy" is also a Japanese band, an episode of The X-files, and the name of the Carleton College Division I Women's ultimate team. And it has other meanings in fields as diverse as poetry and mathematics.
        ===
        Source Syzygy in astronomy, biology, math, and Jung - johndcook.com:
        ===
        URL: https://www.johndcook.com/blog/2013/07/15/syzygy/
        ===
        Most relevant content from source: Syzygy in astronomy, biology, math, and Jung Posted on 15 July 2013 by John Syzygy must be really valuable in some word games. I’ve run into the word syzygy in diverse contexts and wondered what the meanings had in common. In the game that Selchow and Righter used to insist must always be called “Scrabble(R) Brand Crossword Game”, the word syzygy is quite rare, because it requires the single Z in the set of tiles, both Y’s, and one of the two blanks (for the other Y), in addition to one of the 4 S’s and one of the 3 G’s. It could be a good Hangman word, perhaps. My colleagues and I have decades of consulting experience helping companies solve complex problems involving data privacy, applied math, and statistics.
        ===
        Source SYZYGY Definition & Meaning - Merriam-Webster:
        ===
        URL: https://www.merriam-webster.com/dictionary/syzygy
        ===
        Most relevant content from source: Syzygy Definition & Meaning - Merriam-Webster Word of the Day Word of the Day My Words Save Word   syzygy Examples of syzygy in a Sentence —Lacy Schley, Discover Magazine, 30 May 2017 As reported by Old Farmer’s Almanac, the Sturgeon supermoon will reach its peak—otherwise known as a syzygy, when the moon is almost directly between the Sun and the Earth in a straight line configuration—at 9:36 p.m. EST on Thursday, August 11. The first known use of syzygy was circa 1847 See more words from the same year Dictionary Entries Near syzygy syzygy Post the Definition of syzygy to Facebook  Facebook Share the Definition of syzygy on Twitter  Twitter More from Merriam-Webster on syzygy ### Can you solve 4 words at once? Word of the Day ### 15 Words That Used to Mean Something Different
        ===
        Source Uncommon Knowledge: What Do Poetry and Planets Have in Common?:
        ===
        URL: https://www.uncommongoods.com/blog/2016/uncommon-knowledge-poetry-planets-common/
        ===
        Most relevant content from source: Uncommon Knowledge: What Do Poetry and Planets Have in Common? Uncommon Knowledge Uncommon Knowledge: What Do Poetry and Planets Have in Common? The shortest word to incorporate three y’s, it’s also a term shared by poetry and astronomy. But to understand why poetry and astronomy share this weird word, we need to look at its etymological root. Eric is a copywriter for UncommonGoods. Save my name, email, and website in this browser for the next time I comment. 50 Last Minute Father’s Day Gifts for Unique Dads 100 Thoughtful Father’s Day Gifts for Unique Dads When you visit our blog, you’ll meet artists, discover uncommon knowledge, immerse yourself in creative design, and get to know the people who keep UncommonGoods going strong.
        ===
        """
