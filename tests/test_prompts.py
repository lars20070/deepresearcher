#!/usr/bin/env python3

from deepresearcher.logger import logger
from deepresearcher.prompts import (
    query_writer_instructions,
    reflection_instructions,
    summarizer_instructions,
)


def test_queries() -> None:
    logger.info("Testing all query templates")
    topic = "syzygy"

    formatted = query_writer_instructions.format(research_topic=topic)
    assert topic in formatted

    formatted = summarizer_instructions.format()
    assert "When creating a NEW summary" in formatted

    formatted = reflection_instructions.format(research_topic=topic)
    assert topic in formatted
