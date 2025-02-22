#!/usr/bin/env python3

import operator
from dataclasses import dataclass, field
from typing import Annotated


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)
