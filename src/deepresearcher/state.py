#!/usr/bin/env python3


from pydantic import BaseModel, Field


class SummaryState(BaseModel):
    research_topic: str = Field(None)
    search_query: str = Field(None)
    web_research_results: list = Field(default_factory=list)
    sources_gathered: list = Field(default_factory=list)
    research_loop_count: int = Field(0)
    running_summary: str = Field(None)


class SummaryStateInput(BaseModel):
    research_topic: str = Field(None)


class SummaryStateOutput(BaseModel):
    running_summary: str = Field(None)
