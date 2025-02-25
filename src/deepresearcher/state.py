#!/usr/bin/env python3


from pydantic import BaseModel, Field


class SummaryState(BaseModel):
    research_topic: str = Field(None, description="research topic")
    search_query: str = Field(None, description="search query")
    web_research_results: list = Field(default_factory=list)
    sources_gathered: list = Field(default_factory=list)
    research_loop_count: int = Field(0, description="research loop count")
    running_summary: str = Field(None, description="running summary")


class SummaryStateInput(BaseModel):
    research_topic: str = Field(None, description="reasearch topic")


class SummaryStateOutput(BaseModel):
    running_summary: str = Field(None, description="running summary")
