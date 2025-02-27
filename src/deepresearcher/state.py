#!/usr/bin/env python3

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

#################################################################
#
# States for the simple deep researcher assistant (no HITL)
#
#################################################################


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


#################################################################
#
# States for the report deep researcher assistant (with HITL)
#
#################################################################


class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(description="Whether to perform web research for this section of the report.")
    content: str = Field(description="The content of the section.")


class Sections(BaseModel):
    sections: list[Section] = Field(
        description="Sections of the report.",
    )


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: list[SearchQuery] = Field(
        description="List of search queries.",
    )


class Feedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: list[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )


# TODO: Convert TypeDicts to Pydantic models.
class ReportStateInput(TypedDict):
    topic: str  # Report topic


class ReportStateOutput(TypedDict):
    final_report: str  # Final report


class ReportState(TypedDict):
    topic: str  # Report topic
    feedback_on_report_plan: str  # Feedback on the report plan
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[list, operator.add]  # Send() API key
    report_sections_from_research: str  # String of any completed sections from research to write final sections
    final_report: str  # Final report


class SectionState(TypedDict):
    section: Section  # Report section
    search_iterations: int  # Number of search iterations done
    search_queries: list[SearchQuery]  # List of search queries
    source_str: str  # String of formatted source content from web search
    report_sections_from_research: str  # String of any completed sections from research to write final sections
    completed_sections: list[Section]  # Final key we duplicate in outer state for Send() API


class SectionOutputState(TypedDict):
    completed_sections: list[Section]  # Final key we duplicate in outer state for Send() API
