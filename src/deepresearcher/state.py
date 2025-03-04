#!/usr/bin/env python3

from typing import Literal

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


class ReportStateInput(BaseModel):
    topic: str = Field(None, description="Report topic")


class ReportStateOutput(BaseModel):
    final_report: str = Field(None, description="Finalized report")


class ReportState(BaseModel):
    topic: str = Field(None, description="Report topic")
    feedback_on_report_plan: str = Field(None, description="Feedback on the report plan")
    sections: list[Section] = Field(default_factory=list, description="List of report sections")
    completed_sections: list[Section] = Field(default_factory=list, description="Completed sections of the report")
    report_sections_from_research: str = Field(None, description="String of any completed sections from research to write final sections")
    final_report: str = Field(None, description="Final report")


class SectionState(BaseModel):
    section: Section = Field(None, description="Report section")
    search_iterations: int = Field(None, description="Number of search iterations done")
    search_queries: list[SearchQuery] = Field(default_factory=list, description="List of search queries")
    source_str: str = Field(None, description="String of formatted source content from web search")
    report_sections_from_research: str = Field(None, description="String of any completed sections from research to write final sections")
    completed_sections: list[Section] = Field(default_factory=list, description="Final key we duplicate in outer state for Send() API")


class SectionOutputState(BaseModel):
    completed_sections: list[Section] = Field(default_factory=list, description="Final key we duplicate in outer state for Send() API")
