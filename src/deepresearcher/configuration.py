#!/usr/bin/env python3

import os
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from deepresearcher.logger import logger


class SearchAPI(Enum):
    DUCKDUCKGO = "duckduckgo"
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


class PlannerProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"


class WriterProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""


class ConfigurationMixin:
    """Mixin class shared between all Configuration classes."""

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """
        Create a Configuration instance from a RunnableConfig.
        Values are read first from environment variables (using the uppercased field name) and then from a configurable dictionary.
        """
        logger.info("Create configuration for the research assistant by extending a LangChain RunnableConfig.")
        configurable: dict[str, Any] = config["configurable"] if config and "configurable" in config else {}

        values: dict[str, Any] = {}
        # Loop over all fields defined on the model and cast them immediately
        for field_name, model_field in cls.model_fields.items():
            # Choose the value from the environment or the configurable dictionary
            raw_value = os.environ.get(field_name.upper(), configurable.get(field_name))
            if raw_value is not None:
                expected_type = model_field.annotation  # declared type
                try:
                    values[field_name] = expected_type(raw_value)
                except Exception as e:
                    logger.error(f"Error casting field '{field_name}' to {expected_type}: {e}")
            # If the value is None, skip it (the default defined on the model will be used)

        # Pass only non-None values to override defaults
        filtered_values = {k: v for k, v in values.items() if v is not None}
        return cls(**filtered_values)


class Configuration(BaseModel, ConfigurationMixin):
    """The configurable fields for the research assistant."""

    logger.info("Create configuration for the research assistant.")

    ollama_base_url: str = Field(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/"))
    local_llm: str = Field(os.environ.get("OLLAMA_MODEL", "llama3.3"))
    search_api: SearchAPI = Field(SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value)))
    max_web_research_loops: int = Field(int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3")))


class ConfigurationReport(BaseModel, ConfigurationMixin):
    """The configurable fields for the chatbot."""

    report_structure: str = Field(DEFAULT_REPORT_STRUCTURE)  # Defaults to the default report structure
    number_of_queries: int = Field(2)  # Number of search queries to generate per iteration
    max_search_depth: int = Field(2)  # Maximum number of reflection + search iterations
    planner_provider: PlannerProvider = Field(PlannerProvider.OPENAI)  # Defaults to OpenAI as provider
    planner_model: str = Field("o3-mini")  # Defaults to OpenAI o3-mini as planner model
    writer_provider: WriterProvider = Field(WriterProvider.ANTHROPIC)  # Defaults to Anthropic as provider
    writer_model: str = Field("claude-3-5-sonnet-latest")  # Defaults to Anthropic as provider
    search_api: SearchAPI = Field(SearchAPI.TAVILY)  # Default to TAVILY
