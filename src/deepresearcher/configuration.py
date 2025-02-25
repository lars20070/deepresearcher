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


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    logger.info("Create configuration for the research assistant.")

    ollama_base_url: str = Field(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/"))
    local_llm: str = Field(os.environ.get("OLLAMA_MODEL", "llama3.3"))
    search_api: SearchAPI = Field(SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value)))
    max_web_research_loops: int = Field(int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3")))

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """
        Create a Configuration instance from a RunnableConfig.
        Values are read first from environment variables (using the uppercased field name) and then from a configurable dictionary.
        """
        logger.info("Create configuration for the research assistant by extending a LangChain RunnableConfig.")
        configurable: dict[str, Any] = config["configurable"] if config and "configurable" in config else {}

        values: dict[str, Any] = {}
        # Loop over all fields defined on the model
        for field_name in cls.model_fields:
            env_var_value = os.environ.get(field_name.upper())
            config_value = configurable.get(field_name)
            if env_var_value is not None:
                values[field_name] = env_var_value
            else:
                values[field_name] = config_value

        # Cast all fields to their expected types
        for field_name, model_field in cls.model_fields.items():
            if field_name in values and values[field_name] is not None:
                expected_type = model_field.annotation  # use the declared annotation
                try:
                    values[field_name] = expected_type(values[field_name])
                except Exception as e:
                    logger.error(f"Error casting field '{field_name}' to {expected_type}: {e}")

        # Pass only non-None values to override defaults
        filtered_values = {k: v for k, v in values.items() if v is not None}
        return cls(**filtered_values)
