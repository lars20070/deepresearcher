#!/usr/bin/env python3

import os
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig

from deepresearcher.logger import logger


class SearchAPI(Enum):
    DUCKDUCKGO = "duckduckgo"
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""

    logger.info("Create configuration for the research assistant.")

    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")
    local_llm: str = os.environ.get("OLLAMA_MODEL", "llama3.3")
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))
    max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """
        Create a Configuration instance from a RunnableConfig.

        Take any standard LangChain RunnableConfig instance and extend it.
        """
        logger.info("Create configuration for the research assistant by extending a LangChain RunnableConfig.")
        # 'configurable' is a dictionary of configuration values from RunnableConfig
        # https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig.configurable
        configurable = config["configurable"] if config and "configurable" in config else {}

        values: dict[str, Any] = {}
        for field in fields(cls):
            if not field.init:
                continue

            env_var_value = os.environ.get(field.name.upper())
            config_value = configurable.get(field.name)

            if env_var_value is not None:
                values[field.name] = env_var_value
            else:
                values[field.name] = config_value

        # Ensure max_web_research_loops is an integer
        if "max_web_research_loops" in values and values["max_web_research_loops"]:
            values["max_web_research_loops"] = int(values["max_web_research_loops"])

        # Return a new Configuration instance with the above values
        return cls(**{k: v for k, v in values.items() if v})
