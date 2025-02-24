#!/usr/bin/env python3

import os
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig

from deepresearcher.logger import logger


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""

    logger.info("Create configuration for the research assistant.")

    max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
    local_llm: str = os.environ.get("OLLAMA_MODEL", "llama3.3")
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.TAVILY.value))
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")

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

        # Fill in the values of the Configuration instance
        # (1) from the environment variables, if they exist e.g. MAX_WEB_RESEARCH_LOOPS
        # (2) from the RunnableConfig, if they exist e.g. {"max_web_research_loops": 5, "local_llm": "alpaca"}
        # (3) from the default values of the Configuration class
        values: dict[str, Any] = {f.name: os.environ.get(f.name.upper(), configurable.get(f.name)) for f in fields(cls) if f.init}

        # Return a new Configuration instance with the above values
        return cls(**{k: v for k, v in values.items() if v})
