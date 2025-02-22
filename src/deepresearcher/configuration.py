#!/usr/bin/env python3

import os
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = 3
    local_llm: str = "llama3.2"
    search_api: SearchAPI = SearchAPI.TAVILY  # Default to TAVILY

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {f.name: os.environ.get(f.name.upper(), configurable.get(f.name)) for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in values.items() if v})
