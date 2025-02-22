#!/usr/bin/env python3

# import os

# import pytest
from langchain_core.runnables import RunnableConfig

from deepresearcher.configuration import Configuration, SearchAPI


def test_search_api_values() -> None:
    assert SearchAPI.PERPLEXITY.value == "perplexity"
    assert SearchAPI.TAVILY.value == "tavily"


def test_configuration_defaults() -> None:
    config = Configuration()
    assert config.max_web_research_loops == 3
    assert config.local_llm == "llama3.2"
    assert config.search_api == SearchAPI.TAVILY


def test_configuration_from_runnable_config() -> None:
    runnable_config = RunnableConfig(configurable={"max_web_research_loops": 5, "local_llm": "alpaca"})
    config = Configuration.from_runnable_config(runnable_config)
    assert config.max_web_research_loops == 5
    assert config.local_llm == "alpaca"


# @pytest.mark.parametrize("env_val, expected", [("perplexity", SearchAPI.PERPLEXITY), ("tavily", SearchAPI.TAVILY)])
# def test_configuration_from_env(env_val, expected) -> None:
#     os.environ["SEARCH_API"] = env_val
#     config = Configuration.from_runnable_config()
#     assert config.search_api == expected
#     del os.environ["SEARCH_API"]
