#!/usr/bin/env python3

# import os

# import pytest
from langchain_core.runnables import Runnable, RunnableConfig

from deepresearcher.configuration import Configuration, SearchAPI
from deepresearcher.logger import logger


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


def test_EXAMPLE_runnable() -> None:
    """
    Minimal example of the Runnable class
    """
    logger.info("Starting minimal example of Runnable class.")

    class EchoRunnable(Runnable):
        """Simple Runnable that echoes its input."""

        def invoke(self, input: str, config: RunnableConfig = None) -> str:
            # Accessing configuration attributes
            run_name = config["run_name"] if config and "run_name" in config else "default_run"
            logger.info(f"Run Name: {run_name}")
            return input

    # Create an instance of the Runnable class and its config
    echo_runnable = EchoRunnable()
    config = RunnableConfig(run_name="EchoRun", tags=["example", "echo"], metadata={"purpose": "demonstration"}, max_concurrency=5)

    # Invoke the Runnable without and with configuration
    result = echo_runnable.invoke("Invoke without config.")
    logger.info(f"Return value: {result}")
    result = echo_runnable.invoke("Invoke with config.", config=config)
    logger.info(f"Return value: {result}")
