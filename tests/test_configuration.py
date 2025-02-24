#!/usr/bin/env python3

import pytest
from langchain_core.runnables import Runnable, RunnableConfig

from deepresearcher.configuration import Configuration, SearchAPI
from deepresearcher.logger import logger


def test_search_api_values() -> None:
    assert SearchAPI.DUCKDUCKGO.value == "duckduckgo"
    assert SearchAPI.PERPLEXITY.value == "perplexity"
    assert SearchAPI.TAVILY.value == "tavily"


def test_configuration_defaults() -> None:
    """All configuration values should have default values."""
    config = Configuration()

    assert config.max_web_research_loops == 3
    assert config.local_llm == "llama3.3"
    assert config.search_api == SearchAPI.DUCKDUCKGO


def test_configuration_from_runnable_config() -> None:
    """Some configuration values should stem from the RunnableConfig."""
    runnable_config = RunnableConfig(configurable={"max_web_research_loops": 5, "local_llm": "alpaca"})
    config = Configuration.from_runnable_config(runnable_config)

    assert config.max_web_research_loops == 5  # RunnableConfig value overrides default value
    assert config.local_llm == "alpaca"  # RunnableConfig value overrides default value
    assert config.search_api == SearchAPI.DUCKDUCKGO  # Default value


# This test runs twice, once for each pair of the arguments.
# The first argument is the environment variable value, the second is the expected SearchAPI value in the Configuration class.
@pytest.mark.parametrize(
    "env_value, expected_config", [("duckduckgo", SearchAPI.DUCKDUCKGO), ("perplexity", SearchAPI.PERPLEXITY), ("tavily", SearchAPI.TAVILY)]
)
def test_configuration_from_env(monkeypatch: pytest.MonkeyPatch, env_value: str, expected_config: SearchAPI) -> None:
    """One configuration value should stem from the environment variable. The other two from the defaults."""

    logger.info(f"environmental variable SEARCH_API == {env_value}")
    monkeypatch.setenv("SEARCH_API", env_value)
    config = Configuration.from_runnable_config()

    assert config.max_web_research_loops == 3  # Default value
    assert config.local_llm == "llama3.3"  # Default value
    assert config.search_api == expected_config.value  # Environment variable value overrides default value.


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
