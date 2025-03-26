#!/usr/bin/env python3

import pytest
from langchain_core.runnables import Runnable, RunnableConfig

from deepresearcher.configuration import DEFAULT_REPORT_STRUCTURE, Configuration, ConfigurationReport, PlannerProvider, SearchAPI, WriterProvider
from deepresearcher.logger import logger


def test_search_api_values() -> None:
    assert SearchAPI.DUCKDUCKGO.value == "duckduckgo"
    assert SearchAPI.PERPLEXITY.value == "perplexity"
    assert SearchAPI.TAVILY.value == "tavily"


def test_planner_provider_values() -> None:
    assert PlannerProvider.OPENAI.value == "openai"
    assert PlannerProvider.GROQ.value == "groq"


def test_writer_provider_values() -> None:
    assert WriterProvider.ANTHROPIC.value == "anthropic"
    assert WriterProvider.OPENAI.value == "openai"
    assert WriterProvider.GROQ.value == "groq"


def test_configuration_defaults() -> None:
    """All configuration values should have default values."""
    config = Configuration()

    assert config.max_web_research_loops == 3
    assert config.local_llm == "llama3.3"
    assert config.search_api == SearchAPI.DUCKDUCKGO


def test_configuration_report_defaults() -> None:
    """All configuration values should have default values."""
    config = ConfigurationReport()

    assert config.output_dir == "reports/"
    assert config.report_structure == DEFAULT_REPORT_STRUCTURE
    assert config.number_of_queries == 2
    assert config.max_search_depth == 2
    assert config.planner_provider == PlannerProvider.OPENAI
    assert config.planner_model == "gpt-4o"
    assert config.writer_provider == WriterProvider.ANTHROPIC
    assert config.writer_model == "claude-3-5-sonnet-latest"
    assert config.search_api == SearchAPI.DUCKDUCKGO


def test_configuration_from_runnable_config() -> None:
    """Some configuration values should stem from the RunnableConfig."""
    runnable_config = RunnableConfig(configurable={"max_web_research_loops": 5, "local_llm": "alpaca"})
    config = Configuration.from_runnable_config(runnable_config)

    assert config.max_web_research_loops == 5  # RunnableConfig value overrides default value
    assert config.local_llm == "alpaca"  # RunnableConfig value overrides default value
    assert config.search_api == SearchAPI.DUCKDUCKGO  # Default value


def test_configuration_report_from_runnable_config() -> None:
    """Some configuration values should stem from the RunnableConfig."""
    runnable_config = RunnableConfig(configurable={"number_of_queries": 42, "planner_model": "o1"})
    config = ConfigurationReport.from_runnable_config(runnable_config)

    assert config.report_structure == DEFAULT_REPORT_STRUCTURE
    assert config.number_of_queries == 42  # RunnableConfig value overrides default value
    assert config.max_search_depth == 2
    assert config.planner_provider == PlannerProvider.OPENAI
    assert config.planner_model == "o1"  # RunnableConfig value overrides default value
    assert config.writer_provider == WriterProvider.ANTHROPIC
    assert config.writer_model == "claude-3-5-sonnet-latest"
    assert config.search_api == SearchAPI.DUCKDUCKGO


# This test runs multiple times, once for each pair of arguments.
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
    assert config.search_api == expected_config  # Environment variable value overrides default value.


@pytest.mark.example
def test_runnable() -> None:
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
