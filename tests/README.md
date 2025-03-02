By default all tests with paid API calls are disabled. Please comment out the following line in the `[tool.pytest.ini_options]` section of [`pyproject.toml`](../pyproject.toml) to enable them. Please ensure the [repository secrets](../.github/workflows/build.yaml#27) include all necessary API keys.
```toml
addopts = "-m 'not paid'"
```

`test_EXAMPLE_*` are simple examples for understanding the LangChain / LangGraph codebase. They do not test functionality in the `deepresearcher` package.