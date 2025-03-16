Note that neither the local Ollama models nor the API endpoints are mocked.

By default all tests with paid API calls are disabled. Please comment out the following line in the `[tool.pytest.ini_options]` section of [`pyproject.toml`](../pyproject.toml) to enable them. 
```toml
addopts = "-m 'not paid'"
```
If you do, please ensure the [repository secrets](../.github/workflows/build.yml#27) include all necessary API keys.

Tests marked `@pytest.mark.example` are simple examples for understanding the LangChain / LangGraph codebase. They do not test functionality in the `deepresearcher` package.