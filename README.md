# Deep Researcher

Fully local web research and report writing assistant.

The codebase is a copy of the [`ollama-deep-researcher`](https://github.com/langchain-ai/ollama-deep-researcher) repository. It is extended with extensive testing and logging which should help in understanding and debugging the code.

## Getting started
1. Create an `.env` file and fill in the placeholders. An empty `.env` file is fine as well.
   ```bash
   cp .env.example .env
   ```
2. Start up the local LangGraph server.
   ```bash
   uv run startserver
   ```
3. Open [LangGraph Studio](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024) in a Chrome browser.
   ```bash
   open -a "Google Chrome" 'https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024'
   ```

## Requirements

1. ### Ollama

   [Install Ollama](https://ollama.com/download) and the Llama 3.3 model.
   ```bash
   ollama pull llama3.3
   ```
   By default, `deepresearcher` is working with local Ollama models. If you want to use hosted models instead, please specify the relevant API keys in the `.env` file.
2. ### Pandoc

   By default, research reports are generated in markdown format. If Pandoc is installed on the system, the workflow will automatically generate a PDF version of the report as well.
   ```bash
   brew install pandoc
   ```

## UML diagrams and code structure

* For understanding the codebase, please start with the `graph` and `graph_report` objects in [`src/deepresearcher/graph.py`](src/deepresearcher/graph.py).

* Internally, LangGraph is passing `dict` objects between nodes. But all state objects are pydantic. That makes type coercion at runtime at the beginning of each node method necessary. Very awkward.
  ```python
  if isinstance(state, dict):
    state = SectionState(**state)
  ```

![package diagram](./uml/packages.png "Deep Researcher package structure")
<br>*Deep Researcher package structure*

<br>

![class diagram](./uml/classes.png "Deep Researcher class structure")
<br>*Deep Researcher class structure*
