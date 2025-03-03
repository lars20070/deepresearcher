# deep researcher

Fully local web research and report writing assistant.

The codebase is a copy of the [`ollama-deep-researcher`](https://github.com/langchain-ai/ollama-deep-researcher) repository. It is extended with extensive testing and logging which should help in understanding and debugging the code.

## Getting started
1. Create an `.env` file and fill in the placeholders. An empty one is fine as well.
    ```bash
    cp .env.example .env
    ```
2. Start up the local LangGraph server.
    ```bash
    uv run startserver
    ```
3. Open [LangGraph Studio](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024) in Chrome browser.
    ```bash
    open -a "Google Chrome" 'https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024'
    ```

## UML diagrams

For understanding the codebase, please start with the `graph` and `graph_report` objects in [`src/deepresearcher/graph.py`](src/deepresearcher/graph.py).

![package diagram](./uml/packages.png "Deep Researcher package structure")
<br>*Deep Researcher package structure*

<br>

![class diagram](./uml/classes.png "Deep Researcher class structure")
<br>*Deep Researcher class structure*
