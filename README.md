# deep researcher

Fully local web research and report writing assistant.

The codebase is a copy of the [`ollama-deep-researcher`](https://github.com/langchain-ai/ollama-deep-researcher) repository. It is extended with extensive testing and logging which should help in understanding and debugging the code.

## Getting started
1. Install dependencies
    ```bash
    uv sync
    ```
2. Create `.env` file and fill in the placeholders
    ```bash
    cp .env.example .env
    ```
3. Start up the local LangGraph server
    ```bash
    uv run startserver
    ```
4. Open LangGraph Studio in Chrome browser
    ```bash
    open -a "Google Chrome" 'https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024'
    ```

## UML diagrams

For understanding the codebase, please start with the `graph` object in `src/deepresearcher/graph.py`.

![class diagram](./uml/classes.png "Deep Researcher class structure")
<br>*Deep Researcher class structure*

<br>

![package diagram](./uml/packages.png "Deep Researcher package structure")
<br>*Deep Researcher package structure*
