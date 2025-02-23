#!/usr/bin/env python3

import subprocess
import sys

from dotenv import load_dotenv

from deepresearcher.logger import logger


def main() -> None:
    logger.info("Loading environmet variables")
    load_dotenv()

    logger.info("Starting the LangGraph server")
    cmd = [
        "uvx",
        "--refresh",
        "--from",
        "langgraph-cli[inmem]",
        "--with-editable",
        ".",
        "--python",
        "3.11",
        "langgraph",
        "dev",
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
