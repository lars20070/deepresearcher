#!/usr/bin/env python3

import subprocess
import sys

from deepresearcher.logger import logger


def main() -> None:
    logger.info("Load environmetal variables")
    cmd = ["source", ".env"]
    sys.exit(subprocess.call(cmd))

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
