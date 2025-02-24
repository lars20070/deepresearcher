#!/usr/bin/env python3

import subprocess
import sys

from dotenv import load_dotenv

from deepresearcher.logger import logger

"""
CLI wrappers in order to add these commands to [project.scripts] in pyproject.toml
"""


def startserver() -> None:
    """
    Start the LangGraph server.
    """
    logger.info("Loading environment variables")
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


def uml() -> None:
    """
    Generate UML diagrams.
    """
    logger.info("Generating UML diagrams")
    cmd = [
        "uv",
        "run",
        "pyreverse",
        "-o",
        "png",
        "-A",
        "-k",
        "-d",
        "./uml",
        "./src/deepresearcher",
    ]
    sys.exit(subprocess.call(cmd))
