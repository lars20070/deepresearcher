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
        "dot",
        "-A",
        #        "-k",
        "-d",
        "./uml",
        "./src/deepresearcher",
    ]
    result = subprocess.call(cmd)
    if result != 0:
        logger.error("Failed to generate UML diagrams with pyreverse.")
        sys.exit(result)
    cmd = [
        "dot",
        "-Tpng",
        "-Grankdir=LR",
        "-o",
        "./uml/classes.png",
        "./uml/classes.dot",
    ]
    result = subprocess.call(cmd)
    if result != 0:
        logger.error("Failed to generate UML diagram for classes with Graphviz.")
        sys.exit(result)
    cmd = [
        "dot",
        "-Tpng",
        "-Grankdir=LR",
        "-o",
        "./uml/packages.png",
        "./uml/packages.dot",
    ]
    result = subprocess.call(cmd)
    if result != 0:
        logger.error("Failed to generate UML diagram for packages with Graphviz.")
    sys.exit(result)
