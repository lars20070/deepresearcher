#!/usr/bin/env python3

from deepresearcher import HelloWorld, logger


def test_helloworld() -> None:
    """
    Test HelloWorld
    """
    logger.info("Testing HelloWorld")
    hw = HelloWorld()
    assert hw is not None
