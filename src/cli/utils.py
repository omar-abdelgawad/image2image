"""Utility functions for the CLI."""

import argparse


def is_positive_float(value: str) -> float:
    """A function to check if the value is a positive float

    Args:
        value (str): String to check

    Raises:
        argparse.ArgumentTypeError: Whether value is not a positive float.

    Returns:
        float: The value if it is a positive float.
    """
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive float value")
    return ivalue


def is_positive_int(value: str) -> int:
    """A function to check if the value is a positive int

    Args:
        value (str): String to check.

    Raises:
        argparse.ArgumentTypeError: Whether value is not a positive int.

    Returns:
        int: The value if it is a positive int.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue
