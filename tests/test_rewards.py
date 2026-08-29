import sys
from pathlib import Path

# Add the project root to Python's path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards import (
    length_reward,
    relevance_reward,
    format_reward,
    combined_reward,
)


def test_length_reward():
    response = "This is a good response with a reasonable length."
    score = length_reward(response)

    assert 0.0 <= score <= 1.0


def test_relevance_reward():
    response = "Python is a programming language used for data analysis."
    keywords = ["Python", "programming", "data"]

    score = relevance_reward(response, keywords)

    assert score == 1.0


def test_format_reward():
    response = "Python is useful for data analysis."
    score = format_reward(response)

    assert 0.0 <= score <= 1.0


def test_combined_reward():
    response = "Python is a programming language used for data analysis."
    keywords = ["Python", "programming", "data"]

    score = combined_reward(response, keywords)

    assert 0.0 <= score <= 1.0


def test_empty_response():
    score = combined_reward("", [])

    assert score == 0.0