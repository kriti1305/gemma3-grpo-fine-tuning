"""
Reward functions for GRPO fine-tuning.

The reward system evaluates model responses based on:
1. Answer quality
2. Relevance
3. Conciseness
4. Basic formatting
"""


def length_reward(response: str) -> float:
    """Reward responses with a reasonable length."""
    length = len(response.strip())

    if length == 0:
        return 0.0
    elif 20 <= length <= 500:
        return 1.0
    elif length < 20:
        return 0.3
    else:
        return 0.7


def relevance_reward(response: str, keywords: list[str]) -> float:
    """Reward responses containing relevant keywords."""
    if not response.strip() or not keywords:
        return 0.0

    response_lower = response.lower()

    matches = sum(
        1 for keyword in keywords
        if keyword.lower() in response_lower
    )

    return min(matches / len(keywords), 1.0)


def format_reward(response: str) -> float:
    """Reward responses with basic readable formatting."""
    if not response.strip():
        return 0.0

    score = 0.0

    if len(response.split()) >= 5:
        score += 0.4

    if response[0].isupper():
        score += 0.2

    if response.rstrip()[-1] in ".!?":
        score += 0.2

    if "\n" in response:
        score += 0.2

    return min(score, 1.0)


def combined_reward(
    response: str,
    keywords: list[str] | None = None
) -> float:
    """
    Calculate the final reward score.

    The final score combines multiple reward signals.
    """
    if keywords is None:
        keywords = []

    length_score = length_reward(response)
    relevance_score = relevance_reward(response, keywords)
    format_score = format_reward(response)

    final_score = (
        0.3 * length_score
        + 0.5 * relevance_score
        + 0.2 * format_score
    )

    return round(final_score, 4)