"""Small text helpers shared by retrieval and feature code."""

import re


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and keep alphanumeric tokens.

    This is intentionally simple: it makes the baseline easy to understand.
    """
    return TOKEN_RE.findall((text or "").lower())

