#!/usr/bin/env python3
"""Clearly synthetic business signals layered after relevance ranking."""

from dataclasses import dataclass


@dataclass
class Candidate:
    product_id: str
    title: str
    relevance: float
    in_stock: bool
    conversion_propensity: float
    seller_quality: float


def rerank(candidates, relevance_weight=0.70, conversion_weight=0.20, seller_weight=0.10):
    available = [candidate for candidate in candidates if candidate.in_stock]
    return sorted(
        available,
        key=lambda item: relevance_weight * item.relevance
        + conversion_weight * item.conversion_propensity
        + seller_weight * item.seller_quality,
        reverse=True,
    )


if __name__ == "__main__":
    example = [
        Candidate("P07", "Nike Pegasus", .97, False, .12, .95),
        Candidate("P08", "Adidas Ultraboost", .92, True, .18, .91),
        Candidate("P09", "Nike Revolution", .88, True, .23, .88),
    ]
    print("Before (relevance only):", [item.title for item in example])
    print("After (stock constraint + objectives):", [item.title for item in rerank(example)])

