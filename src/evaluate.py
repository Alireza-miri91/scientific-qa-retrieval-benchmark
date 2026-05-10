"""Evaluation helpers for paragraph retrieval."""

from __future__ import annotations

from typing import Any


def hit_at_k(results: list[dict[str, Any]], gold_ids: set[str], k: int) -> bool:
    """Return True if any gold paragraph appears in the top-k results."""
    top_ids = {result["paragraph_id"] for result in results[:k]}
    return bool(top_ids & gold_ids)


def evaluate_rankings(
    questions: list[dict[str, Any]],
    rankings: dict[str, list[dict[str, Any]]],
    ks: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Compute Hit@k metrics for a method's per-question rankings."""
    metrics: dict[str, float] = {}

    for k in ks:
        hits = []
        for question in questions:
            gold_ids = set(question["gold_paragraph_ids"])
            question_results = rankings[question["question_id"]]
            hits.append(hit_at_k(question_results, gold_ids, k))
        metrics[f"hit@{k}"] = sum(hits) / len(hits)

    return metrics
