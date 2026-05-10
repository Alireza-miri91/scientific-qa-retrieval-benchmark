"""BM25 paragraph retriever for the toy scientific QA benchmark."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text with lowercase alphanumeric terms."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Simple BM25 retriever over paragraph dictionaries."""

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self.corpus = corpus
        tokenized_corpus = [tokenize(paragraph["text"]) for paragraph in corpus]
        self.model = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        scores = self.model.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            {
                "rank": rank,
                "paragraph_id": self.corpus[index]["paragraph_id"],
                "paper_id": self.corpus[index]["paper_id"],
                "title": self.corpus[index]["title"],
                "score": float(scores[index]),
                "text": self.corpus[index]["text"],
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]
