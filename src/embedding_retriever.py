"""Sentence-Transformers paragraph retriever."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".cache" / "sentence-transformers"))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRetriever:
    """Dense vector retriever using Sentence-Transformers embeddings."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.corpus = corpus
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.paragraph_embeddings = self.model.encode(
            [paragraph["text"] for paragraph in corpus],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = cosine_similarity(query_embedding, self.paragraph_embeddings)[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": rank,
                "paragraph_id": self.corpus[int(index)]["paragraph_id"],
                "paper_id": self.corpus[int(index)]["paper_id"],
                "title": self.corpus[int(index)]["title"],
                "score": float(scores[int(index)]),
                "text": self.corpus[int(index)]["text"],
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]
