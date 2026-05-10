"""Run the scientific QA retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bm25_retriever import BM25Retriever
from embedding_retriever import EmbeddingRetriever
from evaluate import evaluate_rankings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sample_corpus.json"
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data" / "sample_questions.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "example_results.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_method(
    method_name: str,
    retriever: BM25Retriever | EmbeddingRetriever,
    questions: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    rankings = {
        question["question_id"]: retriever.retrieve(question["question"], top_k=top_k)
        for question in questions
    }
    metrics = evaluate_rankings(questions, rankings, ks=(1, 3))

    return {
        "method": method_name,
        "metrics": metrics,
        "rankings": rankings,
    }


def print_metrics(results: list[dict[str, Any]]) -> None:
    print("\nRetrieval metrics")
    print("-----------------")
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['method']}: "
            f"Hit@1={metrics['hit@1']:.2f}, "
            f"Hit@3={metrics['hit@3']:.2f}"
        )


def print_example_question(
    questions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    example_question = questions[0]
    print("\nExample top results")
    print("-------------------")
    print(f"Question: {example_question['question']}")
    print(f"Gold paragraph: {', '.join(example_question['gold_paragraph_ids'])}")

    for result in results:
        method = result["method"]
        top_result = result["rankings"][example_question["question_id"]][0]
        print(
            f"{method} top-1: {top_result['paragraph_id']} "
            f"({top_result['title']}, score={top_result['score']:.3f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BM25 and Sentence-Transformers retrieval on toy scientific QA data."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_json(args.corpus)
    questions = load_json(args.questions)

    bm25_result = run_method(
        "BM25",
        BM25Retriever(corpus),
        questions,
        top_k=args.top_k,
    )
    embedding_result = run_method(
        "Sentence-Transformers",
        EmbeddingRetriever(corpus, model_name=args.embedding_model),
        questions,
        top_k=args.top_k,
    )

    results = [bm25_result, embedding_result]
    print_metrics(results)
    print_example_question(questions, results)

    payload = {
        "corpus_size": len(corpus),
        "num_questions": len(questions),
        "top_k": args.top_k,
        "embedding_model": args.embedding_model,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"\nSaved example results to {args.output}")


if __name__ == "__main__":
    main()
