# Scientific QA Retrieval Benchmark

This is a small, public-safe portfolio demo inspired by QASPER-style scientific
question answering. The task is to retrieve the paragraph that best supports an
answer to a scientific question.

The demo compares two retrieval approaches:

- **BM25**: a strong lexical baseline for exact terminology.
- **Sentence-Transformers**: a dense embedding retriever for semantic matching.

## Why This Project Matters

Scientific QA often depends on precise evidence. A system should not only answer
a question, but also find the paragraph that supports the answer. This benchmark
shows how to evaluate retrieval methods with simple, interpretable top-k metrics.

## Data

The repository uses a tiny hand-written toy corpus in `data/`. It is public-safe
and does not include private course files, unpublished notes, or large datasets.

The files are:

- `data/sample_corpus.json`: toy scientific paragraphs.
- `data/sample_questions.json`: QASPER-style questions with gold paragraph IDs.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── sample_corpus.json
│   └── sample_questions.json
├── src/
│   ├── bm25_retriever.py
│   ├── embedding_retriever.py
│   ├── evaluate.py
│   └── main.py
└── outputs/
    └── example_results.json
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

The script prints Hit@1 and Hit@3 for both methods and writes:

```text
outputs/example_results.json
```

## Example Output

The exact dense retrieval scores may vary slightly by package/model version, but
the output will look like:

```text
Retrieval metrics
-----------------
BM25: Hit@1=..., Hit@3=...
Sentence-Transformers: Hit@1=..., Hit@3=...

Example top results
-------------------
Question: ...
Gold paragraph: ...
BM25 top-1: ...
Sentence-Transformers top-1: ...
```

## Skills Demonstrated

- NLP retrieval evaluation
- BM25 lexical search
- Sentence-Transformers embeddings
- top-k metrics such as Hit@1 and Hit@3
- clean experiment structure and reproducible CLI scripts
- public-safe data handling

## Public-Safe Notes

This repo is a clean demo, not a dump of the original research workspace. It
does not include QASPER caches, notebooks, PDFs, course files, generated
experiment logs, credentials, personal data, or huge datasets.
