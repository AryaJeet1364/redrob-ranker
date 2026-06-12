# Redrob Ranker — Intelligent Candidate Ranking

Ranks 100,000 candidates against a Senior AI Engineer job description using a hybrid retrieval pipeline (BM25 + dense embeddings) followed by structured feature scoring and cross-encoder reranking.

**Runtime:** ~47 seconds on CPU
**Language:** Python 3.10+

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Output Format](#output-format)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

Given a pool of 100,000 candidate profiles (JSONL) and a target job description (Senior AI Engineer), this pipeline produces a ranked shortlist of the most relevant candidates. It combines:

- **Lexical retrieval** (BM25) — catches keyword and terminology overlap
- **Dense retrieval** (BGE-small-en-v1.5 embeddings + FAISS) — catches semantic similarity
- **Structured scoring** — rule-based features extracted from candidate metadata
- **Behavioral multiplier** — adjusts scores based on career narrative quality
- **Cross-encoder reranking** — fine-grained relevance scoring on the top candidates

The result is a CSV of ranked candidates with scores and supporting metadata.

---

## Quick Start

```bash
git clone https://github.com/AryaJeet1364/redrob-ranker
cd redrob-ranker
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out submission.csv
```

### Optional arguments

```bash
python rank.py \
  --candidates ./data/candidates.jsonl \
  --jd ./data/job_description.txt \
  --config config.yaml \
  --out submission.csv \
  --top-k 200
```

| Argument | Description | Default |
|---|---|---|
| `--candidates` | Path to candidate JSONL file | `./data/candidates.jsonl` |
| `--jd` | Path to job description text file | `./data/job_description.txt` |
| `--config` | Path to YAML config with scoring weights | `config.yaml` |
| `--out` | Output CSV path | `submission.csv` |
| `--top-k` | Number of candidates passed to cross-encoder reranking | `200` |

---

## Pipeline Stages

The pipeline runs as a sequence of stages, each handled by a dedicated module:

| Stage | Module | What it does |
|---|---|---|
| 1. Parsing | `src/parser.py` | Reads the raw candidate JSONL and converts it into a structured parquet file for fast processing |
| 2. Text building | `src/text_builder.py` | Builds a unified text representation per candidate (skills, experience, career narrative) |
| 3. Lexical retrieval | `src/bm25_indexer.py` | Builds a BM25 index over candidate text and retrieves top lexical matches against the JD |
| 4. Dense retrieval | `src/embedder.py` | Encodes candidates and JD with BGE-small-en-v1.5, indexes with FAISS, retrieves top semantic matches |
| 5. Shortlist merge | `rank.py` | Combines BM25 and embedding results into a union shortlist |
| 6. Structured scoring | `src/feature_engineer.py` | Computes 20 weighted structured features per candidate |
| 7. Behavioral adjustment | `rank.py` | Applies a multiplier based on quality of career narrative |
| 8. Reranking | `rank.py` | Passes top 200 candidates through a cross-encoder for final relevance scoring |
| 9. Output | `rank.py` | Writes the final ranked list to `submission.csv` |

See [docs/architecture.md](docs/architecture.md) for a detailed breakdown of each stage.

---

## Methodology

1. **Parsing** — `src/parser.py` reads the raw JSONL of 100,000 candidates and converts it into a structured parquet file for fast downstream processing.

2. **Text Representation** — `src/text_builder.py` constructs a unified text blob per candidate (skills, experience, career narrative) used for both BM25 and embedding-based retrieval.

3. **Hybrid Retrieval** — `src/bm25_indexer.py` builds a BM25 index over candidate text and retrieves the top-N lexical matches against the JD, while `src/embedder.py` encodes candidates and the JD using BGE-small-en-v1.5, indexes embeddings with FAISS, and retrieves the top-N semantic matches. The two result sets are merged into a union shortlist.

4. **Structured Scoring** — `src/feature_engineer.py` computes 20 structured features per candidate (e.g. years of experience, skill overlap count, education tier, role seniority match, location fit, etc.), each weighted per `config.yaml`.

5. **Behavioral Multiplier** — adjusts the structured score based on qualitative signals in the candidate's career description (e.g. clarity, ownership language, progression narrative), helping surface candidates like Sarvam AI's profile (rank 4) whose written career summary stood out.

6. **Cross-Encoder Reranking** — the top 200 candidates from the previous stages are passed through a cross-encoder model for pairwise relevance scoring against the JD, producing the final ranking.

7. **Output** — the final ranked list is written to `submission.csv`.

---

## Project Structure

| Path | Description |
|---|---|
| `rank.py` | Main entry point — orchestrates the full pipeline |
| `config.yaml` | All scoring weights and pipeline parameters |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files and folders excluded from version control |
| `data/candidates.jsonl` | Input candidate pool (not committed) |
| `src/parser.py` | JSONL to parquet parsing |
| `src/text_builder.py` | Builds candidate text representations |
| `src/bm25_indexer.py` | BM25 lexical retrieval |
| `src/embedder.py` | BGE embeddings and FAISS dense retrieval |
| `src/feature_engineer.py` | 20 structured scoring features |
| `docs/architecture.md` | Detailed pipeline documentation |
| `submission_metadata.yaml` | Run metadata (model versions, timings, etc.) |

---

## Configuration

All tunable parameters live in `config.yaml`, including:

- BM25 and embedding retrieval sizes (top-N per method)
- Feature weights for the 20 structured scoring features
- Behavioral multiplier ranges
- Cross-encoder model name and top-K for reranking
- Output column ordering

Example snippet:

```yaml
retrieval:
  bm25_top_n: 500
  embedding_top_n: 500

scoring:
  weights:
    experience_years: 0.15
    skill_overlap: 0.25
    education_tier: 0.10
    role_seniority_match: 0.20
    location_fit: 0.05
    # ... remaining structured features

behavioral_multiplier:
  min: 0.9
  max: 1.15

reranking:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 200
```

---

## Results

| Metric | Value |
|---|---|
| Top 20 candidates classified as `AI_ML_CORE` | 100% |
| Honeypot/distractor profiles in top 100 | 0 |
| Notable result | Sarvam AI candidate (strongest career description) ranked #4 |
| Total runtime (100,000 candidates, CPU) | ~47 seconds |

---

## Output Format

`submission.csv` contains one row per ranked candidate with the following columns:

| Column | Description |
|---|---|
| `rank` | Final position in the ranking |
| `candidate_id` | Unique identifier from source data |
| `name` | Candidate name |
| `category` | Classification (e.g. `AI_ML_CORE`) |
| `bm25_score` | Raw BM25 retrieval score |
| `embedding_score` | Cosine similarity from dense retrieval |
| `structured_score` | Weighted sum of 20 structured features |
| `behavioral_multiplier` | Multiplier applied based on career narrative |
| `final_score` | Score after cross-encoder reranking |

---

## Requirements

See `requirements.txt`. Core dependencies include:

| Package | Purpose |
|---|---|
| `rank-bm25` | BM25 lexical retrieval |
| `sentence-transformers` | BGE embeddings and cross-encoder model |
| `faiss-cpu` | Dense vector indexing and search |
| `pandas` | Data manipulation |
| `pyyaml` | Config file parsing |
| `pyarrow` | Parquet read/write |

Install with:

```bash
pip install -r requirements.txt
```

---

## License

MIT
