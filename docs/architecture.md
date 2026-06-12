
System Architecture
===================

Two-Phase Pipeline
------------------

Offline Pre-computation (unlimited time):
- parser.py -> candidates_df.parquet
- feature_engineer.py -> features_df.parquet (20 features)
- text_builder.py -> candidate_texts.parquet
- bm25_indexer.py -> bm25_index.pkl
- embedder.py -> embeddings.npy + faiss_index.bin

Online Ranking (< 5 min on CPU):
1. Hybrid retrieval: BM25 (top 5000) + FAISS (top 5000) -> union shortlist (~8000)
2. Structured scoring: 7 features with tuned weights
3. Behavioral multiplier: availability x reachability x notice
4. Cross-encoder reranking on top 200 (ms-marco-MiniLM-L-6-v2)
5. Template-based reasoning -> submission.csv

Key Design Decisions
--------------------

Hybrid retrieval: BM25 and FAISS top-20 had only 2 overlapping candidates,
confirming they find different relevant candidates.

Behavioral as multiplier: A perfect candidate who is inactive/unreachable
is not hirable. Multiplicative pattern is correct per JD specification.

Cross-encoder only on top 200: 200 pairs x 0.15 s = ~30 s on CPU.
Applying to all 100000 would exceed 5-minute constraint.

Template reasoning: Zero hallucination risk. Pulls only facts from profile.
