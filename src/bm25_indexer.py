import pandas as pd
import numpy as np
import pickle
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import time

def simple_tokenize(text):
    """
    Simple but effective tokenizer for BM25.
    Lowercase, split on non-alphanumeric, filter short tokens.
    No stemming — preserves technical terms like FAISS, BM25, NDCG.
    """
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.split(r'[^a-z0-9\-\_]', text)
    # Keep tokens >= 2 chars, filter pure numbers
    tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
    return tokens


def build_jd_query():
    """
    Extract the most signal-dense parts of the JD for BM25 querying.
    This is hardcoded from the job_description.docx.
    Focus on technical requirements, NOT the culture/vibe sections.
    """
    jd_query = """
    senior ai engineer founding team embeddings retrieval ranking vector database
    production experience embeddings sentence transformers bge e5 openai embeddings
    embedding drift index refresh retrieval quality regression production
    vector database hybrid search infrastructure pinecone weaviate qdrant milvus
    opensearch elasticsearch faiss python production code quality
    evaluation frameworks ranking systems ndcg mrr map offline online correlation
    ab testing recruiter feedback ranking retrieval matching systems
    llm fine tuning lora qlora peft learning to rank
    hybrid retrieval dense retrieval bm25 semantic search reranking
    product company applied ml shipped ranking search recommendation system
    real users meaningful scale retrieval quality evaluation
    information retrieval natural language processing transformers
    machine learning engineer nlp engineer applied scientist
    """
    return jd_query.strip()


if __name__ == '__main__':
    print("Loading candidate texts...")
    text_df = pd.read_parquet('/content/drive/MyDrive/redrob_data/candidate_texts.parquet')
    print(f"Loaded {len(text_df):,} text blocks")

    # Tokenize all texts
    print("\nTokenizing candidate texts...")
    t0 = time.time()
    tokenized_corpus = []
    for text in tqdm(text_df['candidate_text'], desc="Tokenizing"):
        tokenized_corpus.append(simple_tokenize(text))
    print(f"Tokenization done in {time.time()-t0:.1f}s")

    # Build BM25 index
    print("\nBuilding BM25 index (this takes 2-3 minutes)...")
    t1 = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built in {time.time()-t1:.1f}s")

    # Save index
    index_path = '/content/drive/MyDrive/redrob_data/bm25_index.pkl'
    print(f"\nSaving BM25 index to {index_path}...")
    with open(index_path, 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'candidate_ids': text_df['candidate_id'].tolist(),
            'tokenized_corpus': tokenized_corpus
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved BM25 index")

    # Quick retrieval test
    print("\n--- BM25 RETRIEVAL TEST ---")
    jd_query = build_jd_query()
    query_tokens = simple_tokenize(jd_query)
    print(f"Query tokens (first 20): {query_tokens[:20]}")

    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:20]

    print(f"\nTop 20 BM25 results:")
    candidate_ids = text_df['candidate_id'].tolist()

    # Load original data for display
    df = pd.read_parquet('/content/drive/MyDrive/redrob_data/candidates_df.parquet')
    id_to_row = df.set_index('candidate_id')

    for rank, idx in enumerate(top_indices, 1):
        cid = candidate_ids[idx]
        score = scores[idx]
        try:
            row = id_to_row.loc[cid]
            title = row['current_title']
            company = row['current_company']
            print(f"  {rank:2}. [{score:.2f}] {cid} | {title} at {company}")
        except:
            print(f"  {rank:2}. [{score:.2f}] {cid}")

    # Save score distribution info
    print(f"\nBM25 score stats:")
    print(f"  Max score: {scores.max():.2f}")
    print(f"  Mean score: {scores.mean():.2f}")
    print(f"  Std: {scores.std():.2f}")
    print(f"  Non-zero scores: {(scores > 0).sum():,}")
    print(f"  Top-1000 threshold: {np.sort(scores)[::-1][999]:.2f}")
    print(f"  Top-5000 threshold: {np.sort(scores)[::-1][4999]:.2f}")
