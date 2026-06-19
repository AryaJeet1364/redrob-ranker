import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
import time
import os

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
BATCH_SIZE = 512          # GPU batch size — reduce to 256 if OOM
EMBEDDING_DIM = 384
OUTPUT_DIR = '/content/drive/MyDrive/redrob_data/'

# BGE query instruction prefix — ONLY for encoding the JD query, NOT candidates
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def encode_candidates(texts, candidate_ids, model):
    """
    Encode all candidate texts in batches.
    No prefix for candidate texts — BGE asymmetric encoding.
    """
    print(f"Encoding {len(texts):,} candidates...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated time on GPU: ~15-20 minutes")

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize for cosine similarity via dot product
        convert_to_numpy=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    elapsed = time.time() - t0
    print(f"\nEncoding complete in {elapsed/60:.1f} minutes")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.4f}")
    return embeddings


def build_faiss_index(embeddings):
    """
    Build FAISS IndexFlatIP (inner product = cosine similarity for normalized vectors).
    CPU index — GPU FAISS index not needed for 100K vectors.
    """
    print(f"\nBuilding FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    # FAISS requires float32
    embeddings_f32 = embeddings.astype(np.float32)
    index.add(embeddings_f32)
    print(f"FAISS index built. Total vectors: {index.ntotal:,}")
    return index


def encode_jd_query(model):
    """
    Encode the JD query with BGE query prefix.
    This is the vector used for FAISS search at ranking time.
    """
    jd_query_path = '/content/drive/MyDrive/redrob_data/jd_query.txt'
    with open(jd_query_path, 'r') as f:
        jd_text = f.read().strip()

    # Add BGE query prefix
    jd_text_with_prefix = BGE_QUERY_PREFIX + jd_text

    jd_vector = model.encode(
        [jd_text_with_prefix],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print(f"JD query vector shape: {jd_vector.shape}")
    print(f"JD query vector norm: {np.linalg.norm(jd_vector[0]):.4f}")
    return jd_vector[0]


# if __name__ == '__main__':
#     print("="*60)
#     print("DAY 4 — EMBEDDING GENERATION")
#     print("="*60)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     if device == 'cpu':
#         print("⚠️  WARNING: Running on CPU. This will take 2-3 hours.")
#         print("   Consider switching to GPU runtime in Colab.")

#     # Load model
#     print(f"\nLoading model: {MODEL_NAME}")
#     t0 = time.time()
#     model = SentenceTransformer(MODEL_NAME)
#     model.to(device)
#     print(f"Model loaded in {time.time()-t0:.1f}s")

#     # Load texts
#     print("\nLoading candidate texts...")
#     text_df = pd.read_parquet('/content/drive/MyDrive/redrob_data/candidate_texts.parquet')
#     candidate_ids = text_df['candidate_id'].tolist()
#     texts = text_df['candidate_text'].tolist()
#     print(f"Loaded {len(texts):,} texts")

#     # Encode candidates
#     embeddings = encode_candidates(texts, candidate_ids, model)

#     # Save embeddings
#     emb_path = os.path.join(OUTPUT_DIR, 'embeddings.npy')
#     np.save(emb_path, embeddings.astype(np.float32))
#     print(f"\n✅ Saved embeddings to {emb_path}")
#     print(f"   File size: {os.path.getsize(emb_path) / 1024**2:.1f} MB")

#     # Save candidate ID mapping
#     id_map_path = os.path.join(OUTPUT_DIR, 'id_map.npy')
#     np.save(id_map_path, np.array(candidate_ids))
#     print(f"✅ Saved id_map to {id_map_path}")

#     # Build and save FAISS index
#     index = build_faiss_index(embeddings)
#     index_path = os.path.join(OUTPUT_DIR, 'faiss_index.bin')
#     faiss.write_index(index, index_path)
#     print(f"✅ Saved FAISS index to {index_path}")
#     print(f"   File size: {os.path.getsize(index_path) / 1024**2:.1f} MB")

#     # Encode JD query and save
#     print("\nEncoding JD query...")
#     jd_vector = encode_jd_query(model)
#     jd_vec_path = os.path.join(OUTPUT_DIR, 'jd_vector.npy')
#     np.save(jd_vec_path, jd_vector.astype(np.float32))
#     print(f"✅ Saved JD vector to {jd_vec_path}")

#     # Quick FAISS test
#     print("\n--- FAISS RETRIEVAL TEST ---")
#     query = jd_vector.reshape(1, -1).astype(np.float32)
#     scores, indices = index.search(query, 20)

#     df = pd.read_parquet('/content/drive/MyDrive/redrob_data/candidates_df.parquet')
#     id_to_row = df.set_index('candidate_id')

#     print(f"\nTop 20 FAISS results (cosine similarity):")
#     for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
#         cid = candidate_ids[idx]
#         try:
#             row = id_to_row.loc[cid]
#             print(f"  {rank:2}. [{score:.4f}] {cid} | {row['current_title']} at {row['current_company']}")
#         except:
#             print(f"  {rank:2}. [{score:.4f}] {cid}")

#     print("\n✅ embedder.py complete")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to candidate_texts.parquet')
    parser.add_argument('--out_dir', required=True, help='Directory to save all embedding artifacts')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"\nLoading model: {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    model.to(device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    print(f"\nLoading texts from {args.input}...")
    text_df = pd.read_parquet(args.input)
    candidate_ids = text_df['candidate_id'].tolist()
    texts = text_df['candidate_text'].tolist()
    print(f"Loaded {len(texts):,} texts")

    embeddings = encode_candidates(texts, candidate_ids, model)

    emb_path = os.path.join(args.out_dir, 'embeddings.npy')
    np.save(emb_path, embeddings.astype(np.float32))
    print(f"✅ Saved embeddings → {emb_path}")

    id_map_path = os.path.join(args.out_dir, 'id_map.npy')
    np.save(id_map_path, np.array(candidate_ids))
    print(f"✅ Saved id_map → {id_map_path}")

    index = build_faiss_index(embeddings)
    index_path = os.path.join(args.out_dir, 'faiss_index.bin')
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index → {index_path}")

    # JD query — hardcoded text, no file dependency
    jd_query = """senior ai engineer founding team embeddings retrieval ranking vector database
    production experience sentence transformers bge e5 vector database hybrid search
    faiss pinecone weaviate qdrant elasticsearch opensearch python ndcg mrr map
    hybrid retrieval dense retrieval bm25 semantic search reranking
    machine learning nlp natural language processing transformers
    information retrieval ranking systems evaluation frameworks"""

    jd_text_with_prefix = BGE_QUERY_PREFIX + jd_query.strip()
    jd_vector = model.encode([jd_text_with_prefix], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
    jd_vec_path = os.path.join(args.out_dir, 'jd_vector.npy')
    np.save(jd_vec_path, jd_vector.astype(np.float32))
    print(f"✅ Saved JD vector → {jd_vec_path}")

    # Also save jd_query.txt for rank.py to use
    jd_query_path = os.path.join(args.out_dir, 'jd_query.txt')
    with open(jd_query_path, 'w') as f:
        f.write(jd_query.strip())
    print(f"✅ Saved jd_query.txt → {jd_query_path}")

    print("\n✅ embedder.py complete")
