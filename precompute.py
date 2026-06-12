"""
Precompute all offline artifacts needed by rank.py.
Run once before ranking. No time limit on this step.

Usage:
    python precompute.py --candidates ./data/candidates.jsonl --out_dir ./artifacts
"""
import subprocess, sys, os, argparse, time

def run(cmd, desc):
    print(f"\n>>> {desc}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./data/candidates.jsonl")
    parser.add_argument("--out_dir", default="./artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run(f"python src/parser.py --input {args.candidates} --output {args.out_dir}/candidates_df.parquet", "Parse JSONL")
    run(f"python src/feature_engineer.py --input {args.out_dir}/candidates_df.parquet --output {args.out_dir}/features_df.parquet", "Feature engineering")
    run(f"python src/text_builder.py --input {args.out_dir}/candidates_df.parquet --output {args.out_dir}/candidate_texts.parquet", "Build text blocks")
    run(f"python src/bm25_indexer.py --input {args.out_dir}/candidate_texts.parquet --output {args.out_dir}/bm25_index.pkl", "BM25 index")
    run(f"python src/embedder.py --input {args.out_dir}/candidate_texts.parquet --out_dir {args.out_dir}", "Embeddings + FAISS (~20 min on GPU)")

    print("\n✅ Precomputation complete. Now run:")
    print(f"  python rank.py --candidates {args.candidates} --out submission.csv")
