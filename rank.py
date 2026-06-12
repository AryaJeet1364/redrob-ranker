
import pandas as pd
import numpy as np
import faiss
import pickle
import re
import time
import argparse
import csv
import os
from sentence_transformers import CrossEncoder

# ============================================================
# CONFIG — tune these weights on Day 6
# ============================================================
# WEIGHTS = {
#     'title_score': 0.30,
#     'skill_depth_score': 0.22,
#     'career_ai_fraction': 0.18,
#     'product_company_fraction': 0.10,
#     'yoe_score': 0.10,
#     'education_score': 0.05,
#     'github_score': 0.05,
# }

WEIGHTS = {
    'title_score': 0.30,
    'skill_depth_score': 0.16,      # REDUCED from 0.22
    'career_ai_fraction': 0.18,
    'product_company_fraction': 0.10,
    'yoe_score': 0.16,              # INCREASED from 0.10
    'education_score': 0.05,
    'github_score': 0.05,
}

# Score fusion weights
TEXT_WEIGHT = 0.40       # combined BM25 + FAISS
STRUCT_WEIGHT = 0.60     # structured features

# Within text score
BM25_WEIGHT = 0.55
FAISS_WEIGHT = 0.45

# Cross-encoder blend
CE_WEIGHT = 0.45
PRE_CE_WEIGHT = 0.55

CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CE_BATCH_SIZE = 16
SHORTLIST_SIZE = 5000    # per retriever
CE_CANDIDATES = 200      # apply cross-encoder to top N

CONSULTING_FIRMS = [
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'tech mahindra', 'mphasis', 'hexaware',
    'l&t infotech', 'ltimindtree', 'hcl technologies', 'hcltech',
    'mindtree', 'niit technologies', 'zensar', 'mastech', 'kpit'
]

JD_HARD_SKILLS = [
    'embeddings', 'embedding', 'vector search', 'vector database',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    'elasticsearch', 'hybrid search', 'bge', 'e5', 'python', 'ndcg',
    'mrr', 'ranking', 'retrieval', 'information retrieval', 'reranking',
    'semantic search', 'dense retrieval', 'bm25', 'machine learning',
    'nlp', 'natural language processing', 'transformers',
    'sentence-transformers', 'sentence transformers'
]

# ============================================================
# HELPERS
# ============================================================

def simple_tokenize(text):
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r'[^a-z0-9\-\_]', text)
    return [t for t in tokens if len(t) >= 2 and not t.isdigit()]

def safe_list(x):
    if x is None:
        return []
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, list):
        return x
    return []

def is_consulting(company):
    if not company or not isinstance(company, str):
        return False
    c = company.lower()
    return any(firm in c for firm in CONSULTING_FIRMS)

# ============================================================
# REASONING GENERATOR — template-based, zero hallucination
# ============================================================

def generate_reasoning(row, feat_row, rank):
    """
    Pull facts directly from parsed data. Never invent anything.
    """
    title = str(row.get('current_title', ''))
    company = str(row.get('current_company', ''))
    yoe = float(row.get('years_of_experience', 0))
    location = str(row.get('location', ''))
    country = str(row.get('country', ''))

    skill_names = safe_list(row.get('skill_names', []))
    skill_profs = safe_list(row.get('skill_proficiencies', []))

    # Find top JD-matching skills
    matching_skills = []
    for name, prof in zip(skill_names, skill_profs):
        if not name:
            continue
        name_lower = str(name).lower()
        if any(jd_skill in name_lower or name_lower in jd_skill
               for jd_skill in JD_HARD_SKILLS):
            matching_skills.append(f"{name} ({prof})")
    top_skills = ', '.join(matching_skills[:3]) if matching_skills else 'adjacent ML skills'

    # Location string
    loc_str = f"{location}, {country}" if location else country

    # Concerns
    concerns = []
    notice = int(feat_row.get('notice_period_days', 90))
    if notice > 90:
        concerns.append(f"notice period {notice} days")
    days_inactive = float(feat_row.get('days_since_active', 0))
    if days_inactive > 90:
        concerns.append(f"inactive {int(days_inactive)} days")
    if feat_row.get('consulting_only_flag', 0) == 1:
        concerns.append("consulting-only background")
    if country.lower() != 'india':
        if not feat_row.get('willing_to_relocate', False):
            concerns.append(f"based in {country}, not open to relocate")

    # Build reasoning
    strengths = f"{yoe:.0f} yrs exp as {title}; key skills: {top_skills}; {loc_str}"
    if concerns:
        concern_str = '; concern: ' + ', '.join(concerns[:2])
    else:
        concern_str = '; strong availability signals'

    return (strengths + concern_str)[:200]  # cap at 200 chars


# ============================================================
# MAIN RANKING PIPELINE
# ============================================================

def run_ranking(
    candidates_path,
    features_path,
    texts_path,
    bm25_pkl_path,
    faiss_bin_path,
    id_map_path,
    jd_vector_path,
    jd_query_path,
    output_path
):
    stage_times = {}
    total_start = time.time()

    # ----------------------------------------------------------
    # LOAD ARTIFACTS
    # ----------------------------------------------------------
    t = time.time()
    print("Loading artifacts...")

    feat = pd.read_parquet(features_path).set_index('candidate_id')
    df_raw = pd.read_parquet(candidates_path).set_index('candidate_id')
    text_df = pd.read_parquet(texts_path).set_index('candidate_id')

    with open(bm25_pkl_path, 'rb') as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data['bm25']
    bm25_ids = bm25_data['candidate_ids']

    index = faiss.read_index(faiss_bin_path)
    id_map = np.load(id_map_path).tolist()
    jd_vector = np.load(jd_vector_path).astype(np.float32)

    with open(jd_query_path, 'r') as f:
        jd_query_text = f.read().strip()

    stage_times['load'] = time.time() - t
    print(f"  Load time: {stage_times['load']:.1f}s")

    # ----------------------------------------------------------
    # HYBRID RETRIEVAL
    # ----------------------------------------------------------
    t = time.time()
    print("Running hybrid retrieval...")

    # BM25
    query_tokens = simple_tokenize(jd_query_text)
    bm25_scores_arr = bm25.get_scores(query_tokens)
    bm25_top_indices = np.argsort(bm25_scores_arr)[::-1][:SHORTLIST_SIZE]
    bm25_top_ids = set(bm25_ids[i] for i in bm25_top_indices)
    bm25_score_map = {bm25_ids[i]: bm25_scores_arr[i] for i in bm25_top_indices}
    bm25_max = bm25_scores_arr[bm25_top_indices[0]]

    # FAISS
    query_vec = jd_vector.reshape(1, -1)
    faiss_scores_raw, faiss_indices = index.search(query_vec, SHORTLIST_SIZE)
    faiss_top_ids = set(id_map[i] for i in faiss_indices[0])
    faiss_score_map = {id_map[faiss_indices[0][i]]: float(faiss_scores_raw[0][i])
                       for i in range(len(faiss_indices[0]))}

    # Union shortlist
    shortlist_ids = list(bm25_top_ids | faiss_top_ids)
    stage_times['retrieval'] = time.time() - t
    print(f"  Shortlist size: {len(shortlist_ids):,} | Time: {stage_times['retrieval']:.1f}s")

    # ----------------------------------------------------------
    # STRUCTURED FEATURE SCORING ON SHORTLIST
    # ----------------------------------------------------------
    t = time.time()
    print("Computing structured scores...")

    results = []
    for cid in shortlist_ids:
        if cid not in feat.index:
            continue

        fr = feat.loc[cid]

        # Skip honeypots entirely
        if fr.get('honeypot_flag', 0) == 1:
            continue

        # Structured score
        struct_score = sum(
            WEIGHTS[col] * float(fr.get(col, 0))
            for col in WEIGHTS
        )

        # Consulting-only hard penalty
        if fr.get('consulting_only_flag', 0) == 1:
            struct_score *= 0.15

        # Text scores (normalized)
        bm25_norm = bm25_score_map.get(cid, 0) / bm25_max if bm25_max > 0 else 0
        faiss_sim = faiss_score_map.get(cid, 0)  # already 0-1

        text_score = BM25_WEIGHT * bm25_norm + FAISS_WEIGHT * faiss_sim

        # Raw fusion score
        raw_score = STRUCT_WEIGHT * struct_score + TEXT_WEIGHT * text_score

        # Behavioral multiplier (multiplicative)
        behavioral = float(fr.get('behavioral_multiplier', 0.1))
        final_score = raw_score * behavioral

        results.append({
            'candidate_id': cid,
            'struct_score': struct_score,
            'text_score': text_score,
            'raw_score': raw_score,
            'behavioral': behavioral,
            'pre_ce_score': final_score,
        })

    results.sort(key=lambda x: x['pre_ce_score'], reverse=True)
    stage_times['scoring'] = time.time() - t
    print(f"  Scored {len(results):,} candidates | Time: {stage_times['scoring']:.1f}s")

    # ============================================================
    # CAREER QUALITY FLOOR - Protect exceptional candidates
    # ============================================================
    raw_scores = [r['raw_score'] for r in results]
    raw_95th = np.percentile(raw_scores, 95)
    
    protected = 0
    for r in results:
        if r['raw_score'] >= raw_95th and r['behavioral'] < 0.4:
            new_multiplier = 0.4
            r['pre_ce_score'] = r['raw_score'] * new_multiplier
            protected += 1
    
    print(f"  Career quality floor: protected {protected} exceptional candidates")
    
    # Re-sort after applying floor
    results.sort(key=lambda x: x['pre_ce_score'], reverse=True)



    

    # ----------------------------------------------------------
    # CROSS-ENCODER RERANKING ON TOP 200
    # ----------------------------------------------------------
    t = time.time()
    print(f"Running cross-encoder on top {CE_CANDIDATES}...")

    # Load full JD text for cross-encoder (richer than query string)
    jd_full = """Senior AI Engineer role at Redrob AI. Requirements: production experience with
    embeddings-based retrieval systems, vector databases, hybrid search infrastructure,
    Python, evaluation frameworks for ranking systems (NDCG, MRR, MAP). Ideal candidate:
    5-9 years experience, shipped ranking/search/recommendation systems to real users,
    product company background, not consulting-only. Values: scrappy product engineering,
    deep ML systems knowledge, retrieval and ranking expertise."""

    ce_model = CrossEncoder(CROSS_ENCODER_MODEL)

    top_200 = results[:CE_CANDIDATES]
    ce_pairs = []
    for r in top_200:
        cid = r['candidate_id']
        if cid in text_df.index:
            cand_text = str(text_df.loc[cid, 'candidate_text'])[:512]
        else:
            cand_text = cid
        ce_pairs.append((jd_full, cand_text))

    ce_scores_raw = ce_model.predict(ce_pairs, batch_size=CE_BATCH_SIZE,
                                      show_progress_bar=True)

    # Normalize CE scores to 0-1
    ce_min, ce_max = ce_scores_raw.min(), ce_scores_raw.max()
    if ce_max > ce_min:
        ce_scores_norm = (ce_scores_raw - ce_min) / (ce_max - ce_min)
    else:
        ce_scores_norm = np.ones_like(ce_scores_raw) * 0.5

    # Blend with pre-CE score
    pre_ce_scores = np.array([r['pre_ce_score'] for r in top_200])
    pre_min, pre_max = pre_ce_scores.min(), pre_ce_scores.max()
    if pre_max > pre_min:
        pre_ce_norm = (pre_ce_scores - pre_min) / (pre_max - pre_min)
    else:
        pre_ce_norm = np.ones_like(pre_ce_scores) * 0.5

    final_scores = CE_WEIGHT * ce_scores_norm + PRE_CE_WEIGHT * pre_ce_norm

    for i, r in enumerate(top_200):
      cid = r['candidate_id']
      if cid in df_raw.index:
          country = str(df_raw.loc[cid].get('country', '')).lower()
          willing = bool(df_raw.loc[cid].get('willing_to_relocate', False))
          if country != 'india' and not willing:
              final_scores[i] *= 0.60

    for i, r in enumerate(top_200):
        r['ce_score'] = float(ce_scores_norm[i])
        r['final_score'] = float(final_scores[i])

    # Sort by final blended score
    top_200.sort(key=lambda x: x['final_score'], reverse=True)

    # Remaining candidates (ranks 101-200 area) keep pre_ce_score
    remaining = results[CE_CANDIDATES:]

    stage_times['ce'] = time.time() - t
    print(f"  Cross-encoder done | Time: {stage_times['ce']:.1f}s")

    # ----------------------------------------------------------
    # SELECT TOP 100 AND GENERATE REASONING
    # ----------------------------------------------------------
    t = time.time()
    print("Generating reasoning and writing CSV...")

    top_100 = top_200[:100]

    # Ensure scores are monotonically non-increasing
    max_final = top_100[0]['final_score']
    min_final = top_100[-1]['final_score']
    score_range = max_final - min_final if max_final > min_final else 1.0

    rows = []
    for rank, r in enumerate(top_100, 1):
        cid = r['candidate_id']

        # Normalize to 0.10-0.99 range for submission score
        # submission_score = 0.10 + 0.89 * (r['final_score'] - min_final) / score_range

        # Normalize to 0.40-0.99 range for submission score (fixes score compression)
        submission_score = 0.40 + 0.59 * (r['final_score'] - min_final) / score_range
        submission_score = round(submission_score, 6)

        # Get profile data
        if cid in df_raw.index:
            profile = df_raw.loc[cid]
            feat_row = feat.loc[cid].to_dict() if cid in feat.index else {}
            reasoning = generate_reasoning(profile, feat_row, rank)
        else:
            reasoning = f"Rank {rank} candidate based on hybrid retrieval and structured scoring."

        rows.append({
            'candidate_id': cid,
            'rank': rank,
            'score': submission_score,
            'reasoning': reasoning
        })

    # Ensure monotonic scores (just in case)
    rows.sort(key=lambda x: x['score'], reverse=True)
    for i, row in enumerate(rows, 1):
        row['rank'] = i

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        writer.writeheader()
        writer.writerows(rows)

    stage_times['output'] = time.time() - t
    total_time = time.time() - total_start

    print(f"\n{'='*50}")
    print(f"RANKING COMPLETE")
    print(f"{'='*50}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nStage breakdown:")
    for stage, t_val in stage_times.items():
        print(f"  {stage:12s}: {t_val:.1f}s")
    print(f"\nOutput: {output_path}")
    print(f"Top candidate: {top_100[0]['candidate_id']} (score={rows[0]['score']})")

    # Verify monotonic scores
    scores_list = [r['score'] for r in rows]
    is_monotonic = all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1))
    print(f"Scores monotonically non-increasing: {is_monotonic}")

    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='/content/drive/MyDrive/redrob_data/candidates_df.parquet')
    parser.add_argument('--features', default='/content/drive/MyDrive/redrob_data/features_df.parquet')
    parser.add_argument('--texts', default='/content/drive/MyDrive/redrob_data/candidate_texts.parquet')
    parser.add_argument('--bm25', default='/content/drive/MyDrive/redrob_data/bm25_index.pkl')
    parser.add_argument('--faiss', default='/content/drive/MyDrive/redrob_data/faiss_index.bin')
    parser.add_argument('--id_map', default='/content/drive/MyDrive/redrob_data/id_map.npy')
    parser.add_argument('--jd_vector', default='/content/drive/MyDrive/redrob_data/jd_vector.npy')
    parser.add_argument('--jd_query', default='/content/drive/MyDrive/redrob_data/jd_query.txt')
    parser.add_argument('--out', default='/content/drive/MyDrive/redrob_data/submission_v1.csv')
    args = parser.parse_args()

    run_ranking(
        candidates_path=args.candidates,
        features_path=args.features,
        texts_path=args.texts,
        bm25_pkl_path=args.bm25,
        faiss_bin_path=args.faiss,
        id_map_path=args.id_map,
        jd_vector_path=args.jd_vector,
        jd_query_path=args.jd_query,
        output_path=args.out
    )
