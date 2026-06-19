import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def safe_list(x):
    if x is None:
        return []
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, list):
        return x
    return []

PROFICIENCY_BOOST = {
    'expert': 3,
    'advanced': 2,
    'intermediate': 1,
    'beginner': 1
}

def build_candidate_text(row):
    """
    Build a single text block per candidate for BM25 and embedding.
    Strategy:
    - Current title and company first (high signal)
    - Summary
    - All career descriptions (richest signal)
    - Skills weighted by proficiency (repeat advanced/expert for BM25 TF boost)
    - Education and certifications
    """
    parts = []

    # Title and company (repeat for emphasis)
    title = str(row.get('current_title', '') or '')
    company = str(row.get('current_company', '') or '')
    headline = str(row.get('headline', '') or '')
    summary = str(row.get('summary', '') or '')

    parts.append(title)
    parts.append(title)  # repeat title for BM25 weight
    parts.append(company)
    parts.append(headline)
    parts.append(summary)

    # Career descriptions — most important signal
    career_desc = str(row.get('career_descriptions', '') or '')
    parts.append(career_desc)

    # All historical titles (catch career progression)
    all_titles = safe_list(row.get('all_titles', []))
    if all_titles:
        parts.append(' '.join(str(t) for t in all_titles if t))

    # Skills — repeat based on proficiency level
    skill_names = safe_list(row.get('skill_names', []))
    skill_profs = safe_list(row.get('skill_proficiencies', []))
    skill_parts = []
    for name, prof in zip(skill_names, skill_profs):
        if not name:
            continue
        repeat = PROFICIENCY_BOOST.get(str(prof), 1)
        skill_parts.extend([str(name)] * repeat)
    if skill_parts:
        parts.append(' '.join(skill_parts))

    # Education fields
    edu_fields = safe_list(row.get('education_fields', []))
    edu_degrees = safe_list(row.get('education_degrees', []))
    if edu_fields:
        parts.append(' '.join(str(f) for f in edu_fields if f))
    if edu_degrees:
        parts.append(' '.join(str(d) for d in edu_degrees if d))

    # Certifications
    cert_names = safe_list(row.get('cert_names', []))
    if cert_names:
        parts.append(' '.join(str(c) for c in cert_names if c))

    # Combine and clean
    full_text = ' '.join(p for p in parts if p and p.strip())
    # Remove excessive whitespace
    full_text = ' '.join(full_text.split())
    return full_text


def build_all_texts(df):
    print("Building candidate text blocks...")
    candidate_ids = []
    texts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building texts"):
        cid = row['candidate_id']
        text = build_candidate_text(row)
        candidate_ids.append(cid)
        texts.append(text)

    return candidate_ids, texts

# Hardcoded paths

# if __name__ == '__main__':
#     print("Loading candidates...")
#     df = pd.read_parquet('/content/drive/MyDrive/redrob_data/candidates_df.parquet')
#     print(f"Loaded {len(df):,} candidates")

#     candidate_ids, texts = build_all_texts(df)

#     # Save candidate_ids and texts as a lightweight parquet
#     text_df = pd.DataFrame({
#         'candidate_id': candidate_ids,
#         'candidate_text': texts
#     })
#     out_path = '/content/drive/MyDrive/redrob_data/candidate_texts.parquet'
#     text_df.to_parquet(out_path, index=False)
#     print(f"\n✅ Saved {len(text_df):,} text blocks to {out_path}")


if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to candidates_df.parquet')
    parser.add_argument('--output', required=True, help='Path to output texts parquet')
    args = parser.parse_args()

    print(f"Loading candidates from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} candidates")

    candidate_ids, texts = build_all_texts(df)

    text_df = pd.DataFrame({
        'candidate_id': candidate_ids,
        'candidate_text': texts
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    text_df.to_parquet(args.output, index=False)
    print(f"\n✅ Saved {len(text_df):,} text blocks to {args.output}")
