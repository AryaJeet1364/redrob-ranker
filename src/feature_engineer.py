import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ============================================================
# CONSTANTS
# ============================================================

EVAL_DATE = pd.Timestamp('2026-06-10')

CONSULTING_FIRMS = [
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'tech mahindra', 'mphasis', 'hexaware',
    'l&t infotech', 'ltimindtree', 'hcl technologies', 'hcltech',
    'mindtree', 'niit technologies', 'zensar', 'mastech', 'kpit'
]

JD_HARD_SKILLS = [
    'embeddings', 'embedding', 'vector search', 'vector database',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    'elasticsearch', 'hybrid search', 'hybrid retrieval',
    'sentence-transformers', 'sentence transformers', 'bge', 'e5',
    'python', 'ndcg', 'mrr', 'map', 'ranking', 'retrieval',
    'information retrieval', 'reranking', 're-ranking',
    'semantic search', 'dense retrieval', 'bm25', 'machine learning',
    'nlp', 'natural language processing', 'transformers'
]

JD_NICE_TO_HAVE = [
    'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'finetuning',
    'learning to rank', 'learning-to-rank', 'xgboost', 'lightgbm',
    'hrtech', 'hr tech', 'recruiting', 'talent', 'distributed systems',
    'kafka', 'spark', 'large scale inference', 'mlops', 'kubeflow'
]

PROFICIENCY_WEIGHTS = {
    'beginner': 0.25,
    'intermediate': 0.5,
    'advanced': 0.75,
    'expert': 1.0
}

# ============================================================
# HELPER: Convert numpy arrays to lists safely
# ============================================================

def safe_list(x):
    """Convert numpy array to list safely"""
    if x is None:
        return []
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, list):
        return x
    return [] if x is None else [x] if not isinstance(x, (list, tuple)) else list(x)

# ============================================================
# TITLE CATEGORIZATION
# ============================================================

AI_ML_KEYWORDS = [
    'ml engineer', 'machine learning engineer', 'ai engineer',
    'artificial intelligence engineer', 'nlp engineer', 'nlp scientist',
    'research engineer', 'applied scientist', 'research scientist',
    'data scientist', 'applied ml', 'deep learning engineer',
    'computer vision engineer', 'cv engineer', 'senior ml',
    'staff ml', 'principal ml', 'senior ai', 'senior applied',
    'ml researcher', 'ai researcher', 'senior data scientist',
    'lead data scientist', 'principal data scientist',
    'search engineer', 'ranking engineer', 'recommendation engineer',
    'information retrieval', 'conversational ai'
]

SOFTWARE_ENG_KEYWORDS = [
    'software engineer', 'software developer', 'backend engineer',
    'backend developer', 'full stack', 'fullstack', 'frontend engineer',
    'frontend developer', 'devops engineer', 'sre ', 'site reliability',
    'platform engineer', 'cloud engineer', 'java developer',
    '.net developer', 'python developer', 'golang developer',
    'node.js developer', 'mobile developer', 'android developer',
    'ios developer', 'systems engineer', 'infrastructure engineer'
]

DATA_ENG_KEYWORDS = [
    'data engineer', 'analytics engineer', 'data analyst',
    'bi engineer', 'business intelligence', 'etl engineer',
    'data pipeline', 'data architect', 'database engineer',
    'data platform', 'data infrastructure'
]

IRRELEVANT_KEYWORDS = [
    'hr manager', 'human resources', 'accountant', 'accounting',
    'operations manager', 'marketing manager', 'content writer',
    'sales executive', 'sales manager', 'graphic designer',
    'mechanical engineer', 'civil engineer', 'electrical engineer',
    'customer support', 'customer success', 'business analyst',
    'project manager', 'product manager', 'scrum master',
    'qa engineer', 'quality assurance', 'test engineer',
    'ui designer', 'ux designer', 'financial analyst',
    'supply chain', 'logistics', 'procurement'
]

def get_title_category(title):
    if not title or not isinstance(title, str):
        return 'UNKNOWN'
    t = title.lower().strip()
    for kw in AI_ML_KEYWORDS:
        if kw in t:
            return 'AI_ML_CORE'
    for kw in SOFTWARE_ENG_KEYWORDS:
        if kw in t:
            return 'SOFTWARE_ENGINEERING'
    for kw in DATA_ENG_KEYWORDS:
        if kw in t:
            return 'DATA_ENGINEERING'
    for kw in IRRELEVANT_KEYWORDS:
        if kw in t:
            return 'IRRELEVANT'
    return 'UNKNOWN'

TITLE_SCORES = {
    'AI_ML_CORE': 1.0,
    'SOFTWARE_ENGINEERING': 0.5,
    'DATA_ENGINEERING': 0.4,
    'CONSULTING': 0.2,
    'IRRELEVANT': 0.05,
    'UNKNOWN': 0.3
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_consulting(company_name):
    if not company_name or not isinstance(company_name, str):
        return False
    c = company_name.lower()
    return any(firm in c for firm in CONSULTING_FIRMS)

def skill_matches_jd(skill_name, jd_skill_list):
    if not skill_name or not isinstance(skill_name, str):
        return False
    s = skill_name.lower()
    return any(jd_skill in s or s in jd_skill for jd_skill in jd_skill_list)

# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def compute_title_features(row):
    cat = get_title_category(row['current_title'])
    if is_consulting(row['current_company']):
        if cat == 'AI_ML_CORE':
            return cat, TITLE_SCORES[cat] * 0.6
    return cat, TITLE_SCORES[cat]

def compute_career_ai_months(all_titles, all_companies):
    all_titles = safe_list(all_titles)
    if len(all_titles) == 0:
        return 0.0
    ai_title_count = sum(1 for t in all_titles if isinstance(t, str) 
                        and get_title_category(t) in ('AI_ML_CORE', 'SOFTWARE_ENGINEERING'))
    total_titles = len(all_titles)
    return ai_title_count / total_titles

def compute_product_company_fraction(all_companies, all_company_sizes):
    all_companies = safe_list(all_companies)
    all_company_sizes = safe_list(all_company_sizes)
    if len(all_companies) == 0:
        return 0.5
    product_count = 0.0
    total = len(all_companies)
    for company, size in zip(all_companies, all_company_sizes):
        if not isinstance(company, str):
            continue
        if is_consulting(company):
            continue
        if size in ('51-200', '201-500', '501-1000', '1001-5000'):
            product_count += 1.0
        elif size in ('11-50', '1-10'):
            product_count += 0.8
        elif size in ('5001-10000'):
            product_count += 0.6
        elif size in ('10001+'):
            product_count += 0.3
        else:
            product_count += 0.4
    return product_count / total if total > 0 else 0.5

def compute_skill_depth_score(skill_names, skill_proficiencies, skill_endorsements, skill_durations):
    skill_names = safe_list(skill_names)
    skill_proficiencies = safe_list(skill_proficiencies)
    skill_endorsements = safe_list(skill_endorsements)
    skill_durations = safe_list(skill_durations)
    if len(skill_names) == 0:
        return 0.0
    total_score = 0.0
    matched = 0
    for name, prof, endorse, dur in zip(skill_names, skill_proficiencies,
                                         skill_endorsements, skill_durations):
        if skill_matches_jd(name, JD_HARD_SKILLS):
            pw = PROFICIENCY_WEIGHTS.get(prof, 0.25)
            ew = min(1.0, endorse / 20.0)
            dw = min(1.0, dur / 36.0)
            total_score += pw * (0.5 + 0.3 * ew + 0.2 * dw)
            matched += 1
    if matched == 0:
        return 0.0
    return min(1.0, total_score / max(matched, 3))

def compute_nice_to_have_score(skill_names, skill_proficiencies, skill_durations, github_score):
    skill_names = safe_list(skill_names)
    skill_proficiencies = safe_list(skill_proficiencies)
    skill_durations = safe_list(skill_durations)
    if len(skill_names) == 0:
        return 0.0
    score = 0.0
    count = 0
    for name, prof, dur in zip(skill_names, skill_proficiencies, skill_durations):
        if skill_matches_jd(name, JD_NICE_TO_HAVE):
            pw = PROFICIENCY_WEIGHTS.get(prof, 0.25)
            dw = min(1.0, dur / 24.0)
            score += pw * dw
            count += 1
    if github_score > 0:
        score += 0.3
        count += 1
    if count == 0:
        return 0.0
    return min(1.0, score / max(count, 2))

def compute_yoe_score(yoe):
    if yoe < 3:
        return 0.2
    elif yoe < 5:
        return 0.6
    elif yoe <= 9:
        return 1.0
    elif yoe <= 12:
        return 0.8
    else:
        return 0.6

def compute_availability_score(open_to_work, days_since_active, apps_30d):
    otw = 1.0 if open_to_work else 0.3
    if pd.isna(days_since_active):
        recency = 0.3
    elif days_since_active < 30:
        recency = 1.0
    elif days_since_active < 60:
        recency = 0.8
    elif days_since_active < 90:
        recency = 0.5
    elif days_since_active < 180:
        recency = 0.2
    else:
        recency = 0.05
    active = min(1.0, apps_30d / 3.0)
    return otw * recency * (0.5 + 0.5 * active)

def compute_reachability_score(response_rate, avg_response_hours, verified_email, verified_phone):
    rr = response_rate if response_rate >= 0 else 0.3
    if avg_response_hours < 24:
        speed = 1.0
    elif avg_response_hours < 72:
        speed = 0.7
    elif avg_response_hours < 168:
        speed = 0.4
    else:
        speed = 0.2
    vb = 1.1 if (verified_email and verified_phone) else 1.0
    return min(1.0, rr * speed * vb)

def compute_notice_period_score(notice_days):
    if notice_days <= 30:
        return 1.0
    elif notice_days <= 60:
        return 0.8
    elif notice_days <= 90:
        return 0.6
    elif notice_days <= 120:
        return 0.4
    else:
        return 0.2

def compute_location_score(location, country, willing_to_relocate):
    if not location or not isinstance(location, str):
        location = ''
    loc = location.lower()
    country_lower = (country or '').lower()
    if country_lower == 'india':
        if any(x in loc for x in ['pune', 'noida']):
            return 1.0
        elif any(x in loc for x in ['delhi', 'ncr', 'gurgaon', 'gurugram', 'new delhi']):
            return 0.9
        elif any(x in loc for x in ['hyderabad', 'mumbai', 'bengaluru', 'bangalore', 'chennai', 'kolkata']):
            return 0.8
        else:
            return 0.7
    else:
        return 0.5 if willing_to_relocate else 0.2

def compute_education_score(education_tiers, education_fields):
    education_tiers = safe_list(education_tiers)
    education_fields = safe_list(education_fields)
    if len(education_tiers) == 0:
        return 0.3
    tier_map = {'tier_1': 1.0, 'tier_2': 0.75, 'tier_3': 0.5, 'tier_4': 0.25, 'unknown': 0.4}
    best_tier = max((tier_map.get(t, 0.3) for t in education_tiers), default=0.3)
    field_bonus = 0.0
    for field in education_fields:
        if not isinstance(field, str):
            continue
        f = field.lower()
        if any(x in f for x in ['computer science', 'cs', 'mathematics', 'statistics',
                                  'machine learning', 'ai', 'data science']):
            field_bonus = 0.1
            break
    return min(1.0, best_tier + field_bonus)

def compute_github_score(github_activity_score):
    if github_activity_score == -1:
        return 0.1
    return 0.1 + 0.9 * (github_activity_score / 100.0)

def compute_consulting_only_flag(all_companies):
    all_companies = safe_list(all_companies)
    if len(all_companies) == 0:
        return 0
    non_consulting = [c for c in all_companies if isinstance(c, str) and not is_consulting(c)]
    return 1 if len(non_consulting) == 0 else 0

def compute_honeypot_flag_from_row(row):
    flags = 0
    if row.get('yoe_gap', 0) > 3:
        flags += 1
    
    skill_profs = row.get('skill_proficiencies', [])
    skill_durs = row.get('skill_durations', [])
    if hasattr(skill_profs, 'tolist'):
        skill_profs = skill_profs.tolist()
    if hasattr(skill_durs, 'tolist'):
        skill_durs = skill_durs.tolist()
    for prof, dur in zip(skill_profs, skill_durs):
        if prof == 'expert' and dur == 0:
            flags += 1
            break
    
    if row.get('total_skills_count', 0) > 14 and get_title_category(row.get('current_title', '')) == 'IRRELEVANT':
        flags += 1
    if row.get('years_of_experience', 0) > 8 and row.get('total_career_entries', 0) < 2:
        flags += 1
    return 1 if flags >= 2 else 0

def compute_all_features(df):
    print("Computing features for 100K candidates...")
    features = []
    
    # Precompute derived columns
    df['last_active_dt'] = pd.to_datetime(df['last_active_date'], errors='coerce')
    df['days_since_active'] = (EVAL_DATE - df['last_active_dt']).dt.days
    df['career_years'] = df['total_duration_months'] / 12
    df['yoe_gap'] = abs(df['career_years'] - df['years_of_experience'])
    df['career_years'] = df['career_years'].fillna(0)
    df['yoe_gap'] = df['yoe_gap'].fillna(0)
    df['days_since_active'] = df['days_since_active'].fillna(999)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Engineering features"):
        title_cat, title_score = compute_title_features(row)
        career_ai_fraction = compute_career_ai_months(row['all_titles'], row['all_companies'])
        product_fraction = compute_product_company_fraction(row['all_companies'], row['all_company_sizes'])
        consulting_only = compute_consulting_only_flag(row['all_companies'])
        skill_depth = compute_skill_depth_score(
            row['skill_names'], row['skill_proficiencies'],
            row['skill_endorsements'], row['skill_durations'])
        nice_to_have = compute_nice_to_have_score(
            row['skill_names'], row['skill_proficiencies'],
            row['skill_durations'], row['github_activity_score'])
        yoe_score = compute_yoe_score(row['years_of_experience'])
        availability = compute_availability_score(
            row['open_to_work'], row['days_since_active'],
            row['applications_submitted_30d'])
        reachability = compute_reachability_score(
            row['recruiter_response_rate'], row['avg_response_time_hours'],
            row['verified_email'], row['verified_phone'])
        notice_score = compute_notice_period_score(row['notice_period_days'])
        location_score = compute_location_score(
            row['location'], row['country'], row['willing_to_relocate'])
        edu_score = compute_education_score(row['education_tiers'], row['education_fields'])
        github_score = compute_github_score(row['github_activity_score'])
        honeypot = compute_honeypot_flag_from_row(row)
        behavioral_multiplier = availability * reachability * (0.6 + 0.4 * notice_score)

        features.append({
            'candidate_id': row['candidate_id'],
            'title_category': title_cat,
            'title_score': title_score,
            'career_ai_fraction': career_ai_fraction,
            'product_company_fraction': product_fraction,
            'consulting_only_flag': consulting_only,
            'skill_depth_score': skill_depth,
            'nice_to_have_score': nice_to_have,
            'yoe_score': yoe_score,
            'years_of_experience': row['years_of_experience'],
            'availability_score': availability,
            'reachability_score': reachability,
            'notice_period_score': notice_score,
            'behavioral_multiplier': behavioral_multiplier,
            'location_score': location_score,
            'education_score': edu_score,
            'github_score': github_score,
            'honeypot_flag': honeypot,
            'days_since_active': row['days_since_active'],
            'yoe_gap': row['yoe_gap'],
        })
    return pd.DataFrame(features)

# Hardcoded paths

# if __name__ == '__main__':
#     print("Loading candidates...")
#     df = pd.read_parquet('/content/drive/MyDrive/redrob_data/candidates_df.parquet')
#     print(f"Loaded {len(df):,} candidates")
#     features_df = compute_all_features(df)
#     out_path = '/content/drive/MyDrive/redrob_data/features_df.parquet'
#     features_df.to_parquet(out_path, index=False)
#     print(f"\n✅ Saved features to {out_path}")
#     print(f"   Shape: {features_df.shape}")
#     print(features_df.describe())

if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to candidates_df.parquet')
    parser.add_argument('--output', required=True, help='Path to output features parquet')
    args = parser.parse_args()

    print(f"Loading candidates from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} candidates")

    features_df = compute_all_features(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features_df.to_parquet(args.output, index=False)
    print(f"\n✅ Saved features to {args.output}")
    print(f"   Shape: {features_df.shape}")
    print(features_df.describe())
