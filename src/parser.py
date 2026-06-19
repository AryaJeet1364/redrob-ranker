import orjson
import pandas as pd
from tqdm import tqdm
import gc
import os

def parse_candidate(c):
    p = c.get('profile', {})
    ch = c.get('career_history', [])
    sk = c.get('skills', [])
    ed = c.get('education', [])
    sig = c.get('redrob_signals', {})
    certs = c.get('certifications', [])

    # Handle salary range
    salary = sig.get('expected_salary_range_inr_lpa', {})

    return {
        # Identity
        'candidate_id': c.get('candidate_id', ''),
        'headline': p.get('headline', ''),
        'summary': p.get('summary', ''),
        'location': p.get('location', ''),
        'country': p.get('country', ''),
        'years_of_experience': p.get('years_of_experience', 0),
        'current_title': p.get('current_title', ''),
        'current_company': p.get('current_company', ''),
        'current_company_size': p.get('current_company_size', ''),
        'current_industry': p.get('current_industry', ''),

        # Career
        'total_career_entries': len(ch),
        'total_duration_months': sum(e.get('duration_months', 0) for e in ch),
        'all_titles': [e.get('title', '') for e in ch],
        'all_companies': [e.get('company', '') for e in ch],
        'all_industries': [e.get('industry', '') for e in ch],
        'all_company_sizes': [e.get('company_size', '') for e in ch],
        'career_descriptions': ' '.join(e.get('description', '') for e in ch),
        'has_current_role': any(e.get('is_current', False) for e in ch),

        # Skills
        'total_skills_count': len(sk),
        'skill_names': [s.get('name', '') for s in sk],
        'skill_proficiencies': [s.get('proficiency', '') for s in sk],
        'skill_endorsements': [s.get('endorsements', 0) for s in sk],
        'skill_durations': [s.get('duration_months', 0) for s in sk],

        # Education
        'education_tiers': [e.get('tier', 'unknown') for e in ed],
        'education_fields': [e.get('field_of_study', '') for e in ed],
        'education_degrees': [e.get('degree', '') for e in ed],

        # Certifications
        'cert_names': [cert.get('name', '') for cert in certs],

        # Signals - flattened
        'profile_completeness_score': sig.get('profile_completeness_score', 0),
        'last_active_date': sig.get('last_active_date', ''),
        'open_to_work': sig.get('open_to_work_flag', False),
        'applications_submitted_30d': sig.get('applications_submitted_30d', 0),
        'recruiter_response_rate': sig.get('recruiter_response_rate', 0),
        'avg_response_time_hours': sig.get('avg_response_time_hours', 999),
        'notice_period_days': sig.get('notice_period_days', 90),
        'preferred_work_mode': sig.get('preferred_work_mode', ''),
        'willing_to_relocate': sig.get('willing_to_relocate', False),
        'github_activity_score': sig.get('github_activity_score', -1),
        'profile_views_received_30d': sig.get('profile_views_received_30d', 0),
        'saved_by_recruiters_30d': sig.get('saved_by_recruiters_30d', 0),
        'interview_completion_rate': sig.get('interview_completion_rate', 0),
        'offer_acceptance_rate': sig.get('offer_acceptance_rate', -1),
        'verified_email': sig.get('verified_email', False),
        'verified_phone': sig.get('verified_phone', False),
        'linkedin_connected': sig.get('linkedin_connected', False),
        'connection_count': sig.get('connection_count', 0),
        'endorsements_received': sig.get('endorsements_received', 0),
        'expected_salary_min': salary.get('min', 0),
        'expected_salary_max': salary.get('max', 0),
        'search_appearance_30d': sig.get('search_appearance_30d', 0),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--output', required=True, help='Path to output parquet')
    args = parser.parse_args()

    print(f"📝 Parsing candidates from {args.input}...")
    records = []
    with open(args.input, 'rb') as f:
        for i, line in enumerate(tqdm(f, total=100000, desc="Parsing JSONL")):
            if line.strip():
                try:
                    records.append(parse_candidate(orjson.loads(line)))
                except Exception as e:
                    print(f"Error at line {i}: {e}")
                    continue

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"✅ Parsed {len(df):,} candidates → {args.output}")
    print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")
