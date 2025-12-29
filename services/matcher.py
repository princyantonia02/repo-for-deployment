from services.semantic_matcher import semantic_match
from services.skill_weights import SKILL_WEIGHTS

def compute_match(resume_text: str, jd_text: str, top_k: int = 5):
    if not resume_text.strip() or not jd_text.strip():
        return 0, [], []

    import re
    def split_sentences(text):
        sentences = re.split(r'[.\n;!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    resume_sentences = split_sentences(resume_text)
    jd_sentences = split_sentences(jd_text)

    matches = semantic_match(resume_sentences, jd_sentences, threshold=0.5)

    if not matches:
        return 0, [], []

    # Compute average similarity as match_score
    avg_score = sum(m['score'] for m in matches) / len(matches)
    match_score = int(round(avg_score * 100))

    # Extract keywords (fix set slicing issue)
    matched_keywords = list(list({m['resume'] for m in matches})[:top_k])
    missing_keywords = list(list({m['jd'] for m in matches})[:top_k])

    return match_score, matched_keywords, missing_keywords


def explain_match(matched, missing):
    """
    Returns a list of explanations for strong matches and critical gaps.
    """
    explanations = []

    for skill in matched:
        weight = SKILL_WEIGHTS.get(skill.lower(), 1)
        if weight >= 4:
            explanations.append(f"✔ Strong match: **{skill}** is a core requirement.")

    for skill in missing:
        weight = SKILL_WEIGHTS.get(skill.lower(), 1)
        if weight >= 4:
            explanations.append(f"❌ Critical gap: **{skill}** significantly impacts your match score.")

    return explanations


def simulate_improvement(current_score, missing_skills):
    """
    Simulates improvement if missing skills are added.
    """
    improvement = 0
    for skill in missing_skills:
        improvement += SKILL_WEIGHTS.get(skill.lower(), 1) * 2
    new_score = min(current_score + improvement, 100)
    return new_score
