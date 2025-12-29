# services/skill_extractor.py

def extract_skills(text, skill_list):
    """
    Check which skills from skill_list are present in the given text.
    Returns a set of matched skills.
    """
    text_lower = text.lower()
    matched = set()
    for skill in skill_list:
        if skill.lower() in text_lower:
            matched.add(skill)
    return matched


def compute_skill_gap(resume_text, jd_text, skill_list):
    """
    Returns three things:
    - skills in both resume and JD
    - skills missing from resume but in JD
    - skills present in resume but not in JD (extra)
    """
    resume_skills = extract_skills(resume_text, skill_list)
    jd_skills = extract_skills(jd_text, skill_list)

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    extra = resume_skills - jd_skills

    return matched, missing, extra


from services.skill_weights import SKILL_IMPORTANCE

def classify_skills(missing_skills):
    classified = {"critical": [], "nice": [], "noise": []}

    for s in missing_skills:
        level = SKILL_IMPORTANCE.get(s.lower(), "nice")
        classified[level].append(s)

    return classified
