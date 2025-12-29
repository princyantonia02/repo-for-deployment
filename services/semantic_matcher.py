from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_match(resume_sentences, jd_sentences, threshold=0.6):
    """
    Returns list of semantic matches between JD lines and resume sentences.
    
    Each match includes:
    - jd: the JD line
    - resume: the closest matching resume sentence
    - score: cosine similarity (0-1)
    """
    if not resume_sentences or not jd_sentences:
        return []

    # Encode all sentences
    resume_emb = model.encode(resume_sentences, convert_to_numpy=True)
    jd_emb = model.encode(jd_sentences, convert_to_numpy=True)

    matches = []

    for i, jd_line in enumerate(jd_sentences):
        sims = cosine_similarity([jd_emb[i]], resume_emb)[0]
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score >= threshold:
            matches.append({
                "jd": jd_line,
                "resume": resume_sentences[best_idx],
                "score": round(float(best_score), 2)
            })

    return matches
