import streamlit as st
import PyPDF2
import io
import os
import requests
import pandas as pd

import re
from services.matcher import compute_match, explain_match, simulate_improvement

from services.matcher import compute_match, explain_match, simulate_improvement
from services.skill_extractor import compute_skill_gap
from services.skills_list import SKILL_LIST

# ---------------- Load env ----------------

HF_API_KEY = st.secrets("HUGGINGFACE_API_KEY")

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

if not HF_API_KEY:
    st.error("HUGGINGFACE_API_KEY not found in .env")
    st.stop()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ResumeGPT", layout="centered")
st.title("🚀 ResumeGPT: Smart Resume Feedback")

uploaded_file = st.file_uploader("Upload resume (PDF/TXT)", type=["pdf", "txt"])
st.subheader("📌 Job Description (Optional)")

job_description = st.text_area(
    "Paste the job description to get resume–JD matching insights",
    height=200,
    placeholder="Example: Looking for a Data Scientist with Python, SQL, ML, and data visualization skills..."
).strip()

jd_provided = len(job_description) > 0
analyze = st.button("Generate Feedback 📝")

# ---------------- Helpers ----------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(file.read()))
    return file.read().decode("utf-8")

def split_sentences(text):
    sentences = re.split(r'[.\n;!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def query_hf(prompt: str) -> str:
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""
    return ""

# ---------------- Main Logic ----------------
if analyze and uploaded_file:
    resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("❌ Could not extract text from resume.")
        st.stop()

    # ---------- Resume–JD Match & Skill Gap ----------
    if jd_provided:
        # Split sentences for semantic matching
        resume_sentences = split_sentences(resume_text)
        jd_sentences = split_sentences(job_description)

        match_score, matched_keywords, missing_keywords = compute_match(
            "\n".join(resume_sentences),
            "\n".join(jd_sentences)
        )

        critical, nice, noise = compute_skill_gap(resume_text, job_description, SKILL_LIST)
        explanations = explain_match(matched_keywords, missing_keywords)
        improved_score = simulate_improvement(match_score, missing_keywords)

        # Score breakdown
        skills_score = round(match_score * 0.4)
        keywords_score = round(match_score * 0.3)
        experience_score = round(match_score * 0.2)
        formatting_score = round(match_score * 0.1)

        # ---------- Display Results ----------
        st.subheader("📊 Resume–Job Match")
        st.metric("Overall Match Score", f"{match_score}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Matched Keywords**")
            for kw in matched_keywords:
                st.write(f"• {kw}")
        with col2:
            st.markdown("**❌ Missing Keywords**")
            for kw in missing_keywords:
                st.write(f"• {kw}")

        st.subheader("🎯 Skill Priority Breakdown")
        st.markdown("**❌ Critical Gaps (Fix Immediately)**")
        st.write(critical if critical else "None")
        st.markdown("**⚠️ Nice-to-Have Skills**")
        st.write(nice if nice else "None")
        st.markdown("**🟢 Low-Impact / Noise**")
        st.write(noise if noise else "None")

        st.subheader("🧠 Why This Score?")
        if explanations:
            for e in explanations:
                st.write(e)
        else:
            st.write("No high-impact skills detected in the current match.")

        st.subheader("🚀 Improvement Simulation")
        st.write(
            f"If you focus on the missing high-impact skills, your match score could reasonably improve from "
            f"**{match_score}% → {improved_score}%**."
        )

        st.subheader("📊 Score Distribution (Derived)")
        st.caption("Scores are derived from overall similarity, not independently calculated.")
        breakdown_df = pd.DataFrame({
            "Category": ["Skills Match", "Keywords", "Experience Relevance", "Formatting"],
            "Score (%)": [skills_score, keywords_score, experience_score, formatting_score]
        })
        st.table(breakdown_df)

    # ---------- AI Resume Feedback ----------
    prompt = f"""
Analyze the resume below and produce concise, scannable feedback.

SECTION 1 — SUMMARY (MANDATORY)
- Output EXACTLY 4 bullet points
- Each bullet must:
  • Start with a strong action verb
  • Be 20 words or fewer
  • Mention a concrete strength or gap
- No generic praise
- Do NOT mention dates, years, timelines, or future-dated issues

SECTION 2 — DETAILED FEEDBACK
Organize feedback under the following headings (use emojis exactly as shown):

🎯 Clarity & Impact  
⚙️ Skills Relevance  
📄 Structure & Formatting  
✨ Improvements / Recommendations  

For each heading:
- Provide 2–6 bullet points
- Be specific and actionable
- Avoid mentioning dates, years, or time-related information
- Do NOT reference hiring decisions, recruiters, or HR language

STYLE RULES
- Be direct and professional
- No fluff, no filler
- No assumptions beyond resume content
- Focus on skills, clarity, and alignment only

Resume content:
{resume_text}
"""
    with st.spinner("Analyzing resume..."):
        ai_feedback = query_hf(prompt)

    if ai_feedback:
        st.subheader("🤖 AI Resume Feedback")
        st.markdown(f"<div style='font-size:16px; line-height:1.5'>{ai_feedback}</div>",
                    unsafe_allow_html=True)
    else:
        st.info("AI feedback unavailable. Showing analytical insights only.")
