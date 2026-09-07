
import io
import json
import re
from typing import Any, Dict

import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from google import genai


MODEL_NAME = "gemini-3.5-flash"


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract selectable text from a PDF resume."""
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    """Extract paragraphs and table text from a DOCX resume."""
    document = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(cell for cell in cells if cell)
            if row_text:
                parts.append(row_text)

    return "\n".join(parts).strip()


def extract_resume_text(uploaded_file) -> str:
    """Extract resume text from PDF or DOCX."""
    file_bytes = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    if name.endswith(".docx"):
        return extract_docx_text(file_bytes)

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX resume.")


def clean_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON even if the model surrounds it with markdown fences."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Gemini returned an invalid JSON response.")
        return json.loads(match.group(0))


def analyze_resume(resume_text: str, job_description: str, api_key: str) -> Dict[str, Any]:
    """Analyze the resume with Gemini and return structured ATS feedback."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert ATS resume evaluator and professional recruiter.

Analyze the resume below. If a job description is provided, evaluate the resume
against that job description. Do NOT invent qualifications, experience,
education, certifications, metrics, or skills that are not present.

Important:
- ATS score is an estimate, not a guarantee of passing any real ATS.
- Reward relevant keywords only when they are supported by the resume.
- Penalize missing important job-description keywords.
- Check ATS-friendly structure, section headings, readability, measurable
  achievements, skills, dates, contact information, and keyword alignment.
- Give practical improvements that the candidate can actually make.
- Never recommend keyword stuffing.

Return ONLY valid JSON with exactly this structure:
{{
  "ats_score": 0,
  "score_label": "Poor|Needs Improvement|Good|Very Good|Excellent",
  "summary": "2-4 sentence overall assessment",
  "category_scores": {{
    "keyword_match": 0,
    "format_ats_compatibility": 0,
    "experience_impact": 0,
    "skills_relevance": 0,
    "clarity_readability": 0
  }},
  "strengths": ["...", "..."],
  "critical_issues": ["...", "..."],
  "improvements": [
    {{
      "priority": "High|Medium|Low",
      "section": "Summary|Experience|Education|Skills|Projects|Formatting|Other",
      "issue": "...",
      "recommendation": "..."
    }}
  ],
  "missing_keywords": ["...", "..."],
  "suggested_summary": "A rewritten professional summary based ONLY on facts in the resume",
  "ats_checklist": {{
    "contact_information": "Pass|Needs Work",
    "standard_sections": "Pass|Needs Work",
    "job_title_alignment": "Pass|Needs Work",
    "keyword_alignment": "Pass|Needs Work",
    "measurable_achievements": "Pass|Needs Work",
    "formatting": "Pass|Needs Work",
    "file_readability": "Pass|Needs Work"
  }}
}}

Scoring guidance:
- 90-100 = Excellent
- 80-89 = Very Good
- 70-79 = Good
- 50-69 = Needs Improvement
- 0-49 = Poor

Resume:
---BEGIN RESUME---
{resume_text}
---END RESUME---

Job description:
---BEGIN JOB DESCRIPTION---
{job_description if job_description.strip() else "No job description provided. Evaluate general ATS readiness."}
---END JOB DESCRIPTION---
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    result = clean_json_response(response.text)

    # Defensive normalization.
    score = int(result.get("ats_score", 0))
    result["ats_score"] = max(0, min(100, score))

    for key in [
        "strengths",
        "critical_issues",
        "improvements",
        "missing_keywords",
    ]:
        if not isinstance(result.get(key), list):
            result[key] = []

    return result


def score_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 60:
        return "orange"
    return "red"


def main():
    st.set_page_config(
        page_title="Resume ATS Analyzer",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 Resume ATS Analyzer")
    st.caption(
        "Upload your resume to get an estimated ATS score, keyword analysis, "
        "and actionable improvements powered by Gemini 2.5 Flash."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.secrets.get("GEMINI_API_KEY", ""),
            help="Your key is used only for the current analysis request.",
        )
        st.info(
            "Tip: For deployment, store GEMINI_API_KEY in Streamlit Secrets "
            "instead of putting it in your GitHub code."
        )

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
        help="PDF and DOCX are supported. Text-based PDFs work best.",
    )

    job_description = st.text_area(
        "Job Description (optional, but recommended)",
        height=220,
        placeholder=(
            "Paste the job description here. The analyzer will compare your "
            "resume with the role's requirements and identify missing keywords."
        ),
    )

    analyze_button = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if analyze_button:
        if not api_key.strip():
            st.error("Please enter your Gemini API key or configure GEMINI_API_KEY in Streamlit Secrets.")
            return

        try:
            with st.spinner("Reading your resume and analyzing it with Gemini..."):
                resume_text = extract_resume_text(uploaded_file)

                if not resume_text:
                    st.error(
                        "No selectable text was found. If this is a scanned/image-only PDF, "
                        "please use a text-based PDF or DOCX."
                    )
                    return

                # Prevent accidental oversized prompts while retaining most resumes.
                resume_text = resume_text[:60000]
                result = analyze_resume(resume_text, job_description, api_key.strip())

            st.success("Analysis complete!")

            score = result["ats_score"]
            st.subheader("ATS Score")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric("Estimated ATS Score", f"{score}/100")
                st.progress(score / 100)

            label = result.get("score_label", "")
            if label:
                st.markdown(f"**Rating:** {label}")

            st.subheader("📊 Category Scores")
            category_scores = result.get("category_scores", {})
            score_cols = st.columns(5)

            labels = [
                ("Keyword Match", "keyword_match"),
                ("ATS Format", "format_ats_compatibility"),
                ("Experience", "experience_impact"),
                ("Skills", "skills_relevance"),
                ("Clarity", "clarity_readability"),
            ]

            for col, (label_text, key) in zip(score_cols, labels):
                with col:
                    value = int(category_scores.get(key, 0))
                    st.metric(label_text, f"{value}/100")

            st.subheader("📝 Overall Assessment")
            st.write(result.get("summary", "No summary returned."))

            strengths = result.get("strengths", [])
            if strengths:
                st.subheader("✅ Strengths")
                for item in strengths:
                    st.markdown(f"- {item}")

            critical = result.get("critical_issues", [])
            if critical:
                st.subheader("⚠️ Critical Issues")
                for item in critical:
                    st.markdown(f"- {item}")

            missing = result.get("missing_keywords", [])
            if missing:
                st.subheader("🔑 Missing / Weak Keywords")
                st.write(", ".join(missing))

            improvements = result.get("improvements", [])
            if improvements:
                st.subheader("🚀 Recommended Improvements")
                for index, item in enumerate(improvements, start=1):
                    priority = item.get("priority", "Medium")
                    section = item.get("section", "Other")
                    issue = item.get("issue", "")
                    recommendation = item.get("recommendation", "")

                    with st.expander(
                        f"{index}. {section} — {priority} priority"
                    ):
                        st.markdown(f"**Issue:** {issue}")
                        st.markdown(f"**Recommendation:** {recommendation}")

            st.subheader("📋 ATS Checklist")
            checklist = result.get("ats_checklist", {})
            checklist_cols = st.columns(2)

            for i, (key, value) in enumerate(checklist.items()):
                with checklist_cols[i % 2]:
                    icon = "✅" if value == "Pass" else "⚠️"
                    st.markdown(f"**{icon} {key.replace('_', ' ').title()}:** {value}")

            suggested_summary = result.get("suggested_summary", "")
            if suggested_summary:
                st.subheader("✨ Suggested Professional Summary")
                st.info(suggested_summary)

            with st.expander("🔎 Extracted Resume Text"):
                st.text_area(
                    "Text used for analysis",
                    resume_text,
                    height=300,
                    label_visibility="collapsed",
                )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "Check that your Gemini API key is valid, the selected file is a "
                "readable PDF/DOCX, and your API quota is available."
            )

    st.divider()
    st.caption(
        "ATS scores are estimates for guidance only. Different applicant tracking "
        "systems use different parsing and ranking rules."
    )


if __name__ == "__main__":
    main()
