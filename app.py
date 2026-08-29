"""
STEP 5: UI (Streamlit)
------------------------
This is the final piece - it ties together all the functions in
utils/ into a simple web page where the user can upload a resume
and JD, and see the results.

To run:
  1. In the terminal: export GEMINI_API_KEY="your-key-here"
     (on Windows PowerShell: $env:GEMINI_API_KEY="your-key-here")
  2. streamlit run app.py
"""

import streamlit as st
from utils.pdf_extract import extract_text_from_pdf
from utils.rag import JDRetriever
from utils.agents import resume_reviewer_agent, career_advisor_agent, learning_plan_agent

st.set_page_config(page_title="AI Resume & Career Advisor", layout="wide")

st.title("📄 AI Resume & Career Advisor")
st.write("Upload your resume and a target job description to get an AI-powered fit analysis and roadmap.")

# We use st.session_state to remember results across reruns.
# Streamlit reruns the whole script on every interaction (like clicking
# a checkbox), so without session_state, checking a box further down
# the page would wipe out the analysis we already computed.
if "review" not in st.session_state:
    st.session_state.review = None
if "advice" not in st.session_state:
    st.session_state.advice = None
if "learning_plan" not in st.session_state:
    st.session_state.learning_plan = None

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload resume (PDF)", type="pdf")
with col2:
    jd_file = st.file_uploader("Upload job description (PDF)", type="pdf")

analyze_clicked = st.button("Analyze", type="primary", disabled=not (resume_file and jd_file))

if analyze_clicked:
    with st.spinner("Reading the resume..."):
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file)

    with st.spinner("Indexing the job description (RAG)..."):
        retriever = JDRetriever()
        num_chunks = retriever.index_jd(jd_text)
        st.caption(f"Job description split into {num_chunks} chunks")

        # use the resume text as the query to find the relevant JD parts
        relevant_chunks = retriever.retrieve_relevant_chunks(resume_text, top_k=4)

    with st.spinner("Running the Resume Reviewer Agent..."):
        st.session_state.review = resume_reviewer_agent(resume_text, relevant_chunks)

    with st.spinner("Running the Career Advisor Agent..."):
        st.session_state.advice = career_advisor_agent(st.session_state.review)

    # reset any previously generated learning plan since this is a fresh analysis
    st.session_state.learning_plan = None

# ---------- RESULTS DISPLAY ----------
# This reads from session_state, so it stays visible even after the
# checkbox below causes a rerun.
if st.session_state.review is not None:
    review = st.session_state.review
    advice = st.session_state.advice

    st.divider()
    st.subheader("Match Score")
    score = review.get("match_score", 0)
    st.progress(score / 100)
    st.metric("Score", f"{score}/100")
    st.write(review.get("summary", ""))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Matched Skills")
        for skill in review.get("matched_skills", []):
            st.write(f"- {skill}")
    with col_b:
        st.subheader("❌ Missing Skills")
        for skill in review.get("missing_skills", []):
            st.write(f"- {skill}")

    st.subheader("⚠️ Weak Areas")
    for area in review.get("weak_areas", []):
        st.write(f"**{area.get('area')}**: {area.get('reason')}")

    st.divider()
    st.subheader("🗺️ 4-Week Roadmap")
    for week in advice.get("roadmap", []):
        with st.expander(f"Week {week.get('week')}: {week.get('focus')}"):
            for item in week.get("action_items", []):
                st.write(f"- {item}")

    st.subheader("🎯 Interview Tips")
    for tip in advice.get("interview_tips", []):
        st.write(f"- {tip}")

    # Stretch goal - optional
    show_plan = st.checkbox("Also show a 3-month learning plan (stretch goal)")
    if show_plan:
        if st.session_state.learning_plan is None:
            with st.spinner("Building the learning plan..."):
                st.session_state.learning_plan = learning_plan_agent(review)
        plan = st.session_state.learning_plan
        for month in plan.get("months", []):
            st.write(f"**Month {month.get('month')}**")
            st.write(f"Goals: {', '.join(month.get('goals', []))}")
            st.write(f"Milestone: {month.get('milestone')}")