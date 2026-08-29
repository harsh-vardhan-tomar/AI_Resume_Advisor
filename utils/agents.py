"""
STEP 3 & 4: Prompts + Agents
------------------------------
There are two "agents" here - meaning two separate responsibilities:

1. Resume Reviewer Agent -> evaluates the resume against the JD context
2. Career Advisor Agent  -> builds a roadmap from the reviewer's output

An "agent" here is simply a function that:
  - uses a specific prompt
  - calls the LLM
  - returns structured (JSON) output

No complex agent framework is needed - clear separation of
responsibility is what evaluators want to see.

This version uses Groq (free tier, no credit card, runs Llama models).
Groq's API is OpenAI-compatible, so we reuse the standard `openai`
Python package and just point it at Groq's base URL.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file, if present

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-120b"  # free-tier model on Groq (Llama models were deprecated)


def _call_llm(system_prompt, user_prompt):
    """Common helper - calls the LLM and gets a JSON response back"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},  # forces valid JSON output
        temperature=0.3,  # low temperature = more consistent, less creative/hallucination
    )
    return json.loads(response.choices[0].message.content)


def resume_reviewer_agent(resume_text, relevant_jd_chunks):
    """
    AGENT 1: Compares the resume against the relevant parts of the JD.
    Output: match score, matched skills, missing skills, weak areas
    """
    jd_context = "\n---\n".join(relevant_jd_chunks)

    system_prompt = """You are an expert resume reviewer and ATS specialist.
Analyze the resume against the job description context and respond ONLY
with valid JSON in this exact format:
{
  "match_score": <0-100 integer>,
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "weak_areas": [{"area": "...", "reason": "..."}],
  "summary": "2-3 sentence overall assessment"
}

In the summary and anywhere else you write about the applicant, always
refer to them as "the candidate" - never use their actual name, even if
it appears in the resume text."""

    user_prompt = f"""RESUME:
{resume_text}

RELEVANT JOB DESCRIPTION SECTIONS:
{jd_context}

Analyze the fit and return the JSON."""

    return _call_llm(system_prompt, user_prompt)


def career_advisor_agent(review_result):
    """
    AGENT 2: Takes the reviewer's output and builds an actionable roadmap.
    Input: output of resume_reviewer_agent
    Output: prioritized skills + interview prep roadmap
    """
    system_prompt = """You are a career advisor. Based on the missing skills
and weak areas provided, create an actionable roadmap. Respond ONLY with
valid JSON in this format:
{
  "priority_skills": [{"skill": "...", "priority": "critical|nice-to-have"}],
  "roadmap": [{"week": 1, "focus": "...", "action_items": ["...", "..."]}],
  "interview_tips": ["tip1", "tip2"]
}"""

    user_prompt = f"""Missing skills and weak areas from resume review:
{json.dumps(review_result, indent=2)}

Create a 4-week roadmap and interview prep tips."""

    return _call_llm(system_prompt, user_prompt)


def learning_plan_agent(review_result):
    """
    STRETCH GOAL: Builds a 3-month learning plan based on missing skills.
    """
    system_prompt = """You are a learning path designer. Create a 3-month
learning plan for the missing skills. Respond ONLY with valid JSON:
{
  "months": [
    {"month": 1, "goals": ["..."], "resources": ["..."], "milestone": "..."}
  ]
}"""

    user_prompt = f"""Missing skills:
{json.dumps(review_result.get('missing_skills', []), indent=2)}

Create a month-by-month learning plan."""

    return _call_llm(system_prompt, user_prompt)