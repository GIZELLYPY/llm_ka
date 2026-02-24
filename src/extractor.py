"""LLM Extraction - This uses OpenAI + LangChain and forces structured output by instructing 
JSON and validating wih Pydantic -> ExtractedQuestion.model_validate(data) ."""

import json
try:
    from langchain_openai import ChatOpenAI
except ImportError:  # backward-compatible fallback
    from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import APIConnectionError, RateLimitError
from pydantic import ValidationError

from .config import OPENAI_BASE_URL, OPENAI_MODEL
from .schema import ExtractedQuestion

SYSTEM_PROMPT = """
You are an information extraction system.

You will receive a StackOverflow question (title + body).
Extract structured fields and return ONLY  valid JSON that matches this schema:

{
"question_id": int,
"title": str,
"primary_technology": str|null,
"secondary_technologies": [str],
"problem_type": str|null,
"intent": "how-to"|"debugging"|"conceptual"|"opinion"|"unknown",
"difficulty":"beginner"|"intermediate"|"advanced"|"unknown",
"key_entities": [str],
"summary": str,
"confidence": float(0..1)
}

Rules:
- Do NOT invent facts.
- If not sure, use null or "unknown".
- Keep secondary_technologies and key_entities deduplicated.
- summary must be 1 setence, max 25 words.
- confidence: 0.00 to 1.0 reflecting how supported the extraction is.
Return ONLY JSON. No markdown. No extra text.
""".strip()

def extract_one(question_id: int, title:str, body:str) -> ExtractedQuestion:


    # temperature=0: most deterministic, consistent, less creative.
    # Higher values (e.g. 0.7, 1.0): more varied/creative, but less consistent.
    llm_kwargs = {"model": OPENAI_MODEL, "temperature": 0}
    if OPENAI_BASE_URL:
        llm_kwargs["base_url"] = OPENAI_BASE_URL
    llm = ChatOpenAI(**llm_kwargs)
    
    user_prompt = f"""
QuestionId: {question_id}

Title:
{title}

Body:
{body}
""".strip()
    
    msgs = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
        ]

    try:
        resp = llm.invoke(msgs).content.strip()
    except RateLimitError as e:
        raise RuntimeError(
            "OpenAI quota exceeded (HTTP 429 insufficient_quota). "
            "Check billing/credits at https://platform.openai.com/account/billing."
        ) from e
    except APIConnectionError as e:
        raise RuntimeError(
            "Could not connect to OpenAI API. Check internet/proxy/firewall settings and try again."
        ) from e

    # Sometimes models wrap JSON in text; a light cleanup is needed
    # (still stric, but practical)
    if resp.startswith("```"):
        resp = resp.strip("`")
        resp = resp.replace("json", "", 1).strip()

    try:
        data = json.loads(resp)
        obj = ExtractedQuestion.model_validate(data)
        return obj
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON. Got:\n{resp}\nError: {e}") from e
    except ValidationError as e:
        raise RuntimeError(f"JSON did not match schema. Got\n{resp}\nValidation error: {e}") from e
