"""AI Fact-Checking Service with Multi-Provider Priority & Fallback.

Priority:
  1. Anthropic Claude (claude-haiku-4-5-20251001 or model in .env)
  2. Groq (openai/gpt-oss-120b or model in .env)
  3. Graceful fallback if both AI providers are offline/unavailable.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Ensure .env is loaded
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env():
    """Load key-value pairs from .env into os.environ if not already present."""
    if ENV_FILE.exists():
        try:
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val
        except Exception as e:
            logger.warning(f"Error reading .env file: {e}")


load_env()


def clean_json_str(raw: str) -> Dict[str, Any]:
    """Extract and parse clean JSON from an LLM response string."""
    raw = raw.strip()
    # Strip markdown code blocks if present
    if "```" in raw:
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Search for matching curly braces
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def call_anthropic(text: str) -> Dict[str, Any]:
    """Call Anthropic Claude API for deep factual news verification."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured in .env")

    system_prompt = (
        "You are an elite fact-checker and investigative journalist. "
        "Analyze the given news headline/article/claim. Determine if it is factually REAL or FAKE.\n"
        "Guidelines:\n"
        "- If a claim misattributes a book, poem, policy, or discovery to the wrong person, it is FAKE.\n"
        "- If it is a fabricated conspiracy, hoax, or misleading rumor, it is FAKE.\n"
        "- If it reports historically or factually accurate events, authors, or verified news, it is REAL.\n"
        "Output ONLY a raw JSON object with keys:\n"
        "  'verdict': 'REAL' or 'FAKE',\n"
        "  'confidence': integer between 60 and 99,\n"
        "  'summary': concise 2-3 sentence factual explanation specifying the true facts/authors/context,\n"
        "  'details': array of 2 to 4 bullet strings citing specific historical or journalistic facts."
    )

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        data=json.dumps({
            "model": model,
            "max_tokens": 800,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": f"Fact-check this news headline/claim/article:\n\n\"\"\"\n{text}\n\"\"\"",
                }
            ],
        }).encode("utf-8"),
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=12) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        raw_content = data["content"][0]["text"]
        result = clean_json_str(raw_content)
        result["provider"] = f"Anthropic Claude ({model})"
        result["verdict"] = str(result.get("verdict", "FAKE")).strip().upper()
        if result["verdict"] not in ("REAL", "FAKE"):
            result["verdict"] = "FAKE" if "fake" in str(result["verdict"]).lower() else "REAL"
        result["confidence"] = float(result.get("confidence", 85))
        result["success"] = True
        return result


def call_groq(text: str) -> Dict[str, Any]:
    """Call Groq API (Priority 2 / Fallback) with high-speed inference."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured in .env")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an elite fact-checker and investigative journalist. "
                        "Determine if the submitted statement, news headline, or article is REAL or FAKE.\n"
                        "Rules:\n"
                        "- If an attribution is wrong (e.g. attributing a poem/book/quote to the wrong author), it is FAKE.\n"
                        "- If it is a hoax, rumor, satire, or false news, it is FAKE.\n"
                        "- If it is verified, accurate news, it is REAL.\n"
                        "Return a JSON object with:\n"
                        "  'verdict': 'REAL' or 'FAKE',\n"
                        "  'confidence': integer (60-99),\n"
                        "  'summary': 2-3 sentence factual explanation with true facts and corrections,\n"
                        "  'details': list of 2-4 bullet point strings with supporting evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Fact-check this news headline/claim/article:\n\n\"\"\"\n{text}\n\"\"\"",
                },
            ],
            "response_format": {"type": "json_object"},
        }).encode("utf-8"),
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=12) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        raw_content = data["choices"][0]["message"]["content"]
        result = clean_json_str(raw_content)
        result["provider"] = f"Groq ({model})"
        result["verdict"] = str(result.get("verdict", "FAKE")).strip().upper()
        if result["verdict"] not in ("REAL", "FAKE"):
            result["verdict"] = "FAKE" if "fake" in str(result["verdict"]).lower() else "REAL"
        result["confidence"] = float(result.get("confidence", 85))
        result["success"] = True
        return result


def fact_check_with_ai(text: str) -> Dict[str, Any]:
    """Main verification entry point.

    Attempts Anthropic (Priority 1). If it fails or quota exhausted,
    falls back seamlessly to Groq (Priority 2).
    """
    load_env()  # reload in case .env was modified
    errors: List[str] = []

    # 1. First Priority: Anthropic
    try:
        result = call_anthropic(text)
        return result
    except Exception as e:
        logger.warning(f"Anthropic verification failed, falling back to Groq: {e}")
        errors.append(f"Anthropic Claude: {e}")

    # 2. Second Priority: Groq
    try:
        result = call_groq(text)
        return result
    except Exception as e:
        logger.warning(f"Groq verification failed: {e}")
        errors.append(f"Groq: {e}")

    # 3. Both failed
    return {
        "verdict": None,
        "confidence": 0.0,
        "summary": "AI fact-checking service was temporarily unreachable. Showing traditional Machine Learning classification.",
        "details": errors,
        "provider": "Offline / Fallback",
        "success": False,
        "error_message": "; ".join(errors),
    }
