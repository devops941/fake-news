"""Text cleaning helpers for the fake news classifier.

Two concerns live here:
1. Generic NLP cleaning (lowercase, strip URLs/punctuation/numbers, drop stopwords).
2. Removing the Reuters wire-service dateline that prefixes most REAL articles in the
   Kaggle dataset (e.g. "WASHINGTON (Reuters) - "). Left in place, the model learns to
   key off that tag instead of actual writing style and reports an inflated ~99% accuracy.
"""

import re

import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Matches a leading dateline like "WASHINGTON (Reuters) - " or
# "NEW YORK/LONDON (Reuters) -- ". The location is any run of uppercase words,
# optionally with a slash, immediately followed by "(Reuters)" and a dash separator.
REUTERS_PREFIX_PATTERN = re.compile(
    r"^\s*[A-Z][A-Za-z.\s]*(?:/[A-Z][A-Za-z.\s]*)?\(Reuters\)\s*-+\s*"
)


def remove_reuters_prefix(text: str) -> str:
    """Strip a leading '<DATELINE> (Reuters) - ' tag, if present.

    This is the data-leakage fix called out in the spec: without it, the classifier
    learns to associate the literal word "Reuters" with the REAL label.
    """
    if not isinstance(text, str):
        return text
    return REUTERS_PREFIX_PATTERN.sub("", text, count=1)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/digits, and remove English stopwords."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    words = [w for w in text.split(" ") if w and w not in STOPWORDS]
    return " ".join(words)


def preprocess(text: str) -> str:
    """Full pipeline used both at training time and at prediction time."""
    return clean_text(remove_reuters_prefix(text))
