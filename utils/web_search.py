"""Related-articles lookup: lets users cross-check a prediction against real search
results, the way you would on Google — reinforces that this is a screening aid, not
a final verdict."""

from ddgs import DDGS

MAX_QUERY_WORDS = 15


def build_query(headline: str, article: str) -> str:
    """Prefer the headline as the search query; fall back to the start of the article."""
    source = headline.strip() or article.strip()
    words = source.split()
    return " ".join(words[:MAX_QUERY_WORDS])


def search_related(query: str, max_results: int = 5):
    """Return [{title, url, snippet}, ...] for query, or [] if the search fails.

    Network/search errors shouldn't break the real/fake prediction itself, so any
    failure here is swallowed and reported as "no related results" to the caller.
    """
    if not query.strip():
        return []

    try:
        raw_results = DDGS().text(query, max_results=max_results)
    except Exception:
        return []

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw_results
        if r.get("href")
    ]
