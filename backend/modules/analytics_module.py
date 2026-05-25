"""
analytics_module.py — Per-session analytics aggregation helpers.
"""

from collections import Counter
import re


def get_summary(analytics: dict) -> dict:
    """Compute summary stats from a session analytics dict."""
    times  = analytics.get("response_times", [])
    tokens = analytics.get("token_counts", [])
    topics = analytics.get("topics", [])

    avg_time = round(sum(times) / len(times), 2) if times else 0.0

    stop = {
        "what", "this", "that", "with", "from", "have", "been", "will",
        "your", "about", "just", "more", "some", "there", "their", "then",
        "than", "when", "where", "which", "into", "also", "very", "much",
        "does", "only", "over", "such", "make", "like", "know", "tell",
        "give", "help", "want", "need", "please", "okay", "thanks", "can",
        "could", "would", "should", "write", "show", "explain",
    }
    words = re.findall(r"\b[a-z]{4,}\b", " ".join(topics).lower())
    top_words = [w for w, _ in Counter(w for w in words if w not in stop).most_common(10)]

    return {
        "total_messages":   analytics.get("total_messages", 0),
        "avg_response_time": avg_time,
        "total_tokens":     sum(tokens),
        "token_history":    tokens[-20:],
        "response_times":   times[-20:],
        "top_keywords":     top_words,
        "recent_topics":    topics[-8:],
        "start_time":       analytics.get("start_time", ""),
    }
