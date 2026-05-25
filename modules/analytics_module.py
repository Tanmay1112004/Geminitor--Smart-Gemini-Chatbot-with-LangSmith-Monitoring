"""
analytics_module.py — Session analytics helpers: data aggregation and Plotly charts.
"""

import re
from collections import Counter


def get_analytics_summary(analytics: dict) -> dict:
    """
    Compute summary statistics from the session analytics dict.

    Returns a dict with total_messages, total_tokens, avg_response_time, etc.
    """
    times = analytics.get("response_times", [])
    return {
        "total_messages": analytics.get("total_messages", 0),
        "total_tokens": analytics.get("total_tokens", 0),
        "avg_response_time": round(sum(times) / len(times), 2) if times else 0.0,
        "response_times": times,
        "topics": analytics.get("topics", []),
        "token_history": analytics.get("token_history", []),
    }


def response_time_chart(response_times: list):
    """Return a Plotly line chart of response times."""
    import plotly.graph_objects as go

    if not response_times:
        return None

    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(response_times) + 1)),
            y=response_times,
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=6, color="#79c0ff"),
            name="Response Time",
        )
    )
    fig.update_layout(
        title="Response Time per Message",
        xaxis_title="Message #",
        yaxis_title="Seconds",
        template="plotly_dark",
        paper_bgcolor="rgba(13,17,23,0.8)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        height=300,
    )
    return fig


def token_bar_chart(token_history: list):
    """Return a Plotly bar chart of token usage over time."""
    import plotly.graph_objects as go

    if not token_history:
        return None

    fig = go.Figure(
        go.Bar(
            x=list(range(1, len(token_history) + 1)),
            y=token_history,
            marker_color="#388bfd",
            name="Tokens",
        )
    )
    fig.update_layout(
        title="Token Usage per Response",
        xaxis_title="Response #",
        yaxis_title="Tokens (approx.)",
        template="plotly_dark",
        paper_bgcolor="rgba(13,17,23,0.8)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        height=300,
    )
    return fig


def keyword_bar_chart(topics: list):
    """Return a Plotly horizontal bar chart of the top keywords from user messages."""
    import plotly.graph_objects as go

    if not topics:
        return None

    stop_words = {
        "what", "this", "that", "with", "from", "have", "been", "will",
        "your", "about", "just", "more", "some", "there", "their", "then",
        "than", "when", "where", "which", "into", "also", "very", "much",
        "does", "only", "over", "such", "make", "like", "know", "tell",
        "give", "help", "want", "need", "please", "okay", "thanks",
    }

    all_text = " ".join(topics).lower()
    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    words = [w for w in words if w not in stop_words]

    if not words:
        return None

    counts = Counter(words).most_common(15)
    labels = [c[0] for c in counts]
    values = [c[1] for c in counts]

    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color="#3fb950",
        )
    )
    fig.update_layout(
        title="Top Keywords in Your Questions",
        xaxis_title="Frequency",
        template="plotly_dark",
        paper_bgcolor="rgba(13,17,23,0.8)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        height=350,
    )
    return fig
