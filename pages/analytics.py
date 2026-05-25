"""
analytics.py — Geminitor Pro analytics dashboard (Streamlit multipage).
Displays message counts, token usage, response times, and keyword frequency.
"""

import os
import streamlit as st

st.set_page_config(
    page_title="Analytics — Geminitor Pro",
    page_icon="📊",
    layout="wide",
)

# Load CSS
def _load_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)

_load_css()

# ── Back link ─────────────────────────────────────────────────────────────────
st.page_link("app.py", label="← Back to Chat", icon="💬")
st.markdown("---")

st.markdown(
    """
    <div class="geminitor-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>Real-time session metrics for your Geminitor Pro conversation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Guard ─────────────────────────────────────────────────────────────────────
if "analytics" not in st.session_state or st.session_state.analytics["total_messages"] == 0:
    st.info("📭 No data yet — head to the chat page and start a conversation!")
    st.stop()

from modules.analytics_module import (
    get_analytics_summary,
    response_time_chart,
    token_bar_chart,
    keyword_bar_chart,
)

analytics = st.session_state.analytics
data      = get_analytics_summary(analytics)

# ── KPI metrics ───────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("💬 Total Messages",   data["total_messages"])
k2.metric("🔢 Total Tokens",      f"{data['total_tokens']:,}")
k3.metric("⏱️ Avg Response Time", f"{data['avg_response_time']:.2f}s")
k4.metric("📝 Unique Prompts",    len(set(data["topics"])))

st.markdown("---")

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("### ⏱️ Response Time per Message")
    fig_rt = response_time_chart(data["response_times"])
    if fig_rt:
        st.plotly_chart(fig_rt, use_container_width=True)
    else:
        st.info("Not enough data yet.")

with c2:
    st.markdown("### 🔢 Token Usage per Response")
    fig_tok = token_bar_chart(data["token_history"])
    if fig_tok:
        st.plotly_chart(fig_tok, use_container_width=True)
    else:
        st.info("Not enough data yet.")

# ── Charts row 2 ──────────────────────────────────────────────────────────────
st.markdown("### 🔑 Top Keywords in Your Questions")
fig_kw = keyword_bar_chart(data["topics"])
if fig_kw:
    st.plotly_chart(fig_kw, use_container_width=True)
else:
    st.info("Not enough messages to extract keywords yet.")

st.markdown("---")

# ── Session timeline table ─────────────────────────────────────────────────────
st.markdown("### 📅 Session Timeline")
if st.session_state.get("chat_history"):
    import pandas as pd

    rows = []
    for i, msg in enumerate(st.session_state.chat_history):
        rows.append({
            "#":         i + 1,
            "Role":      msg["role"].capitalize(),
            "Preview":   (msg["content"][:90] + "…") if len(msg["content"]) > 90 else msg["content"],
            "Time":      msg.get("timestamp", "—"),
            "Resp(s)":   f"{msg['response_time']:.2f}" if msg.get("response_time") else "—",
            "Tokens":    msg.get("tokens", "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No messages in session yet.")
