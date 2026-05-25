"""
app.py — Geminitor Pro main entry point.
Dark-themed Streamlit chat UI with RAG, vision, export, and analytics.
"""

import os
import time
import streamlit as st
from datetime import datetime

# ── LangSmith tracing (optional) ──────────────────────────────────────────────
_ls_key = os.environ.get("LANGCHAIN_API_KEY", "")
if _ls_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "Geminitor-Pro")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geminitor Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
def _load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)

_load_css()

# ── Module imports ─────────────────────────────────────────────────────────────
from modules.chat_engine import get_chain, get_llm, get_followup_suggestion
from modules.export_module import export_chat_pdf, get_chat_summary

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "chat_history": [],
    "analytics": {
        "total_messages": 0,
        "total_tokens": 0,
        "response_times": [],
        "token_history": [],
        "topics": [],
    },
    "rag_enabled": False,
    "rag_chain": None,
    "image_data": None,
    "image_name": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model = st.selectbox(
        "🧠 Model",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
    )

    persona = st.selectbox(
        "🎭 Persona",
        ["General AI", "Code Assistant", "Medical Helper", "Study Buddy", "Creative Writer"],
        index=0,
    )

    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens  = st.slider("📏 Max Tokens",  256, 4096, 2048, 256)

    st.markdown("---")

    # ── RAG ───────────────────────────────────────────────────────────────────
    st.markdown("### 📄 Document Q&A (RAG)")
    doc_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], key="doc_upload")
    if doc_file and not st.session_state.rag_enabled:
        with st.spinner("Indexing document…"):
            try:
                from modules.rag_module import process_document
                st.session_state.rag_chain   = process_document(doc_file, model)
                st.session_state.rag_enabled = True
                st.success(f"✅ **{doc_file.name}** loaded!")
            except Exception as exc:
                st.error(f"RAG error: {exc}")

    if st.session_state.rag_enabled:
        st.info("📄 Document mode active")
        if st.button("❌ Remove Document"):
            st.session_state.rag_enabled = False
            st.session_state.rag_chain   = None
            st.rerun()

    st.markdown("---")

    # ── Image upload ──────────────────────────────────────────────────────────
    st.markdown("### 🖼️ Image Analysis")
    img_file = st.file_uploader(
        "Upload image (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"], key="img_upload"
    )
    if img_file:
        st.session_state.image_data = img_file
        st.session_state.image_name = img_file.name
        st.image(img_file, use_container_width=True)
        if st.button("❌ Remove Image"):
            st.session_state.image_data = None
            st.session_state.image_name = ""
            st.rerun()
    elif not img_file and st.session_state.image_data:
        st.session_state.image_data = None
        st.session_state.image_name = ""

    st.markdown("---")

    # ── Session history preview ───────────────────────────────────────────────
    st.markdown("### 💬 Session History")
    user_msgs = [m for m in st.session_state.chat_history if m["role"] == "user"]
    if user_msgs:
        for msg in user_msgs[-8:]:
            preview = msg["content"][:35] + "…" if len(msg["content"]) > 35 else msg["content"]
            st.caption(f"• {preview}")
    else:
        st.caption("No messages yet.")

    st.markdown("---")

    col_clr, col_mode = st.columns(2)
    with col_clr:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            for k in ("total_messages", "total_tokens"):
                st.session_state.analytics[k] = 0
            for k in ("response_times", "token_history", "topics"):
                st.session_state.analytics[k] = []
            st.rerun()
    with col_mode:
        st.page_link("pages/analytics.py", label="📊 Analytics", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="geminitor-header">
        <h1>🤖 Geminitor Pro</h1>
        <p>Powered by Google Gemini 2.5 Flash · LangChain · RAG · Vision · Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Active mode badges
badge_cols = st.columns(4)
with badge_cols[0]:
    st.markdown(f"**Model:** `{model}`")
with badge_cols[1]:
    st.markdown(f"**Persona:** `{persona}`")
with badge_cols[2]:
    rag_badge = "🟢 RAG On" if st.session_state.rag_enabled else "⚪ RAG Off"
    st.markdown(rag_badge)
with badge_cols[3]:
    vis_badge = f"🖼️ {st.session_state.image_name}" if st.session_state.image_data else "⚪ No Image"
    st.markdown(vis_badge)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
for idx, msg in enumerate(st.session_state.chat_history):
    role    = msg["role"]
    content = msg["content"]
    ts      = msg.get("timestamp", "")

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
            st.caption(f"🕐 {ts}")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)

            meta1, meta2, meta3, meta4 = st.columns([2, 2, 3, 1])
            with meta1:
                if msg.get("response_time"):
                    st.caption(f"⏱️ {msg['response_time']:.2f}s")
            with meta2:
                if msg.get("tokens"):
                    st.caption(f"🔢 ~{msg['tokens']} tokens")
            with meta3:
                st.caption(f"🕐 {ts}")
            with meta4:
                st.button("📋", key=f"copy_{idx}", help="Copy to clipboard")

            if msg.get("follow_up"):
                st.info(f"💡 **Suggested follow-up:** _{msg['follow_up']}_")

            fb1, fb2, _ = st.columns([1, 1, 8])
            with fb1:
                if st.button("👍", key=f"up_{idx}"):
                    st.toast("Thanks for the positive feedback!")
            with fb2:
                if st.button("👎", key=f"dn_{idx}"):
                    st.toast("Thanks — we'll keep improving!")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT TOOLBAR  (only when there are messages)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.chat_history:
    st.markdown("---")
    exp1, exp2 = st.columns(2)

    with exp1:
        if st.button("📥 Export Chat as PDF", use_container_width=True):
            with st.spinner("Generating PDF…"):
                try:
                    pdf_bytes = export_chat_pdf(st.session_state.chat_history)
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"geminitor_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"PDF export failed: {exc}")

    with exp2:
        if st.button("📝 Summarise Conversation", use_container_width=True):
            with st.spinner("Summarising…"):
                try:
                    llm     = get_llm(model, temperature)
                    summary = get_chat_summary(st.session_state.chat_history, llm)
                    with st.expander("📋 Conversation Summary", expanded=True):
                        st.markdown(summary)
                except Exception as exc:
                    st.error(f"Summary failed: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT & RESPONSE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
user_input = st.chat_input("Type your message here…")

if user_input:
    ts_now = datetime.now().strftime("%H:%M:%S")

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "timestamp": ts_now}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
        st.caption(f"🕐 {ts_now}")

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_ ⏳")
        start_time = time.time()

        response   = ""
        follow_up  = ""
        tokens_est = 0

        try:
            MAX_RETRIES = 3

            # ── Choose response pathway ────────────────────────────────────
            if st.session_state.rag_enabled and st.session_state.rag_chain:
                for attempt in range(MAX_RETRIES):
                    try:
                        raw = st.session_state.rag_chain.invoke({"query": user_input})
                        response = raw.get("result", str(raw)) if isinstance(raw, dict) else str(raw)
                        break
                    except Exception as exc:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        time.sleep(2 ** attempt)

            elif st.session_state.image_data:
                from modules.vision_module import analyze_image
                for attempt in range(MAX_RETRIES):
                    try:
                        response = analyze_image(
                            st.session_state.image_data, user_input, model
                        )
                        break
                    except Exception as exc:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        time.sleep(2 ** attempt)

            else:
                chain = get_chain(
                    model, temperature, max_tokens, persona,
                    st.session_state.chat_history[:-1],
                )
                for attempt in range(MAX_RETRIES):
                    try:
                        response = chain.invoke({"question": user_input})
                        break
                    except Exception as exc:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        time.sleep(2 ** attempt)

            elapsed    = round(time.time() - start_time, 2)
            tokens_est = max(1, int(len(response.split()) * 1.35))

            placeholder.markdown(response)

            # ── Follow-up suggestion ───────────────────────────────────────
            try:
                llm       = get_llm(model, min(temperature + 0.1, 1.0))
                follow_up = get_followup_suggestion(user_input, response, llm)
                if follow_up:
                    st.info(f"💡 **Suggested follow-up:** _{follow_up}_")
            except Exception:
                pass

            # ── Metadata row ───────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns([2, 2, 3, 1])
            with m1:
                st.caption(f"⏱️ {elapsed}s")
            with m2:
                st.caption(f"🔢 ~{tokens_est} tokens")
            with m3:
                st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
            with m4:
                st.button("📋", key="copy_new", help="Copy response")

            # ── Feedback buttons ───────────────────────────────────────────
            fb1, fb2, _ = st.columns([1, 1, 8])
            with fb1:
                if st.button("👍", key="up_new"):
                    st.toast("Thanks for the positive feedback!")
            with fb2:
                if st.button("👎", key="dn_new"):
                    st.toast("Thanks — we'll keep improving!")

            # ── Persist to history & analytics ────────────────────────────
            st.session_state.chat_history.append({
                "role":          "assistant",
                "content":       response,
                "timestamp":     datetime.now().strftime("%H:%M:%S"),
                "response_time": elapsed,
                "tokens":        tokens_est,
                "follow_up":     follow_up,
            })

            an = st.session_state.analytics
            an["total_messages"] += 1
            an["total_tokens"]   += tokens_est
            an["response_times"].append(elapsed)
            an["token_history"].append(tokens_est)
            an["topics"].append(user_input)

        except Exception as exc:
            placeholder.error(f"❌ Error: {exc}")
