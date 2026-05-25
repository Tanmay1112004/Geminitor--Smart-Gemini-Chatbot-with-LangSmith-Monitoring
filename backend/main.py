"""
main.py — Geminitor Pro FastAPI backend.
Serves the HTML/CSS/JS frontend as static files and exposes all /api/* routes.
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith (optional) ───────────────────────────────────────────────────────
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "Geminitor-Pro")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── In-memory session store ───────────────────────────────────────────────────
_sessions: dict = {}

def _session(sid: str) -> dict:
    if sid not in _sessions:
        _sessions[sid] = {
            "rag_chain": None,
            "analytics": {
                "total_messages": 0,
                "response_times": [],
                "token_counts": [],
                "topics": [],
                "start_time": datetime.now().isoformat(),
            },
        }
    return _sessions[sid]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Geminitor Pro", version="2.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def ok(data: dict) -> dict:
    return {"success": True, "data": data, "error": None}

def fail(msg: str, code: int = 500):
    raise HTTPException(status_code=code, detail={"success": False, "data": {}, "error": msg})

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: str = "gemini-2.5-flash"
    persona: str = "General AI"
    temperature: float = 0.7
    max_tokens: int = 2048
    history: list = []

class FeedbackRequest(BaseModel):
    message_index: int
    feedback: str

class ExportRequest(BaseModel):
    history: list = []
    format: str = "txt"

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0"}

# ── Chat (non-streaming) ──────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest, x_session_id: Optional[str] = Header(None)):
    sid = x_session_id or str(uuid.uuid4())
    sess = _session(sid)
    t0 = time.time()
    try:
        from backend.modules.chat_engine import get_response, get_followup
        response = get_response(req.model, req.temperature, req.max_tokens,
                                req.persona, req.history, req.message)
        elapsed = round(time.time() - t0, 2)
        tokens  = max(1, int(len(response.split()) * 1.35))
        follow_up = ""
        try:
            follow_up = get_followup(req.model, req.message, response)
        except Exception:
            pass
        an = sess["analytics"]
        an["total_messages"] += 1
        an["response_times"].append(elapsed)
        an["token_counts"].append(tokens)
        an["topics"].append(req.message[:100])
        return ok({"response": response, "follow_up": follow_up,
                   "response_time": elapsed, "tokens": tokens, "session_id": sid})
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Chat error: %s", exc)
        fail(str(exc))

# ── Chat streaming (SSE) ──────────────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, x_session_id: Optional[str] = Header(None)):
    sid  = x_session_id or str(uuid.uuid4())
    sess = _session(sid)
    t0   = time.time()

    from backend.modules.chat_engine import stream_response, get_followup

    async def generate():
        full = ""
        try:
            async for chunk in stream_response(req.model, req.temperature,
                                               req.max_tokens, req.persona,
                                               req.history, req.message):
                full += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return
        finally:
            elapsed = round(time.time() - t0, 2)
            tokens  = max(1, int(len(full.split()) * 1.35))
            follow_up = ""
            try:
                follow_up = get_followup(req.model, req.message, full)
            except Exception:
                pass
            an = sess["analytics"]
            an["total_messages"] += 1
            an["response_times"].append(elapsed)
            an["token_counts"].append(tokens)
            an["topics"].append(req.message[:100])
            yield f"data: {json.dumps({'done': True, 'response_time': elapsed, 'tokens': tokens, 'follow_up': follow_up, 'session_id': sid})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── PDF / TXT upload (RAG) ────────────────────────────────────────────────────
@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...),
                     x_session_id: Optional[str] = Header(None)):
    sid  = x_session_id or str(uuid.uuid4())
    sess = _session(sid)
    if not file.filename.lower().endswith((".pdf", ".txt")):
        fail("Only PDF and TXT files are accepted.", 400)
    try:
        from backend.modules.rag_module import process_document
        sess["rag_chain"] = await process_document(file)
        return ok({"message": f"'{file.filename}' indexed successfully.", "session_id": sid})
    except Exception as exc:
        log.error("RAG error: %s", exc)
        fail(str(exc))

# ── Image upload (Vision) ─────────────────────────────────────────────────────
@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...),
                       question: str = "Describe this image in detail.",
                       x_session_id: Optional[str] = Header(None)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        fail("Only JPG, PNG, and WebP images are accepted.", 400)
    try:
        from backend.modules.vision_module import analyze_image
        response = await analyze_image(file, question)
        return ok({"response": response, "filename": file.filename})
    except Exception as exc:
        log.error("Vision error: %s", exc)
        fail(str(exc))

# ── Analytics ─────────────────────────────────────────────────────────────────
@app.get("/api/analytics")
async def analytics(x_session_id: Optional[str] = Header(None)):
    sid = x_session_id or ""
    if not sid or sid not in _sessions:
        return ok({"total_messages": 0, "avg_response_time": 0,
                   "total_tokens": 0, "top_keywords": [], "recent_topics": []})
    from backend.modules.analytics_module import get_summary
    return ok(get_summary(_sessions[sid]["analytics"]))

# ── RAG query ─────────────────────────────────────────────────────────────────
@app.post("/api/rag/query")
async def rag_query(req: ChatRequest, x_session_id: Optional[str] = Header(None)):
    sid  = x_session_id or ""
    sess = _session(sid)
    chain = sess.get("rag_chain")
    if not chain:
        fail("No document uploaded for this session.", 400)
    try:
        t0       = time.time()
        response = chain.invoke(req.message)   # LCEL chain takes a plain string
        return ok({"response": str(response), "response_time": round(time.time()-t0, 2)})
    except Exception as exc:
        fail(str(exc))

# ── Export ────────────────────────────────────────────────────────────────────
@app.post("/api/export")
async def export_chat(req: ExportRequest):
    try:
        if req.format == "pdf":
            from backend.modules.export_module import export_to_pdf
            data = export_to_pdf(req.history)
            return StreamingResponse(iter([data]), media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=geminitor_chat.pdf"})
        else:
            from backend.modules.export_module import export_to_txt
            data = export_to_txt(req.history).encode()
            return StreamingResponse(iter([data]), media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=geminitor_chat.txt"})
    except Exception as exc:
        fail(str(exc))

# ── Feedback ──────────────────────────────────────────────────────────────────
@app.post("/api/feedback")
async def feedback(req: FeedbackRequest, x_session_id: Optional[str] = Header(None)):
    log.info("Feedback — session=%s index=%d type=%s", x_session_id, req.message_index, req.feedback)
    return ok({"message": "Feedback recorded."})

# ── Clear history ─────────────────────────────────────────────────────────────
@app.delete("/api/history")
async def clear_history(x_session_id: Optional[str] = Header(None)):
    if x_session_id and x_session_id in _sessions:
        _sessions[x_session_id]["rag_chain"] = None
    return ok({"message": "Session cleared."})

# ── Serve frontend (MUST be registered last) ──────────────────────────────────
_frontend = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
