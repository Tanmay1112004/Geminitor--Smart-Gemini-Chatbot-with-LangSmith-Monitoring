# 🤖 Geminitor Pro

A production-grade AI chat application powered by **Google Gemini 2.5 Flash** and **LangChain**, with a ChatGPT-style dark UI built in pure HTML/CSS/JS and a **FastAPI** backend with real-time SSE streaming.

---

## Demo Images

![demo](https://github.com/Tanmay1112004/Geminitor--Smart-Gemini-Chatbot-with-LangSmith-Monitoring/blob/main/demo_app_screenshots/Screenshot_3-6-2026_184029_79ee36a2-bae9-456b-aace-34eb5de20832-00-1y86fvg5ujd54.pike.replit.dev.jpeg)

![demo](https://github.com/Tanmay1112004/Geminitor--Smart-Gemini-Chatbot-with-LangSmith-Monitoring/blob/main/demo_app_screenshots/Screenshot_3-6-2026_184213_79ee36a2-bae9-456b-aace-34eb5de20832-00-1y86fvg5ujd54.pike.replit.dev.jpeg)

![demo](https://github.com/Tanmay1112004/Geminitor--Smart-Gemini-Chatbot-with-LangSmith-Monitoring/blob/main/demo_app_screenshots/Screenshot_3-6-2026_185856_geminitor-smart-gemini-chatbot-with-lang-smith--kshirsagarrutuj.replit.app.jpeg)

![demo](https://github.com/Tanmay1112004/Geminitor--Smart-Gemini-Chatbot-with-LangSmith-Monitoring/blob/main/demo_app_screenshots/Screenshot_3-6-2026_19121_geminitor-smart-gemini-chatbot-with-lang-smith--kshirsagarrutuj.replit.app.jpeg)

---


## ✨ Features

- ⚡ **Real-time streaming** — responses appear word-by-word via Server-Sent Events
- 🧠 **Multi-turn memory** — full conversation context sent with every request
- 🎭 **5 AI personas** — General AI, Code Assistant, Medical Helper, Study Buddy, Creative Writer
- 📄 **Document Q&A (RAG)** — upload a PDF or TXT and ask questions about it (FAISS + Gemini Embeddings)
- 🖼️ **Vision** — upload an image and ask Gemini to analyze it
- 📊 **Session analytics** — message count, avg response time, token usage, recent topics
- 📥 **Export chat** — download conversation as `.txt` or `.pdf`
- 🌙 **Dark / Light mode** — persisted to localStorage
- 📱 **Responsive** — mobile-friendly with collapsible sidebar

---

## 🏗️ Architecture

```
Geminitor Pro
├── backend/                   # FastAPI backend
│   ├── main.py                # All API routes + static file serving
│   └── modules/
│       ├── chat_engine.py     # LangChain + Gemini streaming chain
│       ├── rag_module.py      # PDF/TXT → FAISS → RAG (LCEL)
│       ├── vision_module.py   # Gemini Vision (image analysis)
│       ├── analytics_module.py
│       └── export_module.py   # Chat → PDF / TXT export
└── frontend/                  # Pure HTML/CSS/JS (no framework)
    ├── index.html
    ├── style.css
    ├── app.js
    └── config.js
```

FastAPI serves both the API (`/api/*`) and the frontend static files (`/`) from a single server on port 5000.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Tanmay1112004/geminitor-pro.git
cd geminitor-pro
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_google_gemini_api_key

# Optional — LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Geminitor-Pro
```

> Get a free Google Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 4. Run the app

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/stream` | SSE streaming chat response |
| `POST` | `/api/chat` | Non-streaming chat response |
| `POST` | `/api/upload/pdf` | Index a PDF or TXT file for RAG |
| `POST` | `/api/upload/image` | Analyze an image with Gemini Vision |
| `GET`  | `/api/analytics` | Session analytics stats |
| `POST` | `/api/export` | Export chat as `.txt` or `.pdf` |
| `POST` | `/api/feedback` | Submit thumbs up/down feedback |
| `GET`  | `/health` | Health check |

Interactive docs available at `/api/docs` (Swagger UI).

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn |
| AI / LLM | Google Gemini 2.5 Flash via LangChain |
| RAG | FAISS + Gemini Embeddings (LCEL pipeline) |
| Vision | Gemini Vision multimodal API |
| Frontend | Vanilla HTML / CSS / JS |
| Streaming | Server-Sent Events (SSE) via `fetch` ReadableStream |
| Export | fpdf2 |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Google Gemini API key |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing (optional) |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable tracing |

---

## 📄 License

MIT

---

Built by Tanmay 🚀
