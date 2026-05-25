/**
 * app.js — Geminitor Pro frontend logic.
 * Vanilla JS, ES6+. No jQuery, no React.
 * Uses fetch() ReadableStream for SSE-style streaming.
 */

/* ── State ─────────────────────────────────────────────────────────────── */
let sessionId      = "";
let chatHistory    = [];   // [{role, content, timestamp}]
let chatTitles     = [];   // sidebar history labels
let isStreaming    = false;
let ragActive      = false;
let pendingImage   = null; // {file, question}
let settings       = {};

/* ── Init ──────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  sessionId = getOrCreateSessionId();
  loadSettings();
  applySettings();
  updateModelBadge();
});

function getOrCreateSessionId() {
  let sid = localStorage.getItem("geminitor_session");
  if (!sid) { sid = crypto.randomUUID(); localStorage.setItem("geminitor_session", sid); }
  return sid;
}

/* ── Settings (localStorage) ───────────────────────────────────────────── */
function loadSettings() {
  const stored = localStorage.getItem("geminitor_settings");
  settings = stored ? JSON.parse(stored) : {};
  const { model = "gemini-2.5-flash", persona = "General AI", temperature = 0.7, max_tokens = 2048, theme = "dark" } = settings;

  document.getElementById("model-select").value   = model;
  document.getElementById("persona-select").value = persona;
  document.getElementById("temp-range").value     = temperature;
  document.getElementById("tokens-range").value   = max_tokens;
  document.getElementById("temp-val").textContent   = temperature;
  document.getElementById("tokens-val").textContent = max_tokens;

  document.body.className = theme === "light" ? "light" : "dark";
  document.getElementById("theme-btn").textContent = theme === "light" ? "☀️" : "🌙";
}

function saveSettings() {
  settings = {
    model:       document.getElementById("model-select").value,
    persona:     document.getElementById("persona-select").value,
    temperature: parseFloat(document.getElementById("temp-range").value),
    max_tokens:  parseInt(document.getElementById("tokens-range").value),
    theme:       document.body.classList.contains("light") ? "light" : "dark",
  };
  localStorage.setItem("geminitor_settings", JSON.stringify(settings));
  updateModelBadge();
}

function applySettings() { saveSettings(); }

function updateSlider(type) {
  if (type === "temp") {
    document.getElementById("temp-val").textContent = document.getElementById("temp-range").value;
  } else {
    document.getElementById("tokens-val").textContent = document.getElementById("tokens-range").value;
  }
  saveSettings();
}

function updateModelBadge() {
  const m = document.getElementById("model-select").value;
  document.getElementById("model-badge-header").textContent = m;
}

/* ── Input handling ────────────────────────────────────────────────────── */
function onInputChange() {
  const ta  = document.getElementById("msg-input");
  const btn = document.getElementById("send-btn");
  const cc  = document.getElementById("char-count");

  // Auto-resize
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 140) + "px";

  const len = ta.value.length;
  cc.textContent = len;
  cc.style.color = len > 3800 ? "#e57373" : "var(--subtext)";
  btn.disabled   = len === 0 || isStreaming;
  closeAttachMenu();
}

function onKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
}

function handleSend() {
  if (isStreaming) return;

  // Image pending — route to vision endpoint
  if (pendingImage) {
    const ta       = document.getElementById("msg-input");
    const question = ta.value.trim() || "Describe this image in detail.";
    ta.value = ""; ta.style.height = "auto";
    document.getElementById("char-count").textContent = "0";
    document.getElementById("send-btn").disabled = true;
    sendImageMessage(pendingImage.file, question);
    clearImagePreview();
    return;
  }

  const ta   = document.getElementById("msg-input");
  const text = ta.value.trim();
  if (!text) return;
  ta.value = ""; ta.style.height = "auto";
  document.getElementById("char-count").textContent = "0";
  document.getElementById("send-btn").disabled = true;
  sendMessage(text);
}

function sendSuggestion(text) {
  document.getElementById("msg-input").value = text;
  onInputChange();
  handleSend();
}

/* ── Core send / stream ────────────────────────────────────────────────── */
async function sendMessage(text) {
  hideEmpty();
  isStreaming = true;
  saveSettings();

  // Add user message to UI and history
  const ts = formatTime(new Date());
  appendUserMessage(text, ts);
  chatHistory.push({ role: "user", content: text, timestamp: ts });
  updateChatHistoryList(text);

  // Typing indicator
  showTypingIndicator();

  const payload = {
    message:     text,
    model:       settings.model,
    persona:     settings.persona,
    temperature: settings.temperature,
    max_tokens:  settings.max_tokens,
    history:     chatHistory.slice(-20),
  };
  const headers = { "Content-Type": "application/json", "X-Session-ID": sessionId };

  const endpoint = ragActive ? "/api/rag/query" : "/api/chat/stream";

  try {
    if (ragActive) {
      // RAG: non-streaming
      const res  = await fetch(endpoint, { method: "POST", headers, body: JSON.stringify(payload) });
      const json = await res.json();
      hideTypingIndicator();
      if (!json.success) throw new Error(json.error || "RAG error");
      const { response, response_time } = json.data;
      const botTs = formatTime(new Date());
      appendBotMessage(response, { response_time, tokens: null, follow_up: "", ts: botTs });
      chatHistory.push({ role: "assistant", content: response, timestamp: botTs });
    } else {
      // Streaming via fetch ReadableStream
      const res = await fetch(endpoint, { method: "POST", headers, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      hideTypingIndicator();

      const { el, contentEl } = createBotMessageEl();
      let rawText  = "";
      let metadata = {};
      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const evt = JSON.parse(raw);
            if (evt.error) throw new Error(evt.error);
            if (evt.chunk) {
              rawText += evt.chunk;
              contentEl.innerHTML = renderMarkdown(rawText);
              hljs.highlightAll();
              scrollToBottom();
            }
            if (evt.done) { metadata = evt; }
          } catch (e) { /* skip bad lines */ }
        }
      }

      // Finalize
      const botTs = formatTime(new Date());
      finalizeBotMessage(el, contentEl, rawText, { ...metadata, ts: botTs });
      chatHistory.push({ role: "assistant", content: rawText, timestamp: botTs });
    }
  } catch (err) {
    hideTypingIndicator();
    appendErrorMessage(err.message);
  }

  isStreaming = false;
  document.getElementById("send-btn").disabled =
    document.getElementById("msg-input").value.trim() === "";
}

/* ── Image message via Vision endpoint ─────────────────────────────────── */
async function sendImageMessage(file, question) {
  hideEmpty();
  isStreaming = true;

  const ts = formatTime(new Date());
  // Show user message with thumbnail
  const msgs = document.getElementById("messages");
  const userDiv = document.createElement("div");
  userDiv.className = "message";
  userDiv.innerHTML = `
    <div class="msg-inner user-inner">
      <div class="avatar user-avatar">${getInitial()}</div>
      <div class="msg-content">
        <div class="msg-role">You</div>
        <div class="user-bubble">
          <img src="${URL.createObjectURL(file)}" style="max-width:180px;max-height:120px;border-radius:8px;display:block;margin-bottom:6px;" />
          ${escapeHtml(question)}
        </div>
        <div class="msg-meta">${ts}</div>
      </div>
    </div>`;
  msgs.appendChild(userDiv);
  chatHistory.push({ role: "user", content: `[Image: ${file.name}] ${question}`, timestamp: ts });
  updateChatHistoryList(`[Image] ${question}`);
  scrollToBottom();

  showTypingIndicator();
  const form = new FormData();
  form.append("file", file);
  try {
    const res  = await fetch(`/api/upload/image?question=${encodeURIComponent(question)}`,
                              { method: "POST", headers: { "X-Session-ID": sessionId }, body: form });
    const json = await res.json();
    if (!json.success) throw new Error(json.detail?.error || json.error || "Vision error");
    hideTypingIndicator();
    const botTs = formatTime(new Date());
    appendBotMessage(json.data.response, { ts: botTs });
    chatHistory.push({ role: "assistant", content: json.data.response, timestamp: botTs });
  } catch (err) {
    hideTypingIndicator();
    appendErrorMessage(err.message);
  }

  isStreaming = false;
  document.getElementById("send-btn").disabled =
    document.getElementById("msg-input").value.trim() === "";
}

/* ── Message DOM helpers ───────────────────────────────────────────────── */
function hideEmpty() {
  const es = document.getElementById("empty-state");
  if (es) es.style.display = "none";
}

function appendUserMessage(text, ts) {
  const msgs = document.getElementById("messages");
  const div  = document.createElement("div");
  div.className = "message";
  div.innerHTML = `
    <div class="msg-inner user-inner">
      <div class="avatar user-avatar">${getInitial()}</div>
      <div class="msg-content">
        <div class="msg-role">You</div>
        <div class="user-bubble">${escapeHtml(text)}</div>
        <div class="msg-meta">${ts}</div>
      </div>
    </div>`;
  msgs.appendChild(div);
  scrollToBottom();
}

function createBotMessageEl() {
  const msgs    = document.getElementById("messages");
  const div     = document.createElement("div");
  div.className = "message";
  div.innerHTML = `
    <div class="msg-inner bot-inner">
      <div class="avatar bot-avatar">🤖</div>
      <div class="msg-content">
        <div class="msg-role">Geminitor</div>
        <div class="bot-bubble" id="streaming-content"></div>
        <div class="msg-meta bot-meta" id="streaming-meta"></div>
      </div>
    </div>`;
  msgs.appendChild(div);
  scrollToBottom();
  return { el: div, contentEl: div.querySelector("#streaming-content") };
}

function finalizeBotMessage(el, contentEl, rawText, meta) {
  contentEl.removeAttribute("id");
  contentEl.innerHTML = renderMarkdown(rawText);
  hljs.highlightAll();

  const metaEl = el.querySelector("#streaming-meta");
  if (metaEl) { metaEl.removeAttribute("id"); }

  const timeStr  = meta.response_time ? `⏱️ ${meta.response_time}s` : "";
  const tokenStr = meta.tokens        ? `🔢 ~${meta.tokens} tokens` : "";
  const ts       = meta.ts            ? meta.ts : "";

  const copyId = `copy-${Date.now()}`;
  if (metaEl) {
    metaEl.innerHTML = `
      ${timeStr} ${tokenStr} ${ts}
      <button class="copy-btn" id="${copyId}" onclick="copyText(this, event)">📋 Copy</button>
      <button class="feedback-btn" onclick="sendFeedback(this,'positive')" title="Good response">👍</button>
      <button class="feedback-btn" onclick="sendFeedback(this,'negative')" title="Bad response">👎</button>`;
    // Store raw text for copy
    const copyBtn = document.getElementById(copyId);
    if (copyBtn) copyBtn.dataset.raw = rawText;
  }

  // Follow-up chip
  if (meta.follow_up) {
    const chip    = document.createElement("div");
    chip.className = "follow-up-chip";
    chip.textContent = `💡 ${meta.follow_up}`;
    chip.onclick   = () => sendSuggestion(meta.follow_up);
    contentEl.parentElement.appendChild(chip);
  }

  scrollToBottom();
}

function appendBotMessage(text, meta) {
  const { el, contentEl } = createBotMessageEl();
  finalizeBotMessage(el, contentEl, text, meta);
}

function appendErrorMessage(msg) {
  const msgs = document.getElementById("messages");
  const div  = document.createElement("div");
  div.className = "message";
  div.innerHTML = `
    <div class="msg-inner bot-inner">
      <div class="avatar bot-avatar">🤖</div>
      <div class="msg-content">
        <div class="bot-bubble" style="color:#e57373;border-left-color:#e57373">
          ❌ Error: ${escapeHtml(msg)}
        </div>
      </div>
    </div>`;
  msgs.appendChild(div);
  scrollToBottom();
}

function showTypingIndicator() {
  let ind = document.getElementById("typing-indicator");
  if (!ind) {
    ind = document.createElement("div");
    ind.id = "typing-indicator";
    ind.innerHTML = `
      <div class="typing-inner">
        <div class="avatar bot-avatar">🤖</div>
        <div class="typing-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
      </div>`;
    document.getElementById("messages").appendChild(ind);
  }
  ind.style.display = "block";
  scrollToBottom();
}

function hideTypingIndicator() {
  const ind = document.getElementById("typing-indicator");
  if (ind) ind.style.display = "none";
}

/* ── Sidebar history ───────────────────────────────────────────────────── */
function updateChatHistoryList(text) {
  const title = text.length > 36 ? text.slice(0, 36) + "…" : text;
  chatTitles.unshift(title);
  const list = document.getElementById("chat-history-list");
  list.innerHTML = chatTitles.slice(0, 12).map(t =>
    `<div class="history-item" title="${escapeHtml(t)}">${escapeHtml(t)}</div>`
  ).join("");
}

/* ── Actions ───────────────────────────────────────────────────────────── */
function newChat() {
  chatHistory = [];
  chatTitles  = [];
  document.getElementById("messages").innerHTML = "";
  document.getElementById("chat-history-list").innerHTML = '<p class="empty-history">No history yet</p>';
  const es = document.getElementById("empty-state");
  if (es) es.style.display = "flex";
  ragActive   = false;
  pendingImage = null;
  updateRagBadge(false);
  closeSidebar();
}

function clearChat() {
  if (!confirm("Clear this conversation?")) return;
  newChat();
  fetch("/api/history", { method: "DELETE", headers: { "X-Session-ID": sessionId } });
}

async function exportChat(format) {
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: chatHistory, format }),
    });
    if (!res.ok) throw new Error("Export failed");
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `geminitor_chat.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) { alert("Export error: " + e.message); }
}

/* ── Theme ─────────────────────────────────────────────────────────────── */
function toggleTheme() {
  const isLight = document.body.classList.toggle("light");
  document.body.classList.toggle("dark", !isLight);
  document.getElementById("theme-btn").textContent = isLight ? "☀️" : "🌙";
  saveSettings();
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
  document.getElementById("overlay").classList.toggle("active");
}
function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("overlay").classList.remove("active");
}

/* ── File uploads ──────────────────────────────────────────────────────── */
function triggerPdfUpload()   { document.getElementById("pdf-input").click();   closeAttachMenu(); }
function triggerImageUpload() { document.getElementById("image-input").click(); closeAttachMenu(); }

function toggleAttachMenu() {
  document.getElementById("attach-menu").classList.toggle("hidden");
}
function closeAttachMenu() {
  document.getElementById("attach-menu").classList.add("hidden");
}

async function onPdfSelected(input) {
  const file = input.files[0];
  if (!file) return;
  input.value = "";

  // Show loading state in banner immediately
  setBanner("loading", `<span class="spinner">⏳</span> Indexing <strong>${escapeHtml(file.name)}</strong>…`);

  const form = new FormData();
  form.append("file", file);
  try {
    const res  = await fetch("/api/upload/pdf", { method: "POST", headers: { "X-Session-ID": sessionId }, body: form });
    const json = await res.json();
    if (!json.success) throw new Error(json.detail?.error || json.error || "Upload failed");
    ragActive = true;
    setBanner("active", `📄 <strong>${escapeHtml(file.name)}</strong> loaded — Document Q&amp;A active`);
    updateRagBadge(true, file.name);
    document.getElementById("msg-input").placeholder = `Ask anything about ${file.name}…`;
  } catch (e) {
    setBanner("error", `❌ ${escapeHtml(e.message)}`);
    ragActive = false;
  }
}

function onImageSelected(input) {
  const file = input.files[0];
  if (!file) return;
  input.value = "";

  // Store file and show thumbnail preview — user types question then hits Send
  pendingImage = { file };

  const reader = new FileReader();
  reader.onload = (ev) => {
    document.getElementById("img-thumb").src     = ev.target.result;
    document.getElementById("img-thumb-name").textContent = file.name;
    document.getElementById("image-preview-area").classList.remove("hidden");
  };
  reader.readAsDataURL(file);

  const ta = document.getElementById("msg-input");
  ta.placeholder = "Ask about this image… (or press Send for a full description)";
  ta.focus();
  // Enable send button so user can submit with empty input (defaults to "Describe")
  document.getElementById("send-btn").disabled = false;
}

function clearImagePreview() {
  pendingImage = null;
  document.getElementById("image-preview-area").classList.add("hidden");
  document.getElementById("img-thumb").src     = "";
  document.getElementById("img-thumb-name").textContent = "";
  document.getElementById("msg-input").placeholder = "Message Geminitor Pro…";
  onInputChange();
}

/* ── Upload banner helpers ─────────────────────────────────────────────── */
function setBanner(state, html) {
  const el = document.getElementById("upload-banner");
  el.className = state === "active" ? "active" : state === "loading" ? "active loading" : "active error";
  document.getElementById("upload-banner-text").innerHTML = html;
  // Hide dismiss button while loading
  document.getElementById("upload-banner-dismiss").style.display = state === "loading" ? "none" : "";
}

function dismissUpload() {
  ragActive = false;
  document.getElementById("upload-banner").className = "";   // hide
  document.getElementById("msg-input").placeholder = "Message Geminitor Pro…";
  updateRagBadge(false);
  fetch("/api/history", { method: "DELETE", headers: { "X-Session-ID": sessionId } });
}

function updateRagBadge(active, filename) {
  const label = document.getElementById("rag-label");
  const btn   = document.getElementById("rag-status-btn");
  if (active && filename) {
    label.textContent = filename.slice(0, 18) + (filename.length > 18 ? "…" : "");
    btn.style.color   = "var(--accent)";
  } else {
    label.textContent = "Upload Doc";
    btn.style.color   = "";
  }
}

/* ── Analytics modal ───────────────────────────────────────────────────── */
async function showAnalytics() {
  document.getElementById("analytics-modal").classList.remove("hidden");
  try {
    const res  = await fetch("/api/analytics", { headers: { "X-Session-ID": sessionId } });
    const json = await res.json();
    if (!json.success) return;
    const d = json.data;
    document.getElementById("stat-msgs").textContent   = d.total_messages;
    document.getElementById("stat-time").textContent   = d.avg_response_time + "s";
    document.getElementById("stat-tokens").textContent = d.total_tokens.toLocaleString();

    // Token bars
    const bars    = document.getElementById("token-bars");
    const history = d.token_history || [];
    if (history.length) {
      const max = Math.max(...history, 1);
      bars.innerHTML = history.map(v =>
        `<div class="token-bar" style="height:${Math.round((v/max)*56)+4}px" title="${v} tokens"></div>`
      ).join("");
    } else { bars.innerHTML = '<span style="font-size:.8rem;color:var(--subtext)">No data yet</span>'; }

    // Topics
    const ul     = document.getElementById("topics-list");
    const topics = d.recent_topics || [];
    ul.innerHTML = topics.length
      ? topics.map(t => `<li>${escapeHtml(t)}</li>`).join("")
      : '<li style="color:var(--subtext)">No prompts yet</li>';
  } catch (e) { /* silent fail */ }
}

function closeAnalytics(e) {
  if (!e || e.target === document.getElementById("analytics-modal")) {
    document.getElementById("analytics-modal").classList.add("hidden");
  }
}

/* ── Feedback ──────────────────────────────────────────────────────────── */
async function sendFeedback(btn, type) {
  btn.style.opacity = "1";
  btn.textContent   = type === "positive" ? "👍✓" : "👎✓";
  btn.disabled      = true;
  const idx = chatHistory.filter(m => m.role === "assistant").length - 1;
  await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-ID": sessionId },
    body: JSON.stringify({ message_index: idx, feedback: type }),
  }).catch(() => {});
}

/* ── Utilities ─────────────────────────────────────────────────────────── */
function scrollToBottom() {
  const c = document.getElementById("messages-container");
  requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
}

function copyText(btn, e) {
  e.stopPropagation();
  const text = btn.dataset.raw || btn.closest(".msg-content")?.querySelector(".bot-bubble")?.innerText || "";
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = "✅ Copied";
    setTimeout(() => { btn.textContent = orig; }, 1800);
  });
}

function renderMarkdown(text) {
  marked.setOptions({ breaks: true, gfm: true });
  return marked.parse(text);
}

function escapeHtml(t) {
  return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function formatTime(d) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function getInitial() {
  return "U";
}

function showToast(msg, isError) {
  let toast = document.getElementById("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    toast.style.cssText = "position:fixed;bottom:90px;left:50%;transform:translateX(-50%);background:var(--sidebar);border:1px solid var(--border);color:var(--text);padding:8px 18px;border-radius:20px;font-size:.83rem;z-index:200;box-shadow:0 4px 16px rgba(0,0,0,.4);transition:opacity .3s";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.borderColor = isError ? "#e57373" : "var(--accent)";
  toast.style.opacity = "1";
  clearTimeout(toast._hide);
  toast._hide = setTimeout(() => { toast.style.opacity = "0"; }, 2800);
}

// Close attach menu on outside click
document.addEventListener("click", (e) => {
  if (!e.target.closest("#attach-btn") && !e.target.closest(".attach-menu")) {
    closeAttachMenu();
  }
});

// Save settings on change
document.getElementById("model-select").addEventListener("change",   saveSettings);
document.getElementById("persona-select").addEventListener("change", saveSettings);
