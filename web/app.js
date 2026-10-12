// IECSAP front-end. Vanilla JS, no build step. Same-origin FastAPI backend.
"use strict";

/* ====================================================================== helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; };
const icon = (id, cls = "") => `<svg class="ico ${cls}"><use href="#i-${id}"/></svg>`;
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let toastTimer;
function toast(msg, ms = 3200) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), ms);
}

async function api(path, { method = "GET", body, timeout = 30000, form } = {}) {
  const ctrl = new AbortController(); const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(path, {
      method, signal: ctrl.signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: form || (body ? JSON.stringify(body) : undefined),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch {}
      const err = new Error(detail); err.status = res.status; throw err;
    }
    return await res.json();
  } finally { clearTimeout(timer); }
}

/* Language data. BCP-47 tags are what the browser speech engines expect. */
const BCP47 = { en: "en-IN", ta: "ta-IN", hi: "hi-IN", te: "te-IN", ml: "ml-IN", kn: "kn-IN", bn: "bn-IN", mr: "mr-IN",
  gu: "gu-IN", ur: "ur-PK", es: "es-ES", fr: "fr-FR", de: "de-DE", pt: "pt-PT", it: "it-IT", ar: "ar-SA",
  "zh-cn": "zh-CN", ja: "ja-JP", ko: "ko-KR", ru: "ru-RU" };
let LANGS = Object.keys(BCP47).map((c) => ({ code: c, name: c }));
const langName = (c) => (LANGS.find((l) => l.code === c) || {}).name || c;

const state = { system: null, history: [], msgCount: 0, lastTranslation: null, trHistory: [], charts: [], loaded: {} };

/* ======================================================================= router */
function showView(name) {
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  voice.cancel();
  if (name === "analytics" && !state.loaded.analytics) loadAnalytics();
  if (name === "feedback") loadFeedback();
  if (name === "system") loadSystem();
}
$$(".nav-item").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

/* ======================================================================== theme */
$("#themeToggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("iecsap-theme", next); } catch {}
  if (state.loaded.analytics) renderCharts();
});

/* ===================================================================== speech */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

class VoiceInput {
  constructor() { this.active = false; this.rec = null; this.cancelled = false; this.cb = null; this.recorder = null; this.stream = null; }
  get engine() {
    if (SR) return "browser";
    if (state.system && state.system.stt.server_engine && navigator.mediaDevices && window.MediaRecorder) return "server";
    return null;
  }
  start(lang, cb) {
    if (this.active) this.stop();
    this.cb = cb; this.cancelled = false; this.finalText = "";
    const engine = this.engine;
    if (!engine) { cb.onError("Voice input isn't available here. Use Chrome, Edge or Safari, or install faster-whisper for server-side recognition."); return; }
    this.active = true;
    engine === "browser" ? this._startBrowser(lang) : this._startServer(lang);
  }
  _startBrowser(lang) {
    const rec = (this.rec = new SR());
    rec.lang = BCP47[lang] || lang; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
    rec.onstart = () => this.cb.onStart && this.cb.onStart();
    rec.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) this.finalText += r[0].transcript; else interim += r[0].transcript;
      }
      this.cb.onInterim((this.finalText + interim).trim());
    };
    rec.onerror = (e) => {
      const msg = {
        "not-allowed": "Microphone access was blocked. Allow it in the browser's address bar and try again.",
        "service-not-allowed": "Speech recognition is blocked in this browser.",
        "no-speech": "I didn't catch anything. Try again a little closer to the microphone.",
        "audio-capture": "No microphone was found.",
        "network": "Browser speech recognition needs an internet connection. For offline use, install faster-whisper (see System).",
        "language-not-supported": "This browser can't recognise that language. Try Chrome or Edge.",
      }[e.error];
      if (e.error !== "aborted" && msg) { this.cancelled = true; this.cb.onError(msg); }
    };
    rec.onend = () => this._finish();
    try { rec.start(); } catch (err) { this.active = false; this.cb.onError("Couldn't start the microphone."); }
  }
  async _startServer(lang) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch { this.active = false; this.cb.onError("Microphone access was blocked."); return; }
    const chunks = []; this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    this.recorder.onstop = async () => {
      this.stream.getTracks().forEach((t) => t.stop());
      if (this.cancelled) { this.active = false; this.cb.onEnd(); return; }
      this.cb.onInterim("Transcribing…");
      try {
        const form = new FormData(); form.append("audio", new Blob(chunks, { type: this.recorder.mimeType }), "clip.webm");
        form.append("language", lang || "auto");
        const out = await api("/api/stt", { method: "POST", form, timeout: 120000 });
        this.active = false; this.cb.onFinal(out.text, out.language);
      } catch (err) { this.active = false; this.cb.onError(err.message || "Transcription failed."); }
    };
    this.recorder.start(); this.cb.onStart && this.cb.onStart();
    this.timeout = setTimeout(() => this.stop(), 25000);
  }
  _finish() {
    if (!this.active) return;
    this.active = false;
    if (this.cancelled) { this.cb.onEnd(); return; }
    const text = (this.finalText || "").trim();
    text ? this.cb.onFinal(text) : this.cb.onEnd();
  }
  stop() {
    if (!this.active) return;
    clearTimeout(this.timeout);
    if (this.rec) try { this.rec.stop(); } catch {}
    if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
  }
  cancel() {
    if (!this.active) return;
    this.cancelled = true; clearTimeout(this.timeout);
    if (this.rec) try { this.rec.abort(); } catch {}
    if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
    this.active = false; if (this.cb) this.cb.onEnd();
  }
}
const voice = new VoiceInput();

function speak(text, lang) {
  if (!("speechSynthesis" in window)) { toast("This browser can't read text aloud."); return; }
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); const tag = BCP47[lang] || lang || "en-IN"; u.lang = tag;
  const voices = speechSynthesis.getVoices();
  const match = voices.find((v) => v.lang.toLowerCase() === tag.toLowerCase()) || voices.find((v) => v.lang.toLowerCase().startsWith(tag.slice(0, 2).toLowerCase()));
  if (match) u.voice = match;
  else if (voices.length) toast(`No ${langName(lang)} voice is installed on this device, so the browser will read it with a default voice.`, 4200);
  speechSynthesis.speak(u);
}

/* ====================================================================== console */
const chatScroll = $("#chatScroll"), chatInput = $("#chatInput"), chatForm = $("#chatForm");
const micBtn = $("#micBtn"), voiceBar = $("#voiceBar"), voiceText = $("#voiceText");

function autoGrow() { chatInput.style.height = "auto"; chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + "px"; }
chatInput.addEventListener("input", autoGrow);
chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); chatForm.requestSubmit(); } });

function scrollChat() { chatScroll.scrollTop = chatScroll.scrollHeight; }

function addUserMessage(text, viaVoice, lang) {
  $("#chatEmpty")?.remove();
  const w = document.createElement("div"); w.className = "msg msg-user";
  w.innerHTML = `<div class="msg-who">${viaVoice ? icon("mic") + " Spoken · " + esc(langName(lang)) : "You"}</div><div class="msg-bubble">${esc(text)}</div>`;
  chatScroll.appendChild(w); scrollChat();
}
function addThinking() {
  const w = document.createElement("div"); w.className = "msg msg-bot";
  w.innerHTML = `<div class="msg-who">IECSAP</div><div class="msg-bubble thinking"><i></i><i></i><i></i></div>`;
  chatScroll.appendChild(w); scrollChat(); return w;
}

function routeChip(r) {
  if (r.translation_ok === false || r.route === "ESCALATE") return `<span class="chip amber">Needs a human</span>`;
  switch (r.route) {
    case "ORDER_NOT_FOUND": return `<span class="chip red">Order not found</span>`;
    case "CLARIFY": return `<span class="chip">Needs more detail</span>`;
    case "TICKET": return `<span class="chip accent">Ticket · ${esc(r.category || "General")}</span>`;
    case "QA": return r.escalate ? `<span class="chip amber">No FAQ match</span>` : `<span class="chip green">FAQ answer</span>`;
    default: return `<span class="chip">${esc(r.route)}</span>`;
  }
}

const TAGS_GOOD = ["Accurate", "Fast", "Clear", "Friendly"];
const TAGS_BAD = ["Wrong answer", "Didn't understand me", "Bad translation", "Too generic", "Too slow"];

function recsHtml(r) {
  const recs = r.recommendations || [];
  if (!recs.length || r.route === "ESCALATE") return "";
  const head = { order_history: "Customers who bought items in your order also bought",
                 mentioned_product: "Customers who bought that product also bought",
                 best_sellers: "Popular right now" }[r.recommendation_source] || "You might also like";
  return `<div class="recs"><div class="recs-head">${esc(head)}</div><div class="recs-row">${recs.map((x) =>
    `<span class="rec-chip"><b>${esc(x.product)}</b>${x.confidence != null ? `<small>${Math.round(x.confidence * 100)}% of those buyers · lift ${Number(x.lift).toFixed(1)}×</small>` : ""}</span>`).join("")}</div></div>`;
}

function addBotMessage(r) {
  const w = document.createElement("div"); w.className = "msg msg-bot"; w.dataset.id = r.message_id;
  const understood = r.detected_language !== "en" && r.translation_ok
    ? `<div class="msg-sub">Understood as: “${esc(r.english_text)}”</div>` : "";
  w.innerHTML = `
    <div class="msg-who">IECSAP ${r.llm_used ? '<span class="chip amber" style="height:18px;font-size:11px">AI phrased</span>' : ""}</div>
    <div class="msg-bubble">${esc(r.reply)}</div>
    ${understood}
    ${recsHtml(r)}
    <div class="msg-meta">${routeChip(r)}<span class="chip mono">${esc(r.detected_language)}</span>${r.reply_translation_ok === false ? '<span class="chip amber" title="No translator could render the reply in your language">Shown in English</span>' : ""}${r.input_mode === "voice" ? `<span class="chip">${icon("mic").replace('class="ico "', 'class="ico" style="width:12px;height:12px"')}voice</span>` : ""}</div>
    <div class="msg-tools">
      <button class="icon-btn" data-act="speak" title="Read aloud" aria-label="Read aloud">${icon("speaker")}</button>
      <button class="icon-btn" data-act="copy" title="Copy" aria-label="Copy reply">${icon("copy")}</button>
    </div>
    <div class="fb-ask" role="group" aria-label="Rate this reply">
      <span class="fb-q">Was this helpful?</span>
      <button type="button" class="thumb" data-n="5" title="Helpful" aria-label="Helpful">👍</button>
      <button type="button" class="thumb" data-n="1" title="Not helpful" aria-label="Not helpful">👎</button>
      <span class="stars" aria-label="Or rate 1 to 5 stars">${[1, 2, 3, 4, 5].map((n) => `<button class="star" data-n="${n}" aria-label="${n} star${n > 1 ? "s" : ""}">${icon("star")}</button>`).join("")}</span>
    </div>
    <div class="fb-slot"></div>`;
  chatScroll.appendChild(w); scrollChat();

  $('[data-act="speak"]', w).addEventListener("click", () => speak(r.reply, r.detected_language));
  $('[data-act="copy"]', w).addEventListener("click", () => { navigator.clipboard?.writeText(r.reply); toast("Reply copied"); });
  $$(".star, .thumb", w).forEach((b) => b.addEventListener("click", () => openFeedbackForm(w, r, Number(b.dataset.n))));
}

function paintStars(w, n) { $$(".star", w).forEach((s) => s.classList.toggle("on", Number(s.dataset.n) <= n)); }

function openFeedbackForm(w, r, rating) {
  const slot = $(".fb-slot", w); paintStars(w, rating);
  const tags = rating >= 4 ? TAGS_GOOD : TAGS_BAD; const picked = new Set();
  slot.innerHTML = `
    <form class="fb-form">
      <h4>${rating >= 4 ? "Glad that helped. What worked?" : "Sorry about that. What went wrong?"}</h4>
      <div class="tag-row">${tags.map((t) => `<button type="button" class="tag-btn" data-t="${esc(t)}">${esc(t)}</button>`).join("")}</div>
      <textarea placeholder="Anything else we should know? (optional)" maxlength="500"></textarea>
      <div class="fb-actions"><button type="button" class="btn btn-ghost" data-act="cancel">Cancel</button><button type="submit" class="btn btn-primary">Send feedback</button></div>
    </form>`;
  const form = $(".fb-form", slot);
  $$(".tag-btn", form).forEach((b) => b.addEventListener("click", () => { picked.has(b.dataset.t) ? picked.delete(b.dataset.t) : picked.add(b.dataset.t); b.classList.toggle("on"); }));
  $('[data-act="cancel"]', form).addEventListener("click", () => { slot.innerHTML = ""; paintStars(w, 0); });
  form.querySelector("textarea").focus();
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $('button[type="submit"]', form); btn.disabled = true; btn.textContent = "Sending…";
    try {
      const out = await api("/api/feedback", { method: "POST", body: { message_id: r.message_id, rating, comment: $("textarea", form).value, tags: [...picked] } });
      slot.innerHTML = `<div class="msg-meta"><span class="chip green">${icon("check").replace('class="ico "', 'class="ico" style="width:12px;height:12px"')}Thanks, feedback recorded</span>${out.comment_sentiment ? `<span class="chip">Comment read as ${esc(out.comment_sentiment)}</span>` : ""}</div>`;
      w.classList.add("rated"); const ask = $(".fb-ask", w);
      ask.classList.add("done"); $(".fb-q", ask).textContent = "Thanks for rating";
      $$(".thumb", ask).forEach((b) => b.disabled = true); $(".stars", ask).classList.add("locked");
      refreshFeedbackBadge();
    } catch (err) { btn.disabled = false; btn.textContent = "Send feedback"; toast(err.message || "Couldn't save feedback"); }
  });
}

/* ---- inspector ---- */
function renderInspector(r) {
  state.msgCount++;
  $("#inspectorMeta").textContent = `Message ${state.msgCount} · ${r.latency_ms} ms`;
  const llmStep = (s) => s.stage === "LLM response";
  const timeline = (r.steps || []).map((s) => `
    <li><span class="pin ${llmStep(s) ? (s.detail.startsWith("failed") ? "fail" : "llm") : ""}"></span>
      <div><div class="st">${esc(s.stage)}</div><div class="sd">${esc(s.detail)}</div></div><span class="ms">${s.ms} ms</span></li>`).join("");
  const entities = Object.entries(r.entities || {}).flatMap(([k, vs]) => (vs || []).map((v) => `<span class="etag ${esc(k)}"><b>${esc(k)}</b>${esc(v)}</span>`));
  const o = r.order;
  const order = o ? `
    <div class="block"><div class="block-head">Order on file</div><div class="block-body">
      <div class="kv"><span class="k">Order</span><span class="v mono">${esc(o.order_id)}</span></div>
      <div class="kv"><span class="k">Status</span><span class="v">${esc(o.status)}</span></div>
      <div class="kv"><span class="k">Placed</span><span class="v">${esc(o.order_date)}</span></div>
      <div class="kv"><span class="k">Total</span><span class="v">$${Number(o.total_amount).toFixed(2)}</span></div>
      <div class="kv"><span class="k">Items</span><span class="v">${esc((o.items || []).join(", "))}</span></div></div></div>` : "";
  const recLabel = { order_history: "From this customer's order", mentioned_product: "From the product mentioned", best_sellers: "Best-sellers (no personal match)" }[r.recommendation_source] || "";
  const recs = (r.recommendations || []).length ? `
    <div class="block"><div class="block-head"><span>Cross-sell</span><span style="text-transform:none;letter-spacing:0;font-weight:500">${esc(recLabel)}</span></div><div class="block-body">
      ${r.recommendations.map((x) => `<div class="kv"><span class="v" style="text-align:left">${esc(x.product)}</span><span class="k mono">${x.confidence != null ? Math.round(x.confidence * 100) + "% · " + Number(x.lift).toFixed(2) + "×" : "popular"}</span></div>`).join("")}</div></div>` : "";
  const cmp = r.llm_used ? `
    <div class="block"><div class="block-head">Reply composition</div><div class="block-body compare">
      <div class="cmp base"><small>Verified template (facts only)</small>${esc(r.base_reply_en)}</div>
      <div class="cmp ai"><small>Final · phrased by the LLM from those facts</small>${esc(r.reply_en || r.reply)}</div></div></div>` : "";

  $("#inspectorBody").innerHTML = `
    <div class="block"><div class="block-head"><span>Pipeline</span><span style="text-transform:none;letter-spacing:0;font-weight:500">${r.steps.length} stages</span></div><div class="block-body"><ol class="timeline">${timeline}</ol></div></div>
    <div class="block"><div class="block-head">Understanding</div><div class="block-body">
      <div class="kv"><span class="k">Input</span><span class="v">${r.input_mode === "voice" ? "Speech → text" : "Typed text"}</span></div>
      <div class="kv"><span class="k">Language</span><span class="v">${esc(langName(r.detected_language))}</span></div>
      ${r.detected_language !== "en" ? `<div class="kv"><span class="k">English</span><span class="v">${r.translation_ok ? esc(r.english_text) : "unavailable"}</span></div><div class="kv"><span class="k">Translator</span><span class="v">${esc(r.translation_engine || "none")}</span></div>` : ""}
      <div class="kv"><span class="k">Route</span><span class="v">${routeChip(r)}</span></div>
      ${r.confidence != null ? `<div class="kv"><span class="k">Match score</span><span class="v mono">${Number(r.confidence).toFixed(2)}</span></div>` : ""}</div></div>
    <div class="block"><div class="block-head">Entities</div><div class="block-body"><div class="tags">${entities.join("") || '<span class="muted-xs">None detected</span>'}</div></div></div>
    ${order}${cmp}${recs}`;
}

/* ---- sending ---- */
let sending = false;
async function sendMessage(text, { viaVoice = false, lang = null } = {}) {
  text = text.trim(); if (!text || sending) return;
  sending = true; $("#sendBtn").disabled = true;
  addUserMessage(text, viaVoice, lang || "en");
  const thinking = addThinking();
  const history = state.history.slice(-6);
  try {
    const r = await api("/api/chat", { method: "POST", timeout: 60000, body: {
      message: text, source_lang: viaVoice ? lang : null, input_mode: viaVoice ? "voice" : "text",
      use_llm: $("#llmSwitch").checked, history } });
    thinking.remove(); addBotMessage(r); renderInspector(r);
    const unrated = $$(".msg-bot[data-id]:not(.rated)").length;
    if (unrated === 3 && !state.nudged) { state.nudged = true; toast("Quick favour: tap 👍 / 👎 (or a star) under a reply. Your rating is what the Feedback page is built from.", 6500); }
    state.history.push({ role: "user", text }, { role: "assistant", text: r.reply });
  } catch (err) {
    thinking.remove();
    const w = document.createElement("div"); w.className = "msg msg-bot";
    w.innerHTML = `<div class="msg-who">IECSAP</div><div class="msg-bubble">${err.name === "AbortError" ? "That took too long. The server may be busy; please try again." : "I couldn't reach the server. Is it still running?"}</div>`;
    chatScroll.appendChild(w); scrollChat();
  } finally { sending = false; $("#sendBtn").disabled = false; chatInput.focus(); }
}

chatForm.addEventListener("submit", (e) => { e.preventDefault(); const t = chatInput.value; chatInput.value = ""; autoGrow(); sendMessage(t); });

/* ---- microphone in the console ---- */
function setRecordingUI(on) {
  micBtn.classList.toggle("recording", on); voiceBar.hidden = !on;
  micBtn.innerHTML = icon(on ? "stop" : "mic"); micBtn.setAttribute("aria-label", on ? "Stop recording" : "Speak a message");
  if (!on) voiceText.textContent = "Listening…";
}
micBtn.addEventListener("click", () => {
  if (voice.active) { voice.stop(); return; }
  const lang = $("#voiceLang").value;
  voice.start(lang, {
    onStart: () => setRecordingUI(true),
    onInterim: (t) => { voiceText.textContent = t || "Listening…"; },
    onFinal: (t, detected) => {
      setRecordingUI(false);
      const used = detected && BCP47[detected] ? detected : lang;
      if ($("#autoSend").checked) sendMessage(t, { viaVoice: true, lang: used });
      else { chatInput.value = t; autoGrow(); chatInput.focus(); toast("Transcribed. Edit if needed, then press Send."); }
    },
    onEnd: () => setRecordingUI(false),
    onError: (m) => { setRecordingUI(false); toast(m, 5200); },
  });
});
$("#voiceCancel").addEventListener("click", () => voice.cancel());

/* ---- toolbar ---- */
$("#clearBtn").addEventListener("click", () => {
  voice.cancel(); state.history = []; state.msgCount = 0;
  chatScroll.innerHTML = ""; chatScroll.appendChild(buildEmptyState());
  $("#inspectorBody").innerHTML = `<div class="empty-note">${icon("cpu", "ico-lg")}<p>Send a message to see language detection, translation, classification, entities, order lookup and the final reply composition.</p></div>`;
  $("#inspectorMeta").textContent = "Waiting for a message";
});
$("#summarizeBtn").addEventListener("click", async () => {
  if (!state.history.length) { toast("Have a conversation first, then summarise it."); return; }
  if (!state.system?.llm.enabled) { toast("Hand-off summaries need an LLM. See the System page to enable one.", 4200); return; }
  const btn = $("#summarizeBtn"); btn.disabled = true;
  try {
    const out = await api("/api/summarize", { method: "POST", body: { transcript: state.history.map((t) => ({ role: t.role === "user" ? "user" : "assistant", text: t.text })) }, timeout: 60000 });
    $("#inspectorBody").insertAdjacentHTML("afterbegin", `<div class="summary-card"><h4>Hand-off summary</h4>${esc(out.summary)}</div>`);
    $("#inspector").classList.add("open");
  } catch (err) { toast(err.message || "Couldn't summarise"); } finally { btn.disabled = false; }
});
$("#inspectorToggle").addEventListener("click", () => $("#inspector").classList.toggle("open"));

function buildEmptyState() {
  const d = document.createElement("div"); d.className = "chat-empty"; d.id = "chatEmpty";
  const sample = (state.system?.sample_orders || ["ORD10053"])[0];
  const tr = state.system?.translation; const canTranslate = tr && (tr.llm_fallback || tr.nllb || (tr.marian_pairs || []).length);
  const items = [
    { tag: "Order", text: `Where is my order ${sample}?` },
    { tag: "FAQ", text: "What is your return policy?" },
    { tag: "Unknown ID", text: "My order ORD00000 arrived damaged." },
    { tag: "Billing", text: "I was charged twice for my last purchase." },
  ];
  if (canTranslate) items.push({ tag: "Tamil", text: `என் ஆர்டர் ${sample} சேதமடைந்து வந்தது`, lang: "ta" }, { tag: "Spanish", text: `Hola, mi pedido ${sample} llegó dañado.`, lang: "es" });
  d.innerHTML = `<h2>How can we help today?</h2><p>Try one of these, or use the microphone and speak in your own language.</p><div class="suggestions"></div>`;
  items.forEach((it) => {
    const b = document.createElement("button"); b.className = "suggestion"; b.type = "button";
    b.innerHTML = `<span class="tag">${esc(it.tag)}</span>${esc(it.text)}`;
    b.addEventListener("click", () => sendMessage(it.text, it.lang ? { viaVoice: false } : {}));
    $(".suggestions", d).appendChild(b);
  });
  return d;
}

/* ================================================================ voice & translate */
const trIn = $("#trInput"), trOut = $("#trOutput");
function fillLangSelects() {
  const opts = LANGS.map((l) => `<option value="${esc(l.code)}">${esc(l.name)}</option>`).join("");
  $("#voiceLang").innerHTML = opts; $("#voiceLang").value = "en";
  $("#trSource").innerHTML = `<option value="auto">Detect language</option>` + opts; $("#trSource").value = "en";
  $("#trTarget").innerHTML = opts; $("#trTarget").value = "ta";
}

trIn.addEventListener("input", () => { $("#trCount").textContent = `${trIn.value.length} / 4000`; });

$("#trMic").addEventListener("click", () => {
  const btn = $("#trMic");
  if (voice.active) { voice.stop(); return; }
  let src = $("#trSource").value;
  if (src === "auto" && voice.engine === "browser") { src = "en"; toast("Browser speech recognition needs a fixed language. Using English; pick another source language to change it.", 4500); }
  const setRec = (on) => { btn.classList.toggle("recording", on); $("#trBars").hidden = !on; btn.querySelector("span").textContent = on ? "Stop" : "Start speaking"; btn.querySelector("use").setAttribute("href", on ? "#i-stop" : "#i-mic"); };
  voice.start(src === "auto" ? null : src, {
    onStart: () => setRec(true),
    onInterim: (t) => { trIn.value = t; $("#trCount").textContent = `${t.length} / 4000`; },
    onFinal: (t, detected) => { setRec(false); trIn.value = t; if (detected && BCP47[detected]) $("#trSource").value = detected; runTranslate(); },
    onEnd: () => setRec(false),
    onError: (m) => { setRec(false); toast(m, 5200); },
  });
});

async function runTranslate() {
  const text = trIn.value.trim(); if (!text) { toast("Type or speak something first."); return; }
  const go = $("#trGo"); go.disabled = true; trOut.innerHTML = `<span class="placeholder">Translating…</span>`;
  try {
    const r = await api("/api/translate", { method: "POST", timeout: 60000, body: { text, source: $("#trSource").value, target: $("#trTarget").value } });
    if (!r.ok) {
      trOut.innerHTML = `<span class="placeholder">No translator is available for ${esc(r.source_name)} → ${esc(r.target_name)} yet. Add an API key for the LLM or download a local model (System page).</span>`;
      $("#trEngineChip").textContent = "Engine: none"; state.lastTranslation = null; setTrButtons(false); return;
    }
    trOut.textContent = r.text; $("#trEngineChip").textContent = `Engine: ${r.engine} · ${r.latency_ms} ms`;
    state.lastTranslation = r; setTrButtons(true);
    state.trHistory.unshift({ src: r.source, tgt: r.target, text, out: r.text, engine: r.engine }); renderTrHistory();
  } catch (err) { trOut.innerHTML = `<span class="placeholder">${esc(err.message || "Translation failed")}</span>`; }
  finally { go.disabled = false; }
}
function setTrButtons(on) { ["trSpeak", "trCopy", "trToChat"].forEach((id) => ($("#" + id).disabled = !on)); }
function renderTrHistory() {
  $("#trHistory").innerHTML = state.trHistory.slice(0, 6).map((h) => `
    <div class="hist-row"><span class="pair">${esc(h.src)} → ${esc(h.tgt)}</span><span class="t">${esc(h.text)}</span><span class="t">${esc(h.out)}</span><span class="chip">${esc(h.engine)}</span></div>`).join("") || `<p class="muted">Nothing yet.</p>`;
}
$("#trGo").addEventListener("click", runTranslate);
trIn.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runTranslate(); });
$("#trSpeak").addEventListener("click", () => state.lastTranslation && speak(state.lastTranslation.text, state.lastTranslation.target));
$("#trCopy").addEventListener("click", () => { if (state.lastTranslation) { navigator.clipboard?.writeText(state.lastTranslation.text); toast("Translation copied"); } });
$("#trToChat").addEventListener("click", () => { if (!state.lastTranslation) return; showView("console"); sendMessage(state.lastTranslation.text); });
$("#trSwap").addEventListener("click", () => {
  const s = $("#trSource"), t = $("#trTarget"); const prev = s.value === "auto" ? (state.lastTranslation?.source || "en") : s.value;
  s.value = t.value; t.value = prev;
  if (state.lastTranslation) { trIn.value = state.lastTranslation.text; trOut.innerHTML = `<span class="placeholder">The translation appears here.</span>`; setTrButtons(false); state.lastTranslation = null; }
});

/* ======================================================================= analytics */
let analyticsData = null;
async function loadAnalytics() {
  state.loaded.analytics = true;
  try {
    const [tickets, rules, faq] = await Promise.all([api("/api/analytics/tickets"), api("/api/analytics/rules?top=8"), api("/api/faq")]);
    analyticsData = { tickets, rules };
    const topLift = rules.length ? Math.max(...rules.map((r) => r.lift)) : 0;
    const kpis = [
      [tickets.total.toLocaleString(), "Tickets analysed"], [Object.keys(tickets.categories).length, "Support categories"],
      [rules.length, "Cross-sell rules"], [topLift.toFixed(1) + "×", "Strongest lift"],
      [tickets.orders_referenced.toLocaleString(), "Tickets citing an order"], [Math.round(tickets.orders_not_found_rate * 100) + "%", "Of those, order not found"],
    ];
    $("#kpiRow").innerHTML = kpis.map(([v, l]) => `<div class="kpi"><div class="kpi-value">${esc(v)}</div><div class="kpi-label">${esc(l)}</div></div>`).join("");
    const maxLift = Math.max(...rules.map((r) => r.lift), 1);
    $("#rulesTable tbody").innerHTML = rules.map((r) => `
      <tr><td>${esc(r.antecedents.join(", "))}</td><td>${esc(r.consequents.join(", "))}</td>
      <td class="num">${(r.support * 100).toFixed(1)}%</td>
      <td class="num">${Math.round(r.confidence * 100)}%</td>
      <td class="num"><div class="bar-cell"><div class="mini-bar"><span style="width:${(r.lift / maxLift) * 100}%"></span></div>${r.lift.toFixed(2)}×</div></td></tr>`).join("");
    $("#faqList").innerHTML = faq.map((f) => `<div class="faq-item"><div class="faq-q">${esc(f.question)}</div><div class="faq-a">${esc(f.answer)}</div></div>`).join("");
    renderCharts();
  } catch { $("#kpiRow").innerHTML = `<div class="kpi">Couldn't load analytics. Is the backend running?</div>`; state.loaded.analytics = false; }
}
function renderCharts() {
  if (!analyticsData || typeof Chart === "undefined") return;
  state.charts.forEach((c) => c.destroy()); state.charts = [];
  const ink3 = cssVar("--ink-3"), grid = cssVar("--chart-grid"), accent = cssVar("--accent");
  Chart.defaults.font.family = '"Geist", system-ui, sans-serif'; Chart.defaults.color = ink3;
  const { categories, sentiment } = analyticsData.tickets;
  state.charts.push(new Chart($("#categoryChart"), {
    type: "bar", data: { labels: Object.keys(categories), datasets: [{ data: Object.values(categories), backgroundColor: accent, borderRadius: 4, maxBarThickness: 44 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false }, ticks: { color: ink3 } }, y: { beginAtZero: true, grid: { color: grid }, border: { display: false }, ticks: { color: ink3 } } } } }));
  const colors = { positive: cssVar("--green"), neutral: ink3, negative: cssVar("--red") };
  state.charts.push(new Chart($("#sentimentChart"), {
    type: "doughnut", data: { labels: Object.keys(sentiment), datasets: [{ data: Object.values(sentiment), backgroundColor: Object.keys(sentiment).map((k) => colors[k] || ink3), borderWidth: 2, borderColor: cssVar("--surface") }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: "64%", plugins: { legend: { position: "right", labels: { usePointStyle: true, boxWidth: 8, color: cssVar("--ink-2") } } } } }));
}

/* ======================================================================== feedback */
const stars = (n) => "★".repeat(n) + `<span style="opacity:.25">${"★".repeat(5 - n)}</span>`;
async function refreshFeedbackBadge() {
  try {
    const s = await api("/api/feedback/summary"); const n = s.review_queue.length; const b = $("#fbBadge");
    b.hidden = n === 0; b.textContent = n;
    return s;
  } catch { return null; }
}
async function loadFeedback() {
  const root = $("#fbRoot");
  const s = await refreshFeedbackBadge();
  if (!s) { root.innerHTML = `<div class="empty-state"><h3>Couldn't load feedback</h3><p>Is the backend running?</p></div>`; return; }
  if (!s.total) {
    root.innerHTML = `<div class="card"><div class="empty-state">${icon("star", "ico-lg")}<h3>No feedback yet</h3>
      <p>Rate any reply in the Support console with 👍 / 👎 or the stars under the message.<br>Only ratings given by real users appear here, together with the context that produced them.</p>
      ${s.replies_total ? `<p class="muted-xs">${s.replies_total} repl${s.replies_total === 1 ? "y has" : "ies have"} been sent so far and none have been rated yet.</p>` : ""}</div></div>`;
    return;
  }
  const pos = s.sentiment.positive || 0, withComment = Object.values(s.sentiment).reduce((a, b) => a + b, 0);
  const maxDist = Math.max(...Object.values(s.distribution), 1);
  const pretty = (n) => ({ text: "Typed", voice: "Spoken" }[n] || (BCP47[n] ? langName(n) : n));
  const rows = (arr) => arr.map((g) => `<tr><td>${esc(pretty(g.name))}</td><td class="num">${g.count}</td><td class="num"><div class="bar-cell"><div class="mini-bar"><span style="width:${g.average / 5 * 100}%;background:${g.average >= 4 ? cssVar("--green") : g.average >= 3 ? cssVar("--amber") : cssVar("--red")}"></span></div>${g.average.toFixed(2)}</div></td></tr>`).join("");
  const table = (title, arr) => arr.length ? `<div><div class="field-label" style="margin:14px 0 6px">${title}</div><table class="table"><tbody>${rows(arr)}</tbody></table></div>` : "";
  root.innerHTML = `
  <div class="stack">
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-value">${s.average.toFixed(1)}<span style="font-size:20px;color:var(--ink-3)"> / 5</span></div><div class="kpi-label">Average rating</div></div>
      <div class="kpi"><div class="kpi-value">${s.csat}%</div><div class="kpi-label">Satisfied (4–5 stars)</div></div>
      <div class="kpi"><div class="kpi-value">${s.total}</div><div class="kpi-label">Ratings collected</div></div>
      <div class="kpi"><div class="kpi-value">${s.response_rate}%</div><div class="kpi-label">Of ${s.replies_total} replies rated</div></div>
      <div class="kpi"><div class="kpi-value">${withComment ? Math.round(pos / withComment * 100) + "%" : "–"}</div><div class="kpi-label">Comments read as positive</div></div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-head"><h3>Rating distribution</h3></div><div class="dist">
        ${[5, 4, 3, 2, 1].map((n) => `<div class="dist-row"><span>${n} ★</span><div class="dist-bar ${n >= 4 ? "high" : n === 3 ? "mid" : "low"}"><span style="width:${s.distribution[n] / maxDist * 100}%"></span></div><span class="mono" style="text-align:right">${s.distribution[n]}</span></div>`).join("")}</div>
        ${Object.keys(s.tags).length ? `<div class="field-label" style="margin:18px 0 8px">What people mentioned</div><div class="tags">${Object.entries(s.tags).sort((a, b) => b[1] - a[1]).map(([t, c]) => `<span class="chip">${esc(t)} · ${c}</span>`).join("")}</div>` : ""}
      </div>
      <div class="card"><div class="card-head"><h3>Where replies work, and where they don't</h3></div>
        <p class="muted-xs" style="margin-bottom:-6px">Average star rating and number of rated replies.</p>${table("By route", s.by_route)}${table("LLM-composed vs template", s.llm_vs_template)}${table("Typed vs spoken", s.by_mode)}${table("By language", s.by_language)}</div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-head"><h3>Review queue</h3><span class="muted-xs">Rated 1–2 stars</span></div>
        ${s.review_queue.length ? s.review_queue.map((r) => `<div class="review-item"><div class="review-top"><span class="mini-stars">${stars(r.rating)}</span><span class="chip">${esc(r.route || "n/a")}</span><span class="chip mono">${esc(r.language || "")}</span>${r.input_mode === "voice" ? '<span class="chip">voice</span>' : ""}</div>
          <div class="qa"><b>Customer:</b> ${esc(r.user_message)}</div><div class="qa"><b>Bot:</b> ${esc(r.bot_reply)}</div>${r.comment ? `<div class="qa"><b>Comment:</b> ${esc(r.comment)}</div>` : ""}</div>`).join("") : `<p class="muted">Nothing needs review. No reply has been rated below 3 stars.</p>`}</div>
      <div class="card"><div class="card-head"><h3>Recent comments</h3></div>
        ${s.recent.length ? s.recent.map((r) => `<div class="review-item"><div class="review-top"><span class="mini-stars">${stars(r.rating)}</span>${r.comment_sentiment ? `<span class="chip ${r.comment_sentiment === "positive" ? "green" : r.comment_sentiment === "negative" ? "red" : ""}">${esc(r.comment_sentiment)}</span>` : ""}</div><div>${esc(r.comment)}</div></div>`).join("") : `<p class="muted">No written comments yet.</p>`}</div>
    </div>
  </div>`;
}
$("#fbRefresh").addEventListener("click", loadFeedback);

/* ========================================================================== system */
async function loadSystem() {
  const root = $("#sysRoot");
  try { state.system = await api("/api/system"); } catch { root.innerHTML = `<div class="card"><p>The backend isn't reachable.</p></div>`; return; }
  updateHealth(); const s = state.system;
  const dot = (cls) => `<span class="dot ${cls}"></span>`;
  const tr = s.translation;
  root.innerHTML = `<div class="sys-grid">
    <div class="card sys-card"><h3>${dot(s.llm.enabled ? "ok" : "warn")}Language model</h3>
      <p>${s.llm.enabled ? `Phrases replies from verified facts, translates when no local model is cached, and writes hand-off summaries.` : `Not configured. Replies use the verified templates until you add a key.`}</p>
      <div class="kv"><span class="k">Provider</span><span class="v mono">${esc(s.llm.provider)}</span></div>
      <div class="kv"><span class="k">Model</span><span class="v mono">${esc(s.llm.model || "–")}</span></div>
      <div class="kv"><span class="k">Calls this session</span><span class="v mono">${s.llm.calls}</span></div>
      ${s.llm.last_error ? `<div class="kv"><span class="k">Last error</span><span class="v" style="color:var(--red)">${esc(s.llm.last_error)}</span></div>` : ""}
      <div class="sys-actions"><button class="btn" id="llmTest" ${s.llm.enabled ? "" : "disabled"}>Test connection</button><span class="muted-xs" id="llmTestOut"></span></div>
      ${s.llm.enabled ? "" : `<div class="codeblock">cp .env.example .env\n# then set ANTHROPIC_API_KEY=… (or OPENAI_API_KEY, or run Ollama locally)\n# restart: py server/app.py</div>`}</div>

    <div class="card sys-card"><h3>${dot(voice.engine ? "ok" : "warn")}Speech to text</h3>
      <p>Two engines. The browser's recogniser is used first; Whisper on the server is the offline option.</p>
      <div class="kv"><span class="k">Browser engine</span><span class="v">${SR ? "available" : "not supported here"}</span></div>
      <div class="kv"><span class="k">Server engine (Whisper)</span><span class="v">${s.stt.server_engine ? "installed · " + esc(s.stt.model) : "not installed"}</span></div>
      ${s.stt.server_engine ? "" : `<div class="codeblock">pip install faster-whisper</div>`}</div>

    <div class="card sys-card"><h3>${dot(tr.llm_fallback || tr.nllb || tr.marian_pairs.length ? "ok" : "warn")}Translation</h3>
      <p>Order of preference: local MarianMT / NLLB-200 models, then the LLM, then a tiny offline phrasebook.</p>
      <div class="kv"><span class="k">transformers installed</span><span class="v">${tr.transformers ? "yes" : "no"}</span></div>
      <div class="kv"><span class="k">MarianMT pairs cached</span><span class="v mono">${tr.marian_pairs.length ? esc(tr.marian_pairs.join(", ")) : "none"}</span></div>
      <div class="kv"><span class="k">NLLB-200 cached</span><span class="v">${tr.nllb ? "yes" : "no"}</span></div>
      <div class="kv"><span class="k">LLM fallback</span><span class="v">${tr.llm_fallback ? "active" : "off"}</span></div>
      <div class="codeblock">py -m pip install transformers sentencepiece torch\ncd src &amp;&amp; py warm_translation_models.py nllb</div></div>

    <div class="card sys-card"><h3>${dot("ok")}Data</h3>
      <p>Everything the bot answers from.</p>
      <div class="kv"><span class="k">Orders on file</span><span class="v mono">${s.orders.toLocaleString()}</span></div>
      <div class="kv"><span class="k">FAQ entries</span><span class="v mono">${s.faq_entries}</span></div>
      <div class="sys-actions"><input id="orderProbe" class="select" style="background-image:none;padding-right:10px;width:160px" placeholder="ORD10053"><button class="btn" id="orderProbeBtn">Look up order</button></div>
      <div id="orderProbeOut" class="muted-xs" style="margin-top:8px"></div></div>
  </div>`;
  $("#llmTest")?.addEventListener("click", async () => {
    const out = $("#llmTestOut"); out.textContent = "Testing…";
    const r = await api("/api/system/llm-test", { method: "POST", timeout: 40000 }).catch((e) => ({ ok: false, error: e.message }));
    out.style.color = r.ok ? "var(--green)" : "var(--red)"; out.textContent = r.ok ? `Working · ${r.latency_ms} ms` : r.error;
  });
  $("#orderProbeBtn").addEventListener("click", async () => {
    const id = $("#orderProbe").value.trim().toUpperCase(); if (!id) return;
    const r = await api(`/api/orders/${encodeURIComponent(id)}`).catch(() => null);
    $("#orderProbeOut").textContent = !r ? "Request failed" : r.found ? `${r.order_id}: ${r.status}, ${r.items.join(", ")}, $${r.total_amount.toFixed(2)}` : `${id} is not on file.`;
  });
}
$("#sysRefresh").addEventListener("click", loadSystem);

function updateHealth() {
  const s = state.system;
  $("#dotApi").className = "dot " + (s ? "ok" : "bad"); $("#valApi").textContent = s ? "online" : "offline";
  if (!s) return;
  $("#dotLlm").className = "dot " + (s.llm.enabled ? "ok" : "warn"); $("#valLlm").textContent = s.llm.enabled ? s.llm.provider : "off";
  const eng = voice.engine; $("#dotStt").className = "dot " + (eng ? "ok" : "warn");
  $("#valStt").textContent = eng === "browser" ? "browser" : eng === "server" ? "whisper" : "text only";
  $("#llmSwitch").disabled = !s.llm.enabled; $("#llmSwitch").checked = s.llm.enabled;
  $("#llmSwitchWrap").classList.toggle("disabled", !s.llm.enabled);
  $("#llmSwitchWrap").title = s.llm.enabled ? "Let the language model phrase replies from verified facts" : "No LLM configured. See System.";
  $("#micBtn").disabled = !eng; $("#micBtn").title = eng ? "Speak a message" : "Voice input isn't available in this browser";
  $("#engineNote").textContent = eng === "browser" ? "Voice: browser recognition" : eng === "server" ? "Voice: Whisper on server" : "Voice unavailable in this browser";
  $("#trMic").disabled = !eng;
}

/* ========================================================================== boot */
(async function init() {
  try { LANGS = await api("/api/languages"); } catch {}
  fillLangSelects();
  try { state.system = await api("/api/system"); } catch { state.system = null; }
  updateHealth();
  chatScroll.innerHTML = ""; chatScroll.appendChild(buildEmptyState());
  refreshFeedbackBadge();
  if ("speechSynthesis" in window) speechSynthesis.getVoices();
})();
