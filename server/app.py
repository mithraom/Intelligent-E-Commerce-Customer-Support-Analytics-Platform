"""
IECSAP - Backend API (FastAPI)

Thin HTTP layer over the NLP pipeline in ../src.

  POST /api/chat               full pipeline (+ optional LLM phrasing)
  POST /api/stt                server-side speech-to-text (optional Whisper)
  POST /api/translate          translate text between any two supported languages
  POST /api/summarize          LLM hand-off summary of a conversation
  POST /api/feedback           rate a reply (1-5) with optional comment/tags
  GET  /api/feedback/summary   aggregated feedback for the dashboard
  GET  /api/feedback/export    all feedback as CSV
  GET  /api/orders/{id}        direct order lookup
  GET  /api/analytics/tickets  ticket category / sentiment breakdown
  GET  /api/analytics/rules    market-basket cross-sell rules
  GET  /api/faq                FAQ knowledge base
  GET  /api/system             component health: LLM, STT, translation models
  POST /api/system/llm-test    one live round-trip to prove the LLM key works
  GET  /                       the web console

Run from this directory:   py app.py     ->  http://127.0.0.1:8000
"""
import sys
import threading
import time
import uuid
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
BASE_DIR = SERVER_DIR.parent
SRC_DIR = BASE_DIR / "src"
WEB_DIR = BASE_DIR / "web"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import multilingual                                 # noqa: E402
import speech                                       # noqa: E402
from chatbot import SupportChatbot                  # noqa: E402
from feedback import FeedbackStore                  # noqa: E402
from llm import LLMClient, LLMError                 # noqa: E402
from market_basket import mine_rules                # noqa: E402
from qa_system import FAQRetrievalQA                # noqa: E402

app = FastAPI(title="IECSAP API", version="2.0")

# ---------------------------------------------------------------------------
# Load models once at startup rather than per request.
# ---------------------------------------------------------------------------
print("Loading IECSAP pipeline (classifier, NER, QA, translation, LLM)...")
llm = LLMClient()
multilingual.set_llm(llm)
bot = SupportChatbot(llm=llm)
feedback = FeedbackStore(llm=llm)
_, RULES = mine_rules()
FAQ = FAQRetrievalQA().df


def _warm_up():
    """Load the heavy models in the background so the FIRST chat message isn't slow:
    NLLB (10-30 s to load) and the Ollama model (loads into RAM on first call)."""
    try:
        if multilingual.preload():
            print("[warm-up] NLLB translation model loaded.")
        else:
            print("[warm-up] NLLB not fully downloaded -> Tamil/Hindi/etc. use the LLM fallback. "
                  "Run: cd src && py warm_translation_models.py nllb")
    except Exception as e:
        print(f"[warm-up] translation preload failed: {e}")
    if llm.enabled:
        res = llm.ping()
        print("[warm-up] LLM ready." if res.get("ok") else f"[warm-up] LLM not reachable: {res.get('error')}")


threading.Thread(target=_warm_up, daemon=True).start()
print(f"Ready. LLM: {llm.provider if llm.enabled else 'disabled (template replies)'}")


class Turn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    source_lang: str | None = None          # hint, e.g. "ta" from the speech recogniser
    input_mode: str = "text"                # "text" | "voice"
    use_llm: bool = True
    history: list[Turn] = []


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    source: str = "auto"
    target: str = "en"


class SummarizeRequest(BaseModel):
    transcript: list[Turn]


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    tags: list[str] = []


@app.post("/api/chat")
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    result = bot.handle(req.message, source_lang=req.source_lang, use_llm=req.use_llm,
                        history=[t.model_dump() for t in req.history])
    message_id = uuid.uuid4().hex[:12]
    # Stored in SQLite (not memory) so a rating is still accepted after a server restart.
    feedback.log_message(message_id, {
        "route": result.get("route"), "category": result.get("category"),
        "language": result.get("detected_language"), "input_mode": req.input_mode,
        "llm_used": result.get("llm_used"), "user_message": req.message, "bot_reply": result["reply"],
    })
    return {
        "message_id": message_id,
        "reply": result["reply"],
        "detected_language": result["detected_language"],
        "english_text": result["english_text"],
        "route": result["route"],
        "category": result.get("category"),
        "entities": result.get("entities", {}),
        "confidence": result.get("confidence"),
        "escalate": result.get("escalate", False),
        "translation_ok": result.get("translation_ok", True),
        "reply_translation_ok": result.get("reply_translation_ok", True),
        "reply_en": result.get("reply_en"),
        "translation_engine": result.get("translation_engine"),
        "order": result.get("order"),
        "recommendations": result.get("recommendations", []),
        "recommendation_source": result.get("recommendation_source"),
        "llm_used": result.get("llm_used", False),
        "base_reply_en": result.get("base_reply_en"),
        "steps": result.get("steps", []),
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "input_mode": req.input_mode,
    }


@app.post("/api/stt")
async def stt(audio: UploadFile = File(...), language: str = Form("auto")):
    """Server-side speech-to-text (Whisper). The browser's own recogniser is
    used by default; this endpoint is the offline / Firefox path."""
    if not speech.available():
        raise HTTPException(501, "Server speech-to-text is not installed. Run: pip install faster-whisper")
    data = await audio.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Audio too large (15 MB max)")
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    try:
        return speech.transcribe_bytes(data, suffix=suffix, language=None if language == "auto" else language)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")


@app.post("/api/translate")
def translate(req: TranslateRequest):
    t0 = time.perf_counter()
    src = req.source.lower()
    if src == "auto":
        src = multilingual.detect_language(req.text)
    res = multilingual.translate_ex(req.text, src, req.target.lower())
    return {**res, "source": src, "target": req.target.lower(),
            "source_name": multilingual.LANGUAGE_NAMES.get(src, src),
            "target_name": multilingual.LANGUAGE_NAMES.get(req.target.lower(), req.target),
            "latency_ms": int((time.perf_counter() - t0) * 1000)}


@app.get("/api/languages")
def languages():
    return [{"code": k, "name": v} for k, v in multilingual.LANGUAGE_NAMES.items()]


@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    if not llm.enabled:
        raise HTTPException(503, "No LLM configured. See .env.example.")
    try:
        return {"summary": llm.summarize([t.model_dump() for t in req.transcript])}
    except LLMError as e:
        raise HTTPException(502, f"LLM request failed: {e}")


@app.post("/api/feedback")
def post_feedback(req: FeedbackRequest):
    ctx = feedback.get_message(req.message_id)
    if ctx is None:
        raise HTTPException(404, "Unknown message_id: that reply was never sent by this server")
    row = feedback.add(req.message_id, req.rating, req.comment, req.tags, ctx)
    return {"ok": True, "comment_sentiment": row["comment_sentiment"]}


@app.get("/api/feedback/summary")
def feedback_summary():
    return feedback.summary()


@app.get("/api/feedback/export")
def feedback_export():
    return PlainTextResponse(feedback.export_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=iecsap_feedback.csv"})


@app.get("/api/system")
def system():
    return {"llm": llm.status(), "stt": speech.status(), "translation": multilingual.model_status(),
            "orders": int(len(bot.orders.df)), "sample_orders": list(bot.orders.df.index[:3]), "faq_entries": int(FAQ["canonical_question"].nunique())}


@app.post("/api/system/llm-test")
def llm_test():
    return llm.ping()


@app.get("/api/orders/{order_id}")
def get_order(order_id: str):
    info = bot.orders.lookup(order_id.upper())
    if info is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, **info}


@app.get("/api/analytics/tickets")
def analytics_tickets():
    df = pd.read_csv(BASE_DIR / "data" / "tickets.csv")
    order_tickets = df[df["order_id_used"].notna()]
    not_found_rate = float((order_tickets["order_exists"] == False).mean()) if len(order_tickets) else 0.0
    return {
        "total": int(len(df)),
        "categories": df["category"].value_counts().to_dict(),
        "sentiment": df["sentiment"].value_counts().to_dict(),
        "orders_referenced": int(len(order_tickets)),
        "orders_not_found_rate": round(not_found_rate, 3),
    }


@app.get("/api/analytics/rules")
def analytics_rules(top: int = 10):
    top_rules = RULES.head(top)
    return [
        {"antecedents": sorted(list(a)), "consequents": sorted(list(c)),
         "support": round(float(s), 3), "confidence": round(float(conf), 3), "lift": round(float(l), 3)}
        for a, c, s, conf, l in zip(top_rules["antecedents"], top_rules["consequents"],
                                    top_rules["support"], top_rules["confidence"], top_rules["lift"])
    ]


@app.get("/api/faq")
def faq_list():
    canonical = FAQ.drop_duplicates(subset=["canonical_question"])
    return [{"question": r["canonical_question"], "answer": r["answer"]} for _, r in canonical.iterrows()]


# Static frontend, mounted last so it doesn't shadow /api/*.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
