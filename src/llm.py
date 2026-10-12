"""
IECSAP - LLM integration layer.

The classical pipeline (classifier, NER, order lookup, FAQ retrieval, market
basket) decides WHAT is true and WHAT should happen. The LLM is used only to
phrase that outcome naturally, to translate when no local model is cached, and
to summarise a conversation for a human agent. It is never allowed to invent
order details or policies: every call is given a fact sheet built from the
pipeline's own output and told to use nothing else. This is the "grounded
generation" pattern, and it is why the bot can be fluent without being able to
hallucinate an order status.

Providers (chosen by environment variables, see .env.example):
    anthropic  - Claude via the Messages API           (ANTHROPIC_API_KEY)
    openai     - any OpenAI-compatible chat endpoint   (OPENAI_API_KEY,
                 optionally OPENAI_BASE_URL for Groq, Together, OpenRouter...)
    ollama     - a local model, no key, no internet    (OLLAMA_HOST, default
                 http://127.0.0.1:11434)
    none       - LLM disabled; the system falls back to its template replies

Only the standard library is used for HTTP, so there is nothing extra to install.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_env_file(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE per line). Real environment variables win."""
    path = path or (BASE / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}

SYSTEM_PROMPT = (
    "You are the voice of IECSAP, an e-commerce customer support assistant. "
    "You are given a FACT SHEET from the company's systems, an APPROVED BASELINE REPLY, "
    "and the customer's message. Rewrite the baseline reply so it sounds warm and natural.\n"
    "Rules:\n"
    "1. Keep EVERY fact, number, order ID, status, item and amount from the baseline reply exactly as written.\n"
    "2. Add NOTHING that is not in the fact sheet: no new numbers, order IDs, dates, prices, policies or promises. "
    "Never mention an order the fact sheet does not list.\n"
    "3. The customer's current message is the only thing you answer. Earlier conversation is background only; "
    "never repeat or continue an earlier answer.\n"
    "4. If the fact sheet says the order was not found, say so and ask the customer to double-check the number. "
    "If it says no answer is on file, say a human agent will follow up.\n"
    "5. One to three sentences, plain text, no markdown, no emoji, no sign-off. "
    "If the customer sounds frustrated, acknowledge it in a few words first.\n"
    "6. Reply in English. Translation is handled elsewhere."
)


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        raise LLMError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"network error: {e}") from e


class LLMClient:
    def __init__(self, timeout: float | None = None):
        self.timeout = timeout or float(os.environ.get("LLM_TIMEOUT", "60"))
        self.provider = self._resolve_provider()
        self.model = os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(self.provider, "")
        self.last_error: str | None = None
        self.calls = 0

    # ---------------------------------------------------------------- config
    @staticmethod
    def _resolve_provider() -> str:
        explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if explicit in ("anthropic", "openai", "ollama", "none"):
            return explicit
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "none"

    @property
    def enabled(self) -> bool:
        if self.provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        return self.provider == "ollama"

    def status(self) -> dict:
        return {"provider": self.provider, "model": self.model if self.enabled else None,
                "enabled": self.enabled, "last_error": self.last_error, "calls": self.calls}

    # ----------------------------------------------------------- raw complete
    def complete(self, system: str, user: str, max_tokens: int = 300,
                 temperature: float = 0.3) -> str:
        if not self.enabled:
            raise LLMError("LLM is not configured")
        self.calls += 1
        try:
            if self.provider == "anthropic":
                data = _post_json(
                    os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages",
                    {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                     "system": system, "messages": [{"role": "user", "content": user}]},
                    {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                    self.timeout)
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            elif self.provider == "openai":
                base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                data = _post_json(
                    base + "/chat/completions",
                    {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": user}]},
                    {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}, self.timeout)
                text = data["choices"][0]["message"]["content"]
            else:  # ollama
                host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
                data = _post_json(
                    host + "/api/chat",
                    {"model": self.model, "stream": False, "keep_alive": "30m",
                     "options": {"temperature": temperature, "num_predict": max_tokens},
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": user}]},
                    {}, self.timeout)
                text = data["message"]["content"]
            self.last_error = None
            return text.strip()
        except (LLMError, KeyError, IndexError, json.JSONDecodeError) as e:
            self.last_error = str(e)
            raise LLMError(str(e)) from e

    # ------------------------------------------------------------- task APIs
    @staticmethod
    def build_fact_sheet(route: str, category: str | None, entities: dict,
                         order: dict | None, faq_answer: str | None,
                         draft_reply: str, recommendations: list) -> str:
        lines = [f"Routing decision: {route}"]
        if category:
            lines.append(f"Issue category: {category}")
        if entities:
            lines.append("Entities found in message: " +
                         "; ".join(f"{k}={', '.join(v)}" for k, v in entities.items() if v))
        if route == "ORDER_NOT_FOUND":
            lines.append("Order lookup: NOT FOUND in the order system.")
        elif order:
            lines.append(f"Order lookup: FOUND. id={order['order_id']}, status={order['status']}, "
                         f"placed={order['order_date']}, total=${order['total_amount']:.2f}, "
                         f"items={', '.join(order['items'])}")
        if faq_answer:
            lines.append(f"Knowledge-base answer: {faq_answer}")
        elif route == "QA":
            lines.append("Knowledge-base answer: NONE ON FILE (a human agent must follow up).")
        if recommendations:
            lines.append("Optional cross-sell (mention at most one, lightly, only if the issue is resolved "
                         "or neutral): " + ", ".join(r["product"] for r in recommendations[:2]))
        lines.append(f"Approved baseline reply: {draft_reply}")
        return "\n".join(lines)

    def compose_reply(self, message_en: str, fact_sheet: str, history: list[dict] | None = None,
                      route: str | None = None) -> str:
        """Phrase the already-decided outcome. History is deliberately NOT given for FAQ answers:
        FAQ replies are stateless, and a small model shown an earlier answer tends to repeat
        it (e.g. mentioning an old order number in an answer about return policy)."""
        convo = ""
        if history and route != "QA":
            turns = [t for t in history[-4:] if t.get("role") == "user"]
            if turns:
                convo = ("Earlier customer messages (background only, do not answer them):\n" +
                         "\n".join(f"- {t['text']}" for t in turns) + "\n\n")
        prompt = (f"{convo}FACT SHEET\n{fact_sheet}\n\nCustomer's current message: {message_en}\n\n"
                  "Write the rewritten reply now.")
        return self.complete(SYSTEM_PROMPT, prompt, max_tokens=180, temperature=0.2)

    @staticmethod
    def check_reply(reply: str, allowed_text: str, required: list[str] | None = None) -> str | None:
        """Grounding guard run on every LLM reply. Returns None if the reply is safe, otherwise a
        short reason. It rejects a reply that (a) contains a number / order ID that appears in
        neither the fact sheet nor the customer's message, or (b) drops a token that must survive
        (order ID, status, the figures in an FAQ answer)."""
        def nums(t):
            return set(re.findall(r"\d+(?:\.\d+)?", t))
        invented = nums(reply) - nums(allowed_text)
        if invented:
            return "invented " + ", ".join(sorted(invented)[:3])
        low = reply.lower()
        missing = [r for r in (required or []) if r and r.lower() not in low]
        if missing:
            return "dropped " + ", ".join(missing[:3])
        if not reply.strip():
            return "empty reply"
        return None

    def translate(self, text: str, src: str, tgt: str) -> str:
        system = ("You are a professional translator. Translate the user's text from "
                  f"{src} to {tgt}. Preserve order IDs (like ORD12345), product names, "
                  "numbers and currency exactly. Output only the translation, nothing else.")
        return self.complete(system, text, max_tokens=500, temperature=0.0)

    def summarize(self, transcript: list[dict]) -> str:
        system = ("You write hand-off notes for human support agents. Given a chat transcript, "
                  "produce: one-line issue summary, key facts (order IDs, products, dates), "
                  "customer sentiment, and recommended next action. Plain text, under 90 words.")
        text = "\n".join(f"{'Customer' if t['role'] == 'user' else 'Bot'}: {t['text']}" for t in transcript)
        return self.complete(system, text, max_tokens=220, temperature=0.2)

    def sentiment(self, text: str) -> str | None:
        system = "Classify the sentiment of this customer feedback. Answer with exactly one word: positive, neutral, or negative."
        try:
            out = self.complete(system, text, max_tokens=5, temperature=0.0).lower()
        except LLMError:
            return None
        for label in ("positive", "negative", "neutral"):
            if label in out:
                return label
        return None

    def ping(self) -> dict:
        """Used by the System page to prove the key/model actually works."""
        if not self.enabled:
            return {"ok": False, "error": "LLM is not configured"}
        t0 = time.time()
        try:
            out = self.complete("Reply with the single word: ready", "ping", max_tokens=8, temperature=0)
            return {"ok": True, "latency_ms": int((time.time() - t0) * 1000), "reply": out[:40]}
        except LLMError as e:
            return {"ok": False, "error": str(e)}
