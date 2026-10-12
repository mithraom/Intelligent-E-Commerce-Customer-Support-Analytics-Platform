"""
IECSAP - Customer feedback loop.

Every bot reply can be rated (1-5) and optionally commented on. Feedback is
stored in SQLite alongside the context that produced the reply (route,
category, language, whether the LLM wrote it, and whether the customer spoke
it), so the dashboard can answer questions like "are Tamil voice messages rated
lower than English text?" and "does LLM phrasing score better than templates?".

Free-text comments are run through a sentiment classifier. The LLM is used when
configured; otherwise a small lexicon scorer, so the feature works offline.
Low-rated replies (<= 2 stars) land in a review queue for a human to inspect.

Only feedback typed/clicked by a real user is stored and shown. There is no demo or seeded data:
any leftover rows from the old seed script (source='seed') are deleted when the store opens.
Each bot reply is also logged in a `messages` table, so (a) a rating still works after the server
restarts, and (b) the dashboard can show a true response rate (rated replies / replies sent).
"""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "feedback.db"

_POS = {"good", "great", "excellent", "helpful", "fast", "quick", "perfect", "thanks", "thank",
        "clear", "love", "amazing", "resolved", "easy", "nice", "awesome", "accurate", "useful"}
_NEG = {"bad", "terrible", "useless", "slow", "wrong", "confusing", "unhelpful", "worst", "poor",
        "rude", "irrelevant", "incorrect", "broken", "annoying", "waste", "failed", "ignored", "unclear"}
_NEGATORS = {"not", "no", "never", "isn't", "wasn't", "didn't", "don't", "cannot", "can't"}


def lexicon_sentiment(text: str) -> str:
    tokens = re.findall(r"[a-z']+", text.lower())
    score = 0
    for i, t in enumerate(tokens):
        sign = -1 if i > 0 and tokens[i - 1] in _NEGATORS else 1
        if t in _POS:
            score += sign
        elif t in _NEG:
            score -= sign
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


class FeedbackStore:
    def __init__(self, path: Path | None = None, llm=None):
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY, ts REAL, message_id TEXT, rating INTEGER,
                comment TEXT, tags TEXT, comment_sentiment TEXT,
                route TEXT, category TEXT, language TEXT, input_mode TEXT,
                llm_used INTEGER, user_message TEXT, bot_reply TEXT, source TEXT DEFAULT 'live')""")
            c.execute("""CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY, ts REAL, route TEXT, category TEXT, language TEXT,
                input_mode TEXT, llm_used INTEGER, user_message TEXT, bot_reply TEXT)""")
            c.execute("DELETE FROM feedback WHERE source != 'live'")   # purge old demo/seed rows

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    # ------------------------------------------------------------ message log
    def log_message(self, message_id: str, ctx: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?,?,?,?)", (
                message_id, time.time(), ctx.get("route"), ctx.get("category"), ctx.get("language"),
                ctx.get("input_mode", "text"), int(bool(ctx.get("llm_used"))),
                (ctx.get("user_message") or "")[:500], (ctx.get("bot_reply") or "")[:800]))

    def get_message(self, message_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------------ write
    def add(self, message_id: str, rating: int, comment: str = "", tags: list[str] | None = None,
            context: dict | None = None, source: str = "live") -> dict:
        if not 1 <= int(rating) <= 5:
            raise ValueError("rating must be between 1 and 5")
        context = context or {}
        comment = (comment or "").strip()[:1000]
        sentiment = None
        if comment:
            if self.llm and self.llm.enabled:
                sentiment = self.llm.sentiment(comment)
            sentiment = sentiment or lexicon_sentiment(comment)
        row = {
            "id": uuid.uuid4().hex[:12], "ts": time.time(), "message_id": message_id,
            "rating": int(rating), "comment": comment, "tags": ",".join(tags or []),
            "comment_sentiment": sentiment, "route": context.get("route"),
            "category": context.get("category"), "language": context.get("language"),
            "input_mode": context.get("input_mode", "text"), "llm_used": int(bool(context.get("llm_used"))),
            "user_message": (context.get("user_message") or "")[:500],
            "bot_reply": (context.get("bot_reply") or "")[:800], "source": source,
        }
        with self._conn() as c:
            # one rating per reply: re-rating replaces the earlier one
            c.execute("DELETE FROM feedback WHERE message_id = ? AND source = ?", (message_id, source))
            c.execute(f"INSERT INTO feedback ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                      list(row.values()))
        return row

    # ------------------------------------------------------------------- read
    def summary(self) -> dict:
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM feedback WHERE source = 'live' ORDER BY ts DESC")]
            replies = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n = len(rows)
        if n == 0:
            return {"total": 0, "average": None, "csat": None, "distribution": {str(i): 0 for i in range(1, 6)},
                    "by_route": [], "by_language": [], "by_mode": [], "llm_vs_template": [],
                    "sentiment": {}, "tags": {}, "recent": [], "review_queue": [], "demo_entries": 0,
                    "replies_total": replies, "response_rate": 0}

        def group(key, label=None):
            buckets: dict = {}
            for r in rows:
                k = r[key] if key != "llm_used" else ("LLM-composed" if r[key] else "Template")
                k = k or "n/a"
                buckets.setdefault(k, []).append(r["rating"])
            return sorted(({"name": k, "count": len(v), "average": round(sum(v) / len(v), 2)}
                           for k, v in buckets.items()), key=lambda d: -d["count"])

        tag_counts: dict = {}
        for r in rows:
            for t in filter(None, (r["tags"] or "").split(",")):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        sentiments: dict = {}
        for r in rows:
            if r["comment_sentiment"]:
                sentiments[r["comment_sentiment"]] = sentiments.get(r["comment_sentiment"], 0) + 1

        def public(r):
            return {k: r[k] for k in ("id", "ts", "rating", "comment", "tags", "comment_sentiment", "route",
                                      "category", "language", "input_mode", "llm_used", "user_message",
                                      "bot_reply", "source")}

        return {
            "total": n,
            "average": round(sum(r["rating"] for r in rows) / n, 2),
            "csat": round(100 * sum(1 for r in rows if r["rating"] >= 4) / n),
            "distribution": {str(i): sum(1 for r in rows if r["rating"] == i) for i in range(1, 6)},
            "by_route": group("route"), "by_language": group("language"),
            "by_mode": group("input_mode"), "llm_vs_template": group("llm_used"),
            "sentiment": sentiments, "tags": tag_counts,
            "recent": [public(r) for r in rows if r["comment"]][:8],
            "review_queue": [public(r) for r in rows if r["rating"] <= 2][:6],
            "demo_entries": 0,
            "replies_total": replies,
            "response_rate": round(100 * min(n, replies) / replies) if replies else 0,
        }

    def export_csv(self) -> str:
        import csv, io
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM feedback WHERE source = 'live' ORDER BY ts")]
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return buf.getvalue()


if __name__ == "__main__":
    store = FeedbackStore(Path("/tmp/fb_test.db"))
    store.add("m1", 5, "Really helpful and fast", ["accurate"], {"route": "TICKET", "language": "en"})
    store.add("m2", 1, "This was not helpful", ["wrong answer"], {"route": "QA", "language": "ta", "input_mode": "voice"})
    print(store.summary()["sentiment"], store.summary()["average"])
