"""
IECSAP - Speech-to-text.

Two engines, so voice input works on any machine:

1. Browser engine (default, zero install): the Web Speech API built into
   Chrome / Edge / Safari. The microphone audio is recognised by the browser's
   speech service and the transcript is sent to the backend as ordinary text.
   Nothing to configure here; see web/app.js (VoiceInput).

2. Server engine (optional, offline): OpenAI's Whisper, run locally through
   `faster-whisper`. The browser records audio with MediaRecorder and uploads
   it to POST /api/stt. This is what Firefox users and offline deployments use.
   Install with:   pip install faster-whisper
   The first call downloads the chosen model once (default "base", ~140 MB).
   Set WHISPER_MODEL=small for better accuracy on Tamil/Hindi.

Whisper identifies the spoken language by itself, so the language selector can
stay on "auto"; pass a hint only to force one.
"""
from __future__ import annotations

import os
import tempfile
import time

_model = None
_load_error: str | None = None


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _get_model():
    global _model, _load_error
    if _model is None:
        from faster_whisper import WhisperModel
        name = os.environ.get("WHISPER_MODEL", "base")
        try:
            _model = WhisperModel(name, device="cpu", compute_type="int8")
        except Exception as e:  # download blocked, disk full, ...
            _load_error = str(e)
            raise
    return _model


def transcribe_bytes(audio: bytes, suffix: str = ".webm", language: str | None = None) -> dict:
    """Returns {text, language, language_probability, duration_s, latency_ms}."""
    if not available():
        raise RuntimeError("faster-whisper is not installed (pip install faster-whisper)")
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        segments, info = _get_model().transcribe(path, language=language or None, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return {"text": text, "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "duration_s": round(float(info.duration), 2),
            "latency_ms": int((time.time() - t0) * 1000)}


def status() -> dict:
    return {"server_engine": available(), "model": os.environ.get("WHISPER_MODEL", "base") if available() else None,
            "load_error": _load_error}
