"""
IECSAP - Concept 5: Multi-lingual NLP (Week 14)
Lets customers write or speak in their own language: detect the language of the
incoming message, translate it to English for the classification / NER / QA
pipeline, then translate the reply back.

What changed in this version (all fixes for real failures seen on the console):

1. No global HF_HUB_OFFLINE switch any more. That switch is read once when
   `transformers` is imported, so it also blocked the Whisper speech model from
   ever downloading inside the same server process. Instead, a model is used
   ONLY if its weights are completely present in the Hugging Face cache
   (`_is_cached`) and it is loaded with local_files_only=True. A half-finished
   download (the ~1.8 GB of 2.4 GB case) is therefore treated as "not cached"
   and the app falls back immediately instead of hanging or crashing.

2. NLLB-200 is loaded ONCE and shared by every language pair. The old code made
   one `pipeline()` per (source, target) pair, i.e. a separate 2.4 GB copy of the
   model in RAM for ta->en, en->ta, hi->en, ...

3. Language ID looks at the Unicode script first (Tamil, Devanagari, Telugu, ...),
   which is exact for non-Latin text, then at English-word hints, then langdetect.
   langdetect alone is unreliable on short text ("My order ORD12345 arrived
   damaged." -> Danish, 99.99%).

4. Translation is sentence by sentence (NLLB degrades on long multi-sentence
   input), cached (FAQ answers are translated once, not on every chat), and
   order IDs are protected: if the translator mangles "ORD10053" it is restored.

Engine order: MarianMT pair (if cached) -> NLLB-200 (if cached) -> LLM (if
configured) -> tiny offline phrasebook (demo phrases only).
"""
import re
import threading

from langdetect import detect, detect_langs
from langdetect.detector_factory import DetectorFactory
DetectorFactory.seed = 42  # reproducible detection

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {
    "ta": "tam_Taml", "hi": "hin_Deva", "te": "tel_Telu", "ml": "mal_Mlym",
    "kn": "kan_Knda", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
    "ur": "urd_Arab", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "pt": "por_Latn", "it": "ita_Latn", "ar": "arb_Arab", "zh-cn": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ru": "rus_Cyrl", "en": "eng_Latn",
}

LANGUAGE_NAMES = {
    "en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu", "ml": "Malayalam",
    "kn": "Kannada", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "ur": "Urdu",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "it": "Italian",
    "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
}

# --------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------
_ENGLISH_HINTS = {
    "i", "me", "my", "mine", "you", "your", "yours", "he", "she", "it", "we", "they", "them",
    "is", "am", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "the", "a", "an", "this", "that", "these", "those", "and", "or", "but", "if", "not", "no",
    "what", "how", "when", "where", "why", "which", "who", "can", "will", "tell", "about",
    "please", "thanks", "thank", "help", "hello", "hi",
    "order", "refund", "return", "exchange", "cancel", "ship", "shipping", "delivery",
    "delivered", "arrived", "damaged", "broken", "charged", "charge", "payment", "account",
    "password", "product", "item", "warranty", "tracking", "track", "invoice", "receipt",
    "policy", "cash", "status",
}

# (first, last) code points -> languages that use the script. One entry = certain.
_SCRIPTS = [
    ((0x0B80, 0x0BFF), ("ta",)),
    ((0x0C00, 0x0C7F), ("te",)),
    ((0x0C80, 0x0CFF), ("kn",)),
    ((0x0D00, 0x0D7F), ("ml",)),
    ((0x0980, 0x09FF), ("bn",)),
    ((0x0A80, 0x0AFF), ("gu",)),
    ((0x0900, 0x097F), ("hi", "mr")),
    ((0x0600, 0x06FF), ("ar", "ur")),
    ((0x3040, 0x30FF), ("ja",)),
    ((0xAC00, 0xD7AF), ("ko",)),
    ((0x4E00, 0x9FFF), ("zh-cn",)),
    ((0x0400, 0x04FF), ("ru",)),
]


def _looks_english(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _ENGLISH_HINTS)
    return hits >= 2 or (len(tokens) <= 4 and hits >= 1)


def _script_languages(text: str):
    """Languages implied by the dominant non-Latin script in `text`, or None."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for (lo, hi), langs in _SCRIPTS:
            if lo <= cp <= hi:
                counts[langs] = counts.get(langs, 0) + 1
                break
    if not counts:
        return None
    langs, n = max(counts.items(), key=lambda kv: kv[1])
    letters = sum(1 for ch in text if ch.isalpha())
    return langs if letters and n / letters >= 0.3 else None


def detect_language(text: str, hint: str | None = None) -> str:
    """Script first (exact), then the speech/recogniser hint, then English-word
    heuristic, then langdetect."""
    hint = (hint or "").lower()
    hint = "" if hint == "auto" else hint
    cands = _script_languages(text)
    if cands:
        if hint in cands:
            return hint
        if len(cands) == 1:
            return cands[0]
        try:  # Devanagari -> hi/mr, Arabic script -> ar/ur
            for guess in detect_langs(text):
                if guess.lang in cands:
                    return guess.lang
        except Exception:
            pass
        return cands[0]
    if hint:
        return hint
    if _looks_english(text):
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"


# --------------------------------------------------------------------------
# Local models (only ever loaded from a COMPLETE cache, never downloaded here)
# --------------------------------------------------------------------------
def _is_cached(repo_id: str) -> bool:
    """True only if the model's weight file is fully downloaded. A partial
    download leaves `*.incomplete` blobs, which try_to_load_from_cache ignores."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            if isinstance(try_to_load_from_cache(repo_id, fname), str):
                return True
        except Exception:
            continue
    return False


class _Seq2Seq:
    """One loaded seq2seq translation model (Marian pair or NLLB)."""

    def __init__(self, repo: str, nllb: bool):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.repo, self.nllb = repo, nllb
        self.tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(repo, local_files_only=True)
        self.model.eval()
        self.lock = threading.Lock()

    def translate(self, text: str, src: str, tgt: str) -> str:
        import torch
        with self.lock:
            kwargs = {"max_new_tokens": 256, "num_beams": 4}
            if self.nllb:
                self.tok.src_lang = NLLB_CODES[src]
                kwargs["forced_bos_token_id"] = self.tok.convert_tokens_to_ids(NLLB_CODES[tgt])
            inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                out = self.model.generate(**inputs, **kwargs)
            return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


_PROTECT = re.compile(r"\bORD\d{5}\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?\u0964])\s+")


class MachineTranslator:
    def __init__(self):
        self._models: dict = {}
        self._failed: set = set()
        self._cache: dict = {}
        self._lock = threading.Lock()
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            self.available = True
        except Exception:
            self.available = False

    # ---- model access
    def _get(self, repo: str, nllb: bool):
        if not self.available or repo in self._failed:
            return None
        with self._lock:
            if repo in self._models:
                return self._models[repo]
            if not _is_cached(repo):
                return None          # not (fully) downloaded: skip, never download here
            try:
                self._models[repo] = _Seq2Seq(repo, nllb)
            except Exception as e:   # corrupt cache, out of memory, ...
                print(f"[translation] could not load {repo}: {e}")
                self._failed.add(repo)
                return None
            return self._models[repo]

    def _engine_for(self, src: str, tgt: str):
        """(model, engine_name) for this pair, or (None, None)."""
        marian = self._get(f"Helsinki-NLP/opus-mt-{src}-{tgt}", nllb=False)
        if marian:
            return marian, "MarianMT"
        if src in NLLB_CODES and tgt in NLLB_CODES:
            nllb = self._get(NLLB_MODEL, nllb=True)
            if nllb:
                return nllb, "NLLB-200"
        return None, None

    def preload(self):
        """Load NLLB now (called in a background thread at server start) so the
        first Tamil message doesn't pay the 10-30 s model-load cost."""
        return self._get(NLLB_MODEL, nllb=True) is not None

    @property
    def nllb_loaded(self) -> bool:
        return NLLB_MODEL in self._models

    # ---- translation
    def translate(self, text: str, src: str, tgt: str):
        """Returns (translated_text, engine_name or None). Sentence by sentence,
        cached, with order IDs protected."""
        key = (text, src, tgt)
        if key in self._cache:
            return self._cache[key]
        model, engine = self._engine_for(src, tgt)
        if model is None:
            return text, None
        out_parts = []
        try:
            for sent in filter(None, (s.strip() for s in _SENT_SPLIT.split(text))):
                out = model.translate(sent, src, tgt)
                for tok in _PROTECT.findall(sent):          # restore a mangled order ID
                    if tok not in out:
                        out = f"{out} ({tok})"
                out_parts.append(out)
        except Exception as e:
            print(f"[translation] {engine} {src}->{tgt} failed: {e}")
            return text, None
        result = (" ".join(out_parts), engine)
        if len(self._cache) > 1000:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result


class DemoPhrasebookTranslator:
    """Offline stand-in used only when no real model is available."""
    ES_TO_EN = {"hola, mi pedido ord30012 llego danado.": "Hello, my order ORD30012 arrived damaged."}

    def to_english(self, text: str, src_lang: str) -> str:
        return text if src_lang == "en" else self.ES_TO_EN.get(text.lower(), text)

    def from_english(self, text: str, tgt_lang: str) -> str:
        if tgt_lang == "en":
            return text
        return f"[ES] {text}" if tgt_lang == "es" else text


_machine = MachineTranslator()
_demo = DemoPhrasebookTranslator()
_llm = None  # set by the server via set_llm(); optional


def set_llm(client) -> None:
    global _llm
    _llm = client


def preload() -> bool:
    return _machine.preload()


def translate_ex(text: str, src: str, tgt: str) -> dict:
    """General translation src -> tgt, pivoting through English when needed.
    Returns {text, ok, engine}."""
    src, tgt = (src or "en").lower(), (tgt or "en").lower()
    if not text.strip() or src == tgt:
        return {"text": text, "ok": True, "engine": "none (same language)"}

    # 1) local models, pivoting through English
    steps = ([(src, "en")] if src != "en" else []) + ([("en", tgt)] if tgt != "en" else [])
    out, engines = text, []
    for s, t in steps:
        out, eng = _machine.translate(out, s, t)
        if not eng:
            engines = None
            break
        engines.append(eng)
    if engines:
        return {"text": out, "ok": True, "engine": " + ".join(dict.fromkeys(engines))}

    # 2) LLM fallback
    if _llm is not None and _llm.enabled:
        try:
            res = _llm.translate(text, LANGUAGE_NAMES.get(src, src), LANGUAGE_NAMES.get(tgt, tgt))
            if res:
                return {"text": res, "ok": True, "engine": f"LLM ({_llm.provider})"}
        except Exception:
            pass

    # 3) offline phrasebook stub (demo phrases only)
    stub = _demo.to_english(text, src) if tgt == "en" else _demo.from_english(text, tgt)
    if stub != text:
        return {"text": stub, "ok": True, "engine": "offline phrasebook (demo)"}
    return {"text": text, "ok": False, "engine": None}


def detect_and_translate_ex(text: str, hint: str | None = None) -> dict:
    lang = detect_language(text, hint)
    if lang == "en":
        return {"english": text, "language": "en", "ok": True, "engine": None}
    res = translate_ex(text, lang, "en")
    return {"english": res["text"], "language": lang, "ok": res["ok"], "engine": res["engine"]}


def detect_and_translate(text: str):
    """Backwards-compatible tuple API: (english_text, language, translated_ok)."""
    r = detect_and_translate_ex(text)
    return r["english"], r["language"], r["ok"]


def translate_to(text: str, tgt_lang: str) -> str:
    return text if tgt_lang == "en" else translate_ex(text, "en", tgt_lang)["text"]


def model_status() -> dict:
    """What the System page shows. `nllb` is True only for a COMPLETE download."""
    marian = []
    try:
        from huggingface_hub import scan_cache_dir
        for r in scan_cache_dir().repos:
            if "opus-mt-" in r.repo_id and _is_cached(r.repo_id):
                marian.append(r.repo_id.split("opus-mt-")[-1])
    except Exception:
        pass
    return {"transformers": _machine.available, "marian_pairs": sorted(marian),
            "nllb": _is_cached(NLLB_MODEL), "nllb_loaded": _machine.nllb_loaded,
            "llm_fallback": bool(_llm and _llm.enabled)}


if __name__ == "__main__":
    print("model status:", model_status())
    for s in ["What is your return policy?", "My order ORD12345 arrived damaged.",
              "உங்கள் திரும்பப் பெறும் கொள்கை என்ன?", "எனக்கு ஆர்டர் ORD10053 பற்றி தெரிந்துகொள்ள வேண்டும்",
              "मेरा ऑर्डर ORD10053 कहाँ है?", "Hola, mi pedido ORD30012 llego danado."]:
        en, lang, ok = detect_and_translate(s)
        print(f"{s!r} -> lang={lang} ok={ok} english={en!r}")
