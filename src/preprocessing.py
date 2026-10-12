"""
IECSAP - Preprocessing utilities shared across modules (Week 11).

IMPORTANT (fix for a real failure mode seen on Windows machines behind a
firewall/proxy): this module NEVER calls nltk.download() at import time.
Earlier versions tried to auto-download 'stopwords'/'wordnet' on first run,
which is fine with open internet access but HANGS indefinitely on a
network that blocks nltk.org -- and a hung urlopen() raises
KeyboardInterrupt (not Exception) when the user Ctrl-C's it, which slips
past a bare `except Exception`. To make the pipeline reliable in any
environment (offline, corporate network, restricted lab machine), this
module ships built-in fallbacks and only ever *tries* to use NLTK data
that is already present locally -- it never attempts to fetch it.
"""
import re
import string
import unicodedata

# ---------------------------------------------------------------------------
# Stopwords: try NLTK's corpus if it's already installed locally; otherwise
# use a bundled static list. Either way, NO network call is ever made.
# ---------------------------------------------------------------------------
_FALLBACK_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","you're","you've",
    "you'll","you'd","your","yours","yourself","yourselves","he","him","his",
    "himself","she","she's","her","hers","herself","it","it's","its","itself",
    "they","them","their","theirs","themselves","what","which","who","whom",
    "this","that","that'll","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a",
    "an","the","and","but","if","or","because","as","until","while","of","at",
    "by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other",
    "some","such","no","nor","not","only","own","same","so","than","too","very",
    "s","t","can","will","just","don","don't","should","should've","now","d",
    "ll","m","o","re","ve","y","ain","please","also",
}

try:
    from nltk.corpus import stopwords as _nltk_stopwords
    STOPWORDS = set(_nltk_stopwords.words("english"))
except Exception:
    STOPWORDS = _FALLBACK_STOPWORDS

# ---------------------------------------------------------------------------
# Lemmatization: try NLTK's WordNetLemmatizer if the 'wordnet' corpus is
# already present locally; otherwise fall back to a lightweight rule-based
# suffix stripper. It's less linguistically precise than WordNet but needs
# zero external data and is more than adequate for TF-IDF style features,
# where the goal is just to fold minor inflections together.
# ---------------------------------------------------------------------------
def _build_lemmatizer():
    try:
        from nltk.corpus import wordnet  # noqa: F401  (forces a lookup of local data)
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("test")  # touch the corpus now so failures happen here, at import time
        return lem.lemmatize
    except Exception:
        return _simple_lemmatize


_SUFFIXES = [("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("es", ""), ("s", "")]


def _simple_lemmatize(word: str) -> str:
    if len(word) <= 3:
        return word
    for suf, repl in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)] + repl
    return word


_lemmatize_fn = _build_lemmatizer()

ORDER_ID_RE = re.compile(r"\bORD\d{5}\b")
AMOUNT_RE = re.compile(r"\$\d+(?:\.\d{1,2})?")
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")


# ---------------------------------------------------------------------------
# Order-ID finding. Customers rarely type the ID exactly as stored ("ORD10053"):
# they write "10053", "order 10053", "ord-10053", "#10053" -- and in Tamil/Hindi
# the ID sits inside non-English text. This finder is language-independent and
# returns canonical IDs ("ORD10053") that can be looked up in orders.csv.
# ---------------------------------------------------------------------------
_ORD_PREFIXED = re.compile(r"(?<![A-Za-z])ORD[\s\-_#:.]*(\d{5})(?!\d)", re.I)
_ORDER_WORD = re.compile(r"\borders?\b[^\d\n]{0,25}?(?<![\d$])(\d{5})(?![\d.,/])", re.I)
_HASH = re.compile(r"#\s*(\d{5})(?!\d)")
_BARE = re.compile(r"(?<![\d$.,/])(\d{5})(?![\d.,/]?\d)")


def _ascii_digits(text: str) -> str:
    """Turn Tamil/Devanagari/Arabic-Indic digits into 0-9 so '\u0be7\u0be6\u0be6\u0beb\u0be9' still matches."""
    def conv(m):
        try:
            return str(unicodedata.digit(m.group()))
        except (TypeError, ValueError):
            return m.group()
    return re.sub(r"\d", conv, text)


def find_order_ids(text: str, known=None) -> list:
    """Return canonical order IDs (e.g. ['ORD10053']) found in `text`, in order.

    Explicit forms (ORD10053, 'order 10053', '#10053') are always accepted, so a
    typo'd/unknown ID is reported as "not found" rather than silently ignored.
    A bare 5-digit number is accepted only if it is a real order in `known`,
    which stops phone-number-like digits being mistaken for orders."""
    text = _ascii_digits(text or "")
    found = []

    def add(d):
        oid = "ORD" + d
        if oid not in found:
            found.append(oid)

    spans = []
    for rx in (_ORD_PREFIXED, _ORDER_WORD, _HASH):
        for m in rx.finditer(text):
            add(m.group(1))
            spans.append(m.span(1))
    if known:
        for m in _BARE.finditer(text):
            if any(a <= m.start(1) < b for a, b in spans):
                continue
            if "ORD" + m.group(1) in known:
                add(m.group(1))
    return found


def clean_text(text: str, lemmatize: bool = True) -> str:
    """Lowercase, strip punctuation (but keep placeholders for entities),
    remove stopwords, and lemmatize. Used for classification / QA (bag-of-words style)."""
    text = text.lower()
    text = re.sub(r"[^\w\s$/]", " ", text)  # keep $ and / so amounts/dates survive tokenization upstream
    tokens = text.split()
    tokens = [t.strip(string.punctuation) for t in tokens]
    tokens = [t for t in tokens if t and t not in STOPWORDS]
    if lemmatize:
        tokens = [_lemmatize_fn(t) for t in tokens]
    return " ".join(tokens)


def extract_regex_entities(text: str) -> dict:
    """Fast, high-precision regex pass for structured entities (order IDs, money, dates)
    used to complement spaCy's statistical NER in ner.py."""
    return {
        "ORDER_ID": find_order_ids(text),
        "AMOUNT": AMOUNT_RE.findall(text),
        "DATE": DATE_RE.findall(text),
    }


if __name__ == "__main__":
    sample = "I was charged twice for order ORD12345, please refund $45.00 on 12/6/2026."
    print("using lemmatizer:", "NLTK WordNet" if _lemmatize_fn is not _simple_lemmatize else "built-in fallback")
    print("cleaned:", clean_text(sample))
    print("entities:", extract_regex_entities(sample))
