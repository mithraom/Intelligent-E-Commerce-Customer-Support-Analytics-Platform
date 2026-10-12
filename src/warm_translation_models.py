"""
IECSAP - one-time download of the translation model(s).

The server never downloads a model while a customer is chatting. It uses a model only if
its weights are COMPLETELY in the local Hugging Face cache; a half-finished download is
ignored. Run this once, with internet, to fetch it:

    py -m pip install transformers sentencepiece torch
    cd src
    py warm_translation_models.py nllb      # ONE ~2.4 GB model: Tamil, Hindi, Telugu, ... both directions
    py warm_translation_models.py es fr     # optional small Marian pairs (~300 MB per direction)

Re-running resumes an interrupted download. Wait for the final line:  READY - ...
(a progress bar reaching 100% is not enough; the line proves the model actually translates).
"""
import sys

from multilingual import NLLB_MODEL, _is_cached

DEFAULT_LANGS = ["es", "fr"]


def warm_nllb():
    from huggingface_hub import snapshot_download
    print(f"Downloading {NLLB_MODEL} (~2.4 GB). Safe to re-run; it resumes.")
    snapshot_download(NLLB_MODEL, allow_patterns=["*.json", "*.model", "*.txt", "pytorch_model.bin"])
    if not _is_cached(NLLB_MODEL):
        print("NOT READY: the weight file is still incomplete. Run this command again.")
        return False
    # prove it works end to end, in both directions
    from multilingual import _machine
    print("Download complete. Loading the model for a test translation (10-30 s)...")
    ta_en, e1 = _machine.translate("உங்கள் திரும்பப் பெறும் கொள்கை என்ன?", "ta", "en")
    en_ta, e2 = _machine.translate("Your order has been delivered.", "en", "ta")
    if not e1 or not e2:
        print("NOT READY: files are present but the model failed to load (see the error above).")
        return False
    print(f"  ta -> en : {ta_en}")
    print(f"  en -> ta : {en_ta}")
    print("READY - restart the server. Tamil/Hindi/etc. now use NLLB-200.")
    return True


def warm_marian(lang: str):
    from huggingface_hub import snapshot_download
    for src, tgt in [(lang, "en"), ("en", lang)]:
        repo = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        print(f"Downloading {repo} ...")
        try:
            snapshot_download(repo)
            print("  done." if _is_cached(repo) else "  incomplete, run again.")
        except Exception as e:
            print(f"  could not download {repo}: {e}")


if __name__ == "__main__":
    args = sys.argv[1:] or DEFAULT_LANGS
    if args == ["nllb"]:
        sys.exit(0 if warm_nllb() else 1)
    for code in args:
        warm_marian(code)
